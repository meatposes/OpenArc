import asyncio
import base64
import gc
import io
import logging
from typing import Any, AsyncIterator, Dict, Union

import librosa
import numpy as np
import openvino_genai
from openvino_genai import WhisperPipeline

from src.engine.ov_genai.whisper_languages import LANGUAGES, from_token, to_code

from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig
from src.server.schemas.modeling.contract_whisper import OVGenAI_WhisperGenConfig

logger = logging.getLogger(__name__)


class OVGenAI_Whisper:
    def __init__(self, load_config: ModelLoadConfig):
        
        self.load_config = load_config
        pass

    def prepare_audio(self, gen_config: OVGenAI_WhisperGenConfig) -> list[float]:
        """
        Prepare audio inputs from base64 string for the Whisper pipeline.
        """

        audio_bytes = base64.b64decode(gen_config.audio_base64)
        
        audio_buffer = io.BytesIO(audio_bytes)
        
        audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)

        return audio.astype(np.float32).tolist()

    def _generation_config(self, gen_config: OVGenAI_WhisperGenConfig):
        cfg = self.whisper_model.get_generation_config()
        code = to_code(gen_config.language)       # validated by the route already
        if code and cfg.is_multilingual:
            cfg.language = f"<|{code}|>"
        elif code and code != "en":
            logger.warning(f"[{self.load_config.model_name}] English-only model; ignoring language={code}")
        cfg.task = gen_config.task
        cfg.return_timestamps = bool(gen_config.return_timestamps or gen_config.word_timestamps)
        if gen_config.word_timestamps:
            if self.word_timestamps_enabled:
                cfg.word_timestamps = True
            else:
                logger.warning(f"[{self.load_config.model_name}] word timestamps not available for this model")
        return cfg

    async def transcribe(self, gen_config: OVGenAI_WhisperGenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Run transcription on a given base64 encoded audio and return texts with metrics.

        Yields in order: metrics (dict), transcribed_text (str), details (dict with
        language, duration, segments, words and score).
        """
        audio_list = await asyncio.to_thread(self.prepare_audio, gen_config)
        cfg = self._generation_config(gen_config)

        result = await asyncio.to_thread(self.whisper_model.generate, audio_list, cfg)

        # Collect transcription and metrics
        transcription = result.texts
        perf_metrics = getattr(result, "perf_metrics", None)
        metrics_dict = self.collect_metrics(perf_metrics) if perf_metrics is not None else {}

        final_text = ' '.join(transcription) if isinstance(transcription, list) else transcription

        yield metrics_dict
        yield final_text
        yield self._details(result, gen_config, len(audio_list) / 16000)

    def _details(self, result, gen_config: OVGenAI_WhisperGenConfig, duration: float) -> Dict[str, Any]:
        """The OpenAI verbose_json fields: detected language, segments, words; plus the score."""
        langs = getattr(result, "languages", None) or []
        code = to_code(gen_config.language) or from_token(langs[0] if langs else None)
        if code is None and not self.whisper_model.get_generation_config().is_multilingual:
            code = "en"
        segments = [{"id": i, "start": round(c.start_ts, 3), "end": round(c.end_ts, 3), "text": c.text.strip()}
                    for i, c in enumerate(getattr(result, "chunks", None) or [])]
        words = [{"word": w.word.strip(), "start": round(w.start_ts, 3), "end": round(w.end_ts, 3)}
                 for w in (getattr(result, "words", None) or [])]
        scores = list(getattr(result, "scores", None) or [])
        return {
            "language": LANGUAGES.get(code, code) if code else None,
            "language_code": code,
            "duration": round(duration, 3),
            "segments": segments,
            "words": words,
            "score": round(float(scores[0]), 4) if scores else None,
        }

    def collect_metrics(self, perf_metrics) -> dict:
        """
        Collect key performance metrics from a Whisper perf_metrics object.
        """
        metrics = {
            "num_generated_tokens": perf_metrics.get_num_generated_tokens(),
            "throughput_tokens_per_sec": round(perf_metrics.get_throughput().mean, 4),
            "ttft_s": round(perf_metrics.get_ttft().mean / 1000, 4),
            "load_time_s": round(perf_metrics.get_load_time() / 1000, 4),
            "generate_duration_s": round(perf_metrics.get_generate_duration().mean / 1000, 4),
            "features_extraction_duration_ms": round(perf_metrics.get_features_extraction_duration().mean, 4),
        }

        return metrics

    def load_model(self, loader: ModelLoadConfig) -> None:
        """
        Load (or reload) a Whisper model into a pipeline for the given device.
        """
        pipeline_kwargs = {**(loader.runtime_config or {})}
        if loader.cache_dir:
            pipeline_kwargs['CACHE_DIR'] = loader.cache_dir

        # Word timestamps need extra decoder outputs compiled in, so they are a load-time choice.
        # Try with them; models exported without alignment information fall back to plain.
        self.word_timestamps_enabled = False
        want_words = pipeline_kwargs.pop("word_timestamps", True)
        if want_words:
            try:
                self.whisper_model = WhisperPipeline(loader.model_path, loader.device, word_timestamps=True, **pipeline_kwargs)
                self.word_timestamps_enabled = True
                return
            except Exception as e:  # noqa: BLE001 - not every export supports it
                logger.info(f"[{loader.model_name}] word timestamps unavailable ({e}); loading without")
        self.whisper_model = WhisperPipeline(
            loader.model_path,
            loader.device,
            **pipeline_kwargs
        )

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None

        gc.collect()
        logger.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed


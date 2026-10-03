"""Speaker diarization with a pyannote pipeline whose two networks run on OpenVINO.

The pyannote pipeline (sliding windows, aggregation, VBx clustering) stays as Python on CPU torch.
Only the segmentation model and the WeSpeaker embedding model are swapped for compiled OpenVINO
models, produced by ``src.engine.openvino.pyannote.convert``. Model directory layout:

    config.yaml, segmentation/, embedding/, plda/      the pyannote pipeline (e.g. community-1)
    openvino/{segmentation,fbank,resnet}_model.xml  the converted networks

On GPU the inputs are reshaped to a static batch (padded when a batch comes up short), so each
model compiles once. Segmentation and fbank run in f32; the ResNet runs at the device's default
precision (f16 on Arc), which is where the time goes and matches f32 output.
"""
import asyncio
import base64
import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import openvino as ov
import torch

from src.engine.audio import decode_audio
from src.server.model_registry import ModelRegistry
from src.server.schemas.modeling.contract_diarization import OV_DiarizationGenConfig
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CPU_THREADS = 4
MIN_VOICEPRINT_SECONDS = 3.0


class OV_PyannoteDiarization:
    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.pipeline = None
        self._seg = None
        self._fbank = None
        self._resnet = None

    def load_model(self, load_config: ModelLoadConfig) -> None:
        from pyannote.audio import Pipeline

        model_path = Path(load_config.model_path)
        self._device = load_config.device
        self._gpu = str(self._device).upper().startswith("GPU")

        pipeline = Pipeline.from_pretrained(str(model_path))
        self._samples = int(pipeline._segmentation.duration * SAMPLE_RATE)
        self._seg_batch = pipeline.segmentation_batch_size
        self._emb_batch = pipeline.embedding_batch_size

        core = ov.Core()
        if load_config.cache_dir:
            core.set_property({"CACHE_DIR": load_config.cache_dir})
        if load_config.runtime_config:
            core.set_property(load_config.runtime_config)
        ov_dir = model_path / "openvino"
        seg = core.read_model(ov_dir / "segmentation_model.xml")
        fbank = core.read_model(ov_dir / "fbank_model.xml")
        resnet = core.read_model(ov_dir / "resnet_model.xml")
        self._seg_frames = seg.output(0).get_partial_shape()[1].get_length()
        self._fbank_shape = [d.get_length() for d in list(fbank.output(0).get_partial_shape())[1:]]

        if self._gpu:
            seg.reshape({0: [self._seg_batch, 1, self._samples]})
            fbank.reshape({0: [self._emb_batch, 1, self._samples]})
            resnet.reshape({0: [self._emb_batch, *self._fbank_shape], 1: [self._emb_batch, self._seg_frames]})
            f32 = {"INFERENCE_PRECISION_HINT": "f32"}
            self._seg = core.compile_model(seg, self._device, f32)
            self._fbank = core.compile_model(fbank, self._device, f32)   # audio*32768 squared overflows f16
            self._resnet = core.compile_model(resnet, self._device)
        else:
            cpu = {"INFERENCE_NUM_THREADS": CPU_THREADS}
            self._seg = core.compile_model(seg, "CPU", cpu)
            self._fbank = core.compile_model(fbank, "CPU", cpu)
            self._resnet = core.compile_model(resnet, "CPU", cpu)

        seg_model = pipeline._segmentation.model
        emb_model = pipeline._embedding.model_
        seg_model.forward = lambda x, **_: torch.from_numpy(self._run(self._segment, self._seg_batch, x.float().numpy()))
        emb_model.forward = lambda x, weights=None: torch.from_numpy(
            self._run(self._embed, self._emb_batch, x.float().numpy(), self._weights(x.shape[0], weights)))
        self.pipeline = pipeline

        # compile and warm every kernel before the first request
        self._diarize_array(np.random.default_rng(0).standard_normal(SAMPLE_RATE * 30).astype(np.float32) * 0.01)
        logger.info(f"[{load_config.model_name}] pyannote diarization loaded on {self._device}")

    def _segment(self, x: np.ndarray) -> np.ndarray:
        return self._seg([x])[0]

    def _embed(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self._resnet([self._fbank([x])[0], weights])[0]

    def _weights(self, n: int, weights: Optional[torch.Tensor]) -> np.ndarray:
        if weights is None:
            return np.ones((n, self._seg_frames), np.float32)
        return weights.float().numpy()

    def _run(self, fn, batch: int, *xs: np.ndarray) -> np.ndarray:
        """Run fn on inputs, padding the batch up to the static GPU batch size and trimming back."""
        n = xs[0].shape[0]
        if not self._gpu:
            return fn(*xs)
        out = []
        for i in range(0, n, batch):
            part = [x[i:i + batch] for x in xs]
            m = part[0].shape[0]
            if m < batch:
                part = [np.concatenate([p, np.zeros((batch - m,) + p.shape[1:], p.dtype)]) for p in part]
            out.append(fn(*part)[:m])
        return np.concatenate(out)

    @staticmethod
    def _load_audio(audio_base64: str) -> np.ndarray:
        return decode_audio(base64.b64decode(audio_base64), SAMPLE_RATE)

    def _diarize_array(self, audio: np.ndarray, **speakers) -> Dict[str, Any]:
        t = time.perf_counter()
        out = self.pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": SAMPLE_RATE}, **speakers)
        took = time.perf_counter() - t
        labels = out.speaker_diarization.labels()
        talk = dict.fromkeys(labels, 0.0)
        turns = []
        for seg, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True):
            turns.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk})
            talk[spk] = talk.get(spk, 0.0) + seg.duration
        emb = out.speaker_embeddings
        spk_out = []
        for i, s in enumerate(labels):
            v = None if emb is None or i >= len(emb) or not np.all(np.isfinite(emb[i])) else [round(float(x), 6) for x in emb[i]]
            spk_out.append({"id": s, "seconds": round(talk.get(s, 0.0), 2), "embedding": v})
        duration = len(audio) / SAMPLE_RATE
        return {
            "duration": round(duration, 3),
            "turns": turns,
            "speakers": spk_out,
            "metrics": {"audio_duration_sec": round(duration, 3), "processing_sec": round(took, 3),
                        "realtime_factor": round(duration / took, 1) if took > 0 else None},
        }

    def _voiceprint_array(self, audio: np.ndarray) -> Dict[str, Any]:
        """One embedding for single-speaker speech: fixed windows, zero padded, padding masked out
        through the pooling weights, averaged by the amount of real audio in each window."""
        n = self._samples
        windows, weights, lengths = [], [], []
        for i in range(0, len(audio), n):
            c = audio[i:i + n]
            if len(c) < SAMPLE_RATE * MIN_VOICEPRINT_SECONDS:
                continue
            w = np.zeros((1, n), np.float32); w[0, :len(c)] = c
            m = np.zeros(self._seg_frames, np.float32); m[:int(self._seg_frames * len(c) / n)] = 1
            windows.append(w); weights.append(m); lengths.append(len(c))
        if not windows:
            raise ValueError(f"Need at least {MIN_VOICEPRINT_SECONDS:.0f} s of speech for a voiceprint")
        vs = self._run(self._embed, self._emb_batch, np.stack(windows), np.stack(weights))
        v = (vs * np.array(lengths, np.float32)[:, None]).sum(0) / sum(lengths)
        return {"duration": round(len(audio) / SAMPLE_RATE, 3), "embedding": [round(float(x), 6) for x in v]}

    async def diarize(self, gen_config: OV_DiarizationGenConfig) -> Dict[str, Any]:
        audio = self._load_audio(gen_config.audio_base64)
        if gen_config.task == "voiceprint":
            return await asyncio.to_thread(self._voiceprint_array, audio)
        speakers = {k: v for k, v in (("num_speakers", gen_config.num_speakers),
                                      ("min_speakers", gen_config.min_speakers),
                                      ("max_speakers", gen_config.max_speakers)) if v}
        return await asyncio.to_thread(self._diarize_array, audio, **speakers)

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        removed = await registry.register_unload(model_name)
        self.pipeline = None
        self._seg = self._fbank = self._resnet = None
        gc.collect()
        logger.info(f"[{self.load_config.model_name}] unloaded and memory cleaned up")
        return removed

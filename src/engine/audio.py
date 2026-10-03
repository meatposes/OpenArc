"""Audio decoding shared by the speech engines (Whisper, Qwen3-ASR, diarization).

Uploads are decoded with torchcodec (FFmpeg), which reads every common container and codec:
WAV, FLAC, MP3, Ogg/Opus, WebM (browser and Android recordings), M4A/MP4 AAC (Teams
recordings, iPhone and Mac voice memos). FFmpeg also downmixes and resamples in the same pass.

If torchcodec or its FFmpeg libraries are missing (e.g. a dev venv), decoding falls back to
libsndfile, which reads WAV/FLAC/MP3/Ogg but not WebM or M4A.
"""
import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_UNSET = object()
_audio_decoder = _UNSET


class AudioDecodeError(ValueError):
    """The request carried audio that cannot be decoded: a client error, not a model failure."""


def _probe_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _torchcodec_decoder():
    """AudioDecoder if torchcodec can actually decode here, else None (checked once).

    Importing torchcodec succeeds even when its FFmpeg libraries are missing; they only load on
    the first decode. So the check decodes a tiny WAV rather than trusting the import.
    """
    global _audio_decoder
    if _audio_decoder is _UNSET:
        try:
            from torchcodec.decoders import AudioDecoder
            AudioDecoder(_probe_wav(), sample_rate=SAMPLE_RATE, num_channels=1).get_all_samples()
            _audio_decoder = AudioDecoder
        except Exception as e:   # not installed, or its FFmpeg libraries are missing
            reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            logger.warning(f"torchcodec unusable ({reason}); decoding with libsndfile, no WebM/M4A support")
            _audio_decoder = None
    return _audio_decoder


def decode_audio(raw: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file's bytes to mono float32 at sample_rate."""
    if not raw:
        raise AudioDecodeError("Unreadable audio: empty payload")
    decoder = _torchcodec_decoder()
    if decoder is not None:
        try:
            samples = decoder(raw, sample_rate=sample_rate, num_channels=1).get_all_samples()
        except Exception as e:
            # FFmpeg's message carries torchcodec's build paths; log it, keep the client's short
            logger.info(f"audio decode failed: {e}")
            raise AudioDecodeError("Unreadable audio: not a supported audio file, or the file is corrupt") from e
        return np.ascontiguousarray(samples.data[0].numpy(), dtype=np.float32)

    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioDecodeError(f"Unreadable audio: {e}") from e
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)

import io
from pathlib import Path

import numpy as np
import pytest  # type: ignore[import]
import soundfile as sf

import src.engine.audio as audio_module
from src.engine.audio import AudioDecodeError, decode_audio

FIXTURES = Path(__file__).parent / "fixtures"


def _have_torchcodec() -> bool:
    try:
        return audio_module._torchcodec_decoder() is not None
    except Exception:
        return False


needs_torchcodec = pytest.mark.skipif(not _have_torchcodec(), reason="torchcodec/FFmpeg not installed")


@pytest.fixture
def libsndfile_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_module, "_audio_decoder", None)


def _stereo_wav_48k(seconds: float = 2.0) -> bytes:
    t = np.arange(int(48000 * seconds)) / 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone, tone], 1).astype(np.float32), 48000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@needs_torchcodec
@pytest.mark.parametrize("name", ["tone_48k.webm", "tone_stereo_48k.m4a"])
def test_decodes_webm_and_m4a_to_16k_mono(name: str) -> None:
    audio = decode_audio((FIXTURES / name).read_bytes())
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) / 16000 - 4.0) < 0.1           # 4 s tones, resampled from 48 kHz
    assert 0.05 < np.abs(audio).max() <= 1.0              # lavfi sine is generated at amplitude 1/8


@needs_torchcodec
def test_torchcodec_downmixes_and_resamples_wav() -> None:
    audio = decode_audio(_stereo_wav_48k())
    assert abs(len(audio) - 32000) <= 2


def test_libsndfile_fallback_downmixes_and_resamples(libsndfile_only: None) -> None:
    audio = decode_audio(_stereo_wav_48k())
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 32000) <= 2


@pytest.mark.parametrize("payload", [b"", b"definitely not audio" * 50])
def test_garbage_raises_decode_error(payload: bytes) -> None:
    with pytest.raises(AudioDecodeError, match="Unreadable audio"):
        decode_audio(payload)


def test_garbage_raises_decode_error_on_fallback(libsndfile_only: None) -> None:
    with pytest.raises(AudioDecodeError, match="Unreadable audio"):
        decode_audio(b"definitely not audio" * 50)


def test_torchcodec_without_ffmpeg_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """torchcodec imports fine without FFmpeg and only fails on decode; that must mean fallback."""
    import sys
    import types

    class BrokenDecoder:
        def __init__(self, *a, **k):
            raise OSError("Could not load this library: libtorchcodec_core4.so")

    fake = types.ModuleType("torchcodec.decoders")
    fake.AudioDecoder = BrokenDecoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake)
    monkeypatch.setattr(audio_module, "_audio_decoder", audio_module._UNSET)
    audio = decode_audio(_stereo_wav_48k())
    assert audio_module._audio_decoder is None
    assert abs(len(audio) - 32000) <= 2


def test_decode_error_is_a_value_error() -> None:
    # routes map ValueError to HTTP 400
    assert issubclass(AudioDecodeError, ValueError)

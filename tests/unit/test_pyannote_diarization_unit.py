import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest  # type: ignore[import]
import soundfile as sf

from src.engine.openvino.pyannote.diarization import SAMPLE_RATE, OV_PyannoteDiarization
from src.server.schemas.modeling.contract_diarization import OV_DiarizationGenConfig
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType


@pytest.fixture
def diar() -> OV_PyannoteDiarization:
    d = OV_PyannoteDiarization(ModelLoadConfig(
        model_path="/models/pyannote-community-1-ov", model_name="diar", model_type=ModelType.DIARIZATION,
        engine=EngineType.OPENVINO, device="GPU.0"))
    d._gpu = True
    d._samples = 160000
    d._seg_frames = 589
    d._emb_batch = 4
    d._seg_batch = 4
    return d


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def test_run_pads_to_static_batch_and_trims(diar: OV_PyannoteDiarization) -> None:
    seen = []
    def fn(x, w):
        seen.append(x.shape[0])
        return x[:, 0, :3] + w[:, :3]
    x = np.arange(6 * 160000, dtype=np.float32).reshape(6, 1, 160000)
    w = np.ones((6, 589), np.float32)
    out = diar._run(fn, 4, x, w)
    assert seen == [4, 4]                       # 6 inputs -> one full batch + one padded batch
    assert out.shape == (6, 3)
    np.testing.assert_array_equal(out, x[:, 0, :3] + 1)


def test_run_on_cpu_passes_through(diar: OV_PyannoteDiarization) -> None:
    diar._gpu = False
    fn = MagicMock(return_value=np.zeros((3, 2)))
    diar._run(fn, 4, np.zeros((3, 1, 10), np.float32))
    assert fn.call_args.args[0].shape == (3, 1, 10)


def test_load_audio_downmixes_and_resamples() -> None:
    t = np.arange(48000) / 48000
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], 1).astype(np.float32) * 0.5
    audio = OV_PyannoteDiarization._load_audio(_wav_b64(stereo, 48000))
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - SAMPLE_RATE) <= 1


def test_load_audio_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Unreadable audio"):
        OV_PyannoteDiarization._load_audio(base64.b64encode(b"not audio").decode())


def test_voiceprint_masks_padding_and_weights_by_length(diar: OV_PyannoteDiarization) -> None:
    calls = []
    def embed(x, w):
        calls.append((x.copy(), w.copy()))
        return np.stack([np.full(4, float(m.sum())) for m in w])   # embedding encodes mask length
    diar._embed = embed
    audio = np.ones(160000 + 80000, np.float32)                     # 10 s + 5 s
    out = diar._voiceprint_array(audio)
    x, w = calls[0]
    assert x.shape == (4, 1, 160000)                                # padded to the static batch
    assert w[0].sum() == 589 and w[1].sum() == int(589 * 0.5)       # second window half masked
    assert np.all(x[1, 0, 80000:] == 0)
    expected = (589 * 160000 + int(589 * 0.5) * 80000) / 240000
    assert out["embedding"][0] == pytest.approx(expected, rel=1e-6)


def test_voiceprint_rejects_short_audio(diar: OV_PyannoteDiarization) -> None:
    with pytest.raises(ValueError, match="at least 3 s"):
        diar._voiceprint_array(np.ones(SAMPLE_RATE * 2, np.float32))


def test_diarize_formats_turns_and_speakers(diar: OV_PyannoteDiarization) -> None:
    class Seg:
        def __init__(self, s, e): self.start, self.end, self.duration = s, e, e - s
    excl = MagicMock()
    excl.itertracks.return_value = [(Seg(0.0, 1.5), None, "SPEAKER_00"), (Seg(1.5, 2.0), None, "SPEAKER_01"),
                                    (Seg(2.0, 3.0), None, "SPEAKER_00")]
    full = MagicMock(); full.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
    emb = np.array([[0.1, 0.2], [np.nan, np.nan]], np.float32)
    diar.pipeline = MagicMock(return_value=SimpleNamespace(
        speaker_diarization=full, exclusive_speaker_diarization=excl, speaker_embeddings=emb))
    out = diar._diarize_array(np.zeros(SAMPLE_RATE * 3, np.float32), num_speakers=2)
    assert diar.pipeline.call_args.kwargs == {"num_speakers": 2}
    assert out["duration"] == 3.0
    assert [t["speaker"] for t in out["turns"]] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert out["speakers"][0] == {"id": "SPEAKER_00", "seconds": 2.5, "embedding": [pytest.approx(0.1), pytest.approx(0.2)]}
    assert out["speakers"][1]["embedding"] is None                  # NaN centroid is not passed on


def test_diarize_routes_voiceprint_task(diar: OV_PyannoteDiarization) -> None:
    diar._voiceprint_array = MagicMock(return_value={"embedding": [1.0]})
    diar._diarize_array = MagicMock()
    cfg = OV_DiarizationGenConfig(audio_base64=_wav_b64(np.zeros(SAMPLE_RATE * 4, np.float32), SAMPLE_RATE), task="voiceprint")
    assert asyncio.run(diar.diarize(cfg)) == {"embedding": [1.0]}
    diar._diarize_array.assert_not_called()

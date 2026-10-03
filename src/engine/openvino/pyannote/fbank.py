"""Batched Kaldi fbank for pyannote's WeSpeaker embedding model.

pyannote computes fbank with ``torch.vmap(torchaudio.compliance.kaldi.fbank)``, which cannot be
traced. This is the same maths for community-1's settings (25 ms hamming frames, 10 ms hop,
80 mel bins, preemphasis 0.97, DC removal, snip_edges, no dither), batched and traceable, and
exactly equal to the original on CPU.
"""
import math

import torch
from torchaudio.compliance.kaldi import get_mel_banks


class BatchedFbank(torch.nn.Module):
    def __init__(self, sample_rate=16000, frame_ms=25, shift_ms=10, mel_bins=80, preemph=0.97, samples=160000):
        super().__init__()
        self.win = int(sample_rate * frame_ms / 1000)
        self.hop = int(sample_rate * shift_ms / 1000)
        self.nfft = 1 << math.ceil(math.log2(self.win))
        self.preemph = preemph
        frames = (samples - self.win) // self.hop + 1
        # framing as a gather: OpenVINO's unfold conversion fails on batched input
        idx = torch.arange(frames)[:, None] * self.hop + torch.arange(self.win)[None, :]
        self.register_buffer("idx", idx.reshape(-1))
        self.register_buffer("window", torch.hamming_window(self.win, periodic=False, alpha=0.54, beta=0.46))
        mel, _ = get_mel_banks(mel_bins, self.nfft, float(sample_rate), 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel", torch.nn.functional.pad(mel, (0, 1)).T.contiguous())

    def forward(self, x):
        """x: (batch, 1, samples) in [-1, 1] -> (batch, frames, mel_bins), mean-centred over time."""
        x = x[:, 0] * 32768.0
        f = x.index_select(1, self.idx).reshape(x.shape[0], -1, self.win)
        f = f - f.mean(-1, keepdim=True)
        prev = torch.cat([f[..., :1], f[..., :-1]], -1)
        f = (f - self.preemph * prev) * self.window
        f = torch.nn.functional.pad(f, (0, self.nfft - self.win))
        spec = torch.fft.rfft(f).abs().pow(2)
        e = torch.clamp(spec @ self.mel, min=torch.finfo(torch.float32).eps).log()
        return e - e.mean(1, keepdim=True)

#!/usr/bin/env python3
"""Convert a pyannote speaker-diarization pipeline (e.g. community-1) to OpenArc's layout.

    python -m src.engine.openvino.pyannote.convert <pipeline_dir> <out_dir>

<pipeline_dir> is the downloaded HF repo (config.yaml, segmentation/, embedding/, plda/). The output
directory gets a copy of it plus openvino/{segmentation,fbank,resnet}_model.xml. The embedding model is
split in two because its fbank front end must stay in f32: audio is scaled by 32768 and squared,
which overflows f16. The ResNet behind it is f16-safe.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
import torch


class _ResNet(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, fbank, weights):
        return self.resnet(fbank, weights=weights)[1]


def convert(pipeline_dir: Path, out_dir: Path) -> None:
    import lightning.pytorch as pl
    from pyannote.audio import Pipeline

    from src.engine.openvino.pyannote.fbank import BatchedFbank

    # torch.jit.trace probes every attribute; LightningModule.trainer raises when detached
    pl.LightningModule.trainer = property(lambda self: None)

    if out_dir.resolve() != pipeline_dir.resolve():
        shutil.copytree(pipeline_dir, out_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache", "openvino"))
    pipe = Pipeline.from_pretrained(str(pipeline_dir))
    seg = pipe._segmentation.model.eval()
    emb = pipe._embedding.model_.eval()
    samples = int(pipe._segmentation.duration * 16000)
    fbank = BatchedFbank(samples=samples).eval()
    ov_dir = out_dir / "openvino"
    ov_dir.mkdir(parents=True, exist_ok=True)

    # trace at batch 2: batch 32 needs >8 GB of host memory while tracing; batch stays dynamic
    x = torch.randn(2, 1, samples) * 0.1
    with torch.no_grad():
        frames = seg(x).shape[1]
        f = fbank(x)
    w = torch.rand(2, frames)
    ov.save_model(ov.convert_model(seg, example_input=x, input=[ov.PartialShape([-1, 1, samples])]),
                  ov_dir / "segmentation_model.xml", compress_to_fp16=False)
    ov.save_model(ov.convert_model(fbank, example_input=x, input=[ov.PartialShape([-1, 1, samples])]),
                  ov_dir / "fbank_model.xml", compress_to_fp16=False)
    ov.save_model(ov.convert_model(_ResNet(emb.resnet).eval(), example_input=(f, w),
                                   input=[ov.PartialShape([-1, f.shape[1], f.shape[2]]), ov.PartialShape([-1, frames])]),
                  ov_dir / "resnet_model.xml", compress_to_fp16=False)

    # parity on CPU against the original torch models
    core = ov.Core()
    x = torch.randn(8, 1, samples) * 0.1
    w = torch.rand(8, frames)
    with torch.inference_mode():
        ref_seg = seg(x).numpy()
        ref_emb = emb(x, weights=w).numpy()
    got_seg = core.compile_model(ov_dir / "segmentation_model.xml", "CPU")([x.numpy()])[0]
    got_f = core.compile_model(ov_dir / "fbank_model.xml", "CPU")([x.numpy()])[0]
    got_emb = core.compile_model(ov_dir / "resnet_model.xml", "CPU")([got_f, w.numpy()])[0]
    print(f"segmentation max abs diff {np.abs(got_seg - ref_seg).max():.2e}")
    print(f"embedding    max abs diff {np.abs(got_emb - ref_emb).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pipeline_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    convert(a.pipeline_dir, a.out_dir)

"""Export the shipped student to ONNX for the Android app (SPEC 14.1, 14.2).

Run in the export environment (torch 2.7, onnxruntime) from ``research/``::

    python -m v1.export_onnx --ckpt ~/7oda-work/v1/runs/S1/best.pt

Steps:

1. load the student and build exactly the network that ships on iOS: every
   BatchNorm folded into the preceding conv / linear and the large weight
   tensors snapped to their 8-bit palette (``export_coreml.export_ready``).
   The ONNX file stores these weights in fp32;
2. ``torch.onnx.export`` (TorchScript exporter, opset 17), input ``x``
   float32 ``[1, 10, 256]``, outputs ``embedding`` ``[1, 128]`` and ``hand``
   ``[1, 3]``;
3. run the file with onnxruntime on the 16 inputs of ``reference_io.npz`` (one
   at a time, as on the phone) and compare with the PyTorch outputs stored
   there: max |embedding difference| < 1e-3 and the same hand argmax;
4. write ``artifacts/onnx_check.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from . import models as M
from .export_coreml import ART, export_ready

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, "android", "app", "src", "main", "assets", "hand_encoder.onnx")
EMBEDDING_TOLERANCE = 1e-3


def load_student(ckpt_path: str) -> M.Student:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student = M.Student(**ck.get("model_kw", {}))
    student.load_state_dict(ck["state"])
    return student.eval()


def export(model: torch.nn.Module, out_path: str, opset: int = 17) -> None:
    wrapper = M.ExportWrapper(model).eval()
    x0 = torch.zeros(1, 10, 256, dtype=torch.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (x0,), out_path,
            input_names=["x"], output_names=["embedding", "hand"],
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )


def check(onnx_path: str, reference_path: str) -> dict:
    import onnxruntime as ort

    ref = np.load(reference_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = [(i.name, list(i.shape), i.type) for i in sess.get_inputs()]
    outputs = [(o.name, list(o.shape), o.type) for o in sess.get_outputs()]
    emb, hand = [], []
    for k in range(ref["x"].shape[0]):
        e, h = sess.run(["embedding", "hand"], {"x": ref["x"][k:k + 1].astype(np.float32)})
        emb.append(e[0])
        hand.append(h[0])
    emb = np.stack(emb)
    hand = np.stack(hand)
    d_emb = float(np.abs(emb - ref["embedding"]).max())
    d_hand = float(np.abs(hand - ref["hand"]).max())
    agree = float((hand.argmax(1) == ref["hand"].argmax(1)).mean())
    norms = np.linalg.norm(emb, axis=1)
    return {
        "inputs": [{"name": n, "shape": s, "type": t} for n, s, t in inputs],
        "outputs": [{"name": n, "shape": s, "type": t} for n, s, t in outputs],
        "cases": int(ref["x"].shape[0]),
        "max_abs_diff_embedding": d_emb,
        "max_abs_diff_hand": d_hand,
        "hand_argmax_agreement": agree,
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "tolerance_embedding": EMBEDDING_TOLERANCE,
        "passed": bool(d_emb < EMBEDDING_TOLERANCE and agree == 1.0),
        "onnxruntime": ort.__version__,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--reference", default=os.path.join(ART, "reference_io.npz"))
    ap.add_argument("--report", default=os.path.join(ART, "onnx_check.json"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    student = load_student(args.ckpt)
    ready, names = export_ready(student)
    export(ready, args.out, args.opset)

    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "onnx": os.path.relpath(args.out, REPO),
        "size_bytes": os.path.getsize(args.out),
        "sha256": digest,
        "opset": args.opset,
        "weights": "fp32 storage of the BN-folded, 8-bit palettized student (same weights as HandEncoder.mlpackage)",
        "palettized_tensors": len(names),
        "source_checkpoint": os.path.basename(os.path.dirname(args.ckpt)),
        "reference": os.path.relpath(args.reference, REPO),
        "torch": torch.__version__,
    }
    report.update(check(args.out, args.reference))
    print(json.dumps(report, indent=1))
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    if not report["passed"]:
        raise SystemExit("ONNX outputs do not match reference_io.npz")


if __name__ == "__main__":
    main()

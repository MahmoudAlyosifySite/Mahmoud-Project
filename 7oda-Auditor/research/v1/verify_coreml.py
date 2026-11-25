#!/usr/bin/env python3
"""Check HandEncoder.mlpackage against the PyTorch reference outputs (macOS).

Needs only ``numpy`` and ``coremltools`` (Core ML predictions run on macOS)::

    python3 verify_coreml.py [--model HandEncoder.mlpackage] [--ref reference_io.npz]

Checks, for each of the reference inputs:

* output names and shapes (``embedding`` [1,128], ``hand`` [1,3]);
* max |embedding - PyTorch| < 1e-2 and the embedding norm is 1 (+-1e-2);
* identical argmax of the hand logits (inputs whose two largest reference
  logits are closer than 0.05 are not compared: fp16 may flip a near tie).

Runs on every available compute unit setting (CPU only, and all units,
which may use the GPU or the Neural Engine) and exits with status 1 on any
failure (2 when not on macOS). Paths default to the files next to this script's ``artifacts``
folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
TOL = 1e-2


def run(model_path: str, ref_path: str) -> int:
    import coremltools as ct

    ref = np.load(ref_path)
    x, e_ref, h_ref = ref["x"].astype(np.float32), ref["embedding"], ref["hand"]
    failures = []
    report = {}
    for label, unit in (("cpu_only", ct.ComputeUnit.CPU_ONLY), ("all", ct.ComputeUnit.ALL)):
        model = ct.models.MLModel(model_path, compute_units=unit)
        max_de, max_dn, mism = 0.0, 0.0, 0
        for i in range(x.shape[0]):
            out = model.predict({"x": x[i:i + 1]})
            e = np.asarray(out["embedding"], dtype=np.float32)
            h = np.asarray(out["hand"], dtype=np.float32)
            if e.shape != (1, 128) or h.shape != (1, 3):
                failures.append(f"{label}: bad output shapes {e.shape} {h.shape}")
                break
            max_de = max(max_de, float(np.abs(e[0] - e_ref[i]).max()))
            max_dn = max(max_dn, abs(float(np.linalg.norm(e[0])) - 1.0))
            top2 = np.sort(h_ref[i])[-2:]
            if top2[1] - top2[0] >= 0.05 and int(np.argmax(h[0])) != int(np.argmax(h_ref[i])):
                mism += 1
        report[label] = {"max_abs_embedding_delta": max_de, "max_norm_error": max_dn,
                         "hand_argmax_mismatches": mism, "n": int(x.shape[0])}
        if max_de >= TOL:
            failures.append(f"{label}: max |embedding delta| {max_de:.4g} >= {TOL}")
        if max_dn >= TOL:
            failures.append(f"{label}: embedding norm off by {max_dn:.4g}")
        if mism:
            failures.append(f"{label}: {mism} hand argmax mismatches")
    print(json.dumps(report, indent=1))
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("OK: Core ML outputs match the PyTorch reference")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(ART, "HandEncoder.mlpackage"))
    ap.add_argument("--ref", default=os.path.join(ART, "reference_io.npz"))
    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        print("SKIPPED: Core ML predictions need macOS.")
        return 2
    return run(args.model, args.ref)


if __name__ == "__main__":
    sys.exit(main())

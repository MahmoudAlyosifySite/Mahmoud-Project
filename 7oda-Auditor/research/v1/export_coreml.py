"""Export the student to Core ML (SPEC 14.4).

Run in the export environment (torch 2.7 + coremltools 9.0, see
``requirements-v1.txt``)::

    python -m v1.export_coreml --ckpt /tmp/.../S1/best.pt --val '/tmp/.../val/*.npz'

Steps:

1. load the student, fold every BatchNorm into the preceding conv / linear;
2. 8-bit palettization: each weight tensor with more than 2048 elements is snapped
   to 256 values by 1-D k-means in PyTorch, then the converted model is
   palettized with ``coremltools.optimize.coreml.palettize_weights`` in
   ``unique`` mode, which stores exactly those values as an 8-bit lookup
   table. PyTorch and Core ML therefore hold the same weights, and the
   reference outputs measure only the conversion (fp16 compute);
3. trace and convert to an ML Program (iOS 17), input ``x`` [1, 10, 256]
   float32, outputs ``embedding`` [1, 128] and ``hand`` [1, 3] float32;
4. write ``HandEncoder.mlpackage``, ``reference_io.npz`` (16 inputs and the
   PyTorch fp32 outputs of the palettized model) and the model card
   ``README.md`` to ``research/v1/artifacts``; also a PyTorch checkpoint of
   the palettized student (outside the repository) for evaluation.

The PyTorch-only parts (``fold_batchnorm``, ``palettize_tensor``,
``palettize_model``) do not need coremltools and are unit-tested.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn

from . import models as M

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
WEIGHT_THRESHOLD = 2048
NBITS = 8


# ---------------------------------------------------------------------------
# PyTorch side (no coremltools needed)
# ---------------------------------------------------------------------------

def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    std = torch.sqrt(bn.running_var + bn.eps)
    g = bn.weight.detach() if bn.affine else torch.ones_like(std)
    beta = bn.bias.detach() if bn.affine else torch.zeros_like(std)
    fused = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True)
    fused.weight.data = w * (g / std)[:, None, None]
    fused.bias.data = (b - bn.running_mean) * g / std + beta
    return fused


def _fuse_linear_bn(lin: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    w = lin.weight.detach()
    b = lin.bias.detach() if lin.bias is not None else torch.zeros(w.shape[0])
    std = torch.sqrt(bn.running_var + bn.eps)
    g = bn.weight.detach() if bn.affine else torch.ones_like(std)
    beta = bn.bias.detach() if bn.affine else torch.zeros_like(std)
    fused = nn.Linear(lin.in_features, lin.out_features, bias=True)
    fused.weight.data = w * (g / std)[:, None]
    fused.bias.data = (b - bn.running_mean) * g / std + beta
    return fused


def fold_batchnorm(student: M.Student) -> M.Student:
    """Copy of the student (eval mode) with every BatchNorm folded away."""
    m = copy.deepcopy(student).eval()
    m.stem[0] = _fuse_conv_bn(m.stem[0], m.stem[1])
    m.stem[1] = nn.Identity()
    for blk in m.blocks:
        blk.pw1 = _fuse_conv_bn(blk.pw1, blk.bn1)
        blk.bn1 = nn.Identity()
        blk.dw = _fuse_conv_bn(blk.dw, blk.bn2)
        blk.bn2 = nn.Identity()
        blk.pw2 = _fuse_conv_bn(blk.pw2, blk.bn3)
        blk.bn3 = nn.Identity()
    m.emb[0] = _fuse_linear_bn(m.emb[0], m.emb[1])
    m.emb[1] = nn.Identity()
    return m.eval()


def palettize_tensor(w: np.ndarray, nbits: int = NBITS, iters: int = 30) -> np.ndarray:
    """1-D k-means with 2**nbits centroids (quantile init, Lloyd iterations).

    Values are first rounded to fp16, as the converted model stores them.
    """
    flat = w.astype(np.float16).astype(np.float64).ravel()
    k = 2 ** nbits
    uniq = np.unique(flat)
    if uniq.size <= k:
        return flat.reshape(w.shape).astype(np.float32)
    cent = np.quantile(flat, (np.arange(k) + 0.5) / k)
    cent = np.unique(cent)
    for _ in range(iters):
        mids = (cent[1:] + cent[:-1]) / 2
        lab = np.searchsorted(mids, flat)
        sums = np.bincount(lab, weights=flat, minlength=cent.size)
        cnt = np.bincount(lab, minlength=cent.size)
        new = np.where(cnt > 0, sums / np.maximum(cnt, 1), cent)
        if np.allclose(new, cent):
            cent = new
            break
        cent = np.unique(new)
    cent = np.unique(cent.astype(np.float16).astype(np.float64))
    mids = (cent[1:] + cent[:-1]) / 2
    out = cent[np.searchsorted(mids, flat)]
    return out.reshape(w.shape).astype(np.float32)


def palettize_model(model: nn.Module, nbits: int = NBITS, threshold: int = WEIGHT_THRESHOLD):
    """Snap every weight with more than ``threshold`` elements (the coremltools
    ``weight_threshold`` rule) to 2**nbits values, in place.
    Returns the names of the palettized tensors."""
    names = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2 and p.numel() > threshold:
                p.copy_(torch.from_numpy(palettize_tensor(p.detach().cpu().numpy(), nbits)))
                names.append(name)
    return names


def export_ready(student: M.Student):
    """BN-folded, palettized copy (fp32 PyTorch) and the palettized tensor names."""
    folded = fold_batchnorm(student)
    names = palettize_model(folded)
    return folded.eval(), names


# ---------------------------------------------------------------------------
# Core ML side
# ---------------------------------------------------------------------------

def convert(model: nn.Module, out_path: str, meta: dict):
    import coremltools as ct
    import coremltools.optimize as cto

    wrapper = M.ExportWrapper(model).eval()
    example = torch.zeros(1, 10, 256)
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 10, 256), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32), ct.TensorType(name="hand", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    cfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpPalettizerConfig(
        mode="unique", weight_threshold=WEIGHT_THRESHOLD))
    mlmodel = cto.coreml.palettize_weights(mlmodel, cfg)
    mlmodel.short_description = ("7oda Auditor v1 hand encoder: 128-d L2-normalized embedding and "
                                 "right/left/twoHanded logits from the SPEC 14.1 tensor.")
    mlmodel.author = "7oda Auditor research"
    mlmodel.version = meta.get("version", "1.0")
    mlmodel.input_description["x"] = ("float32 [1,10,256]: accelHP/0.02 (0-2), gyroHP/0.05 (3-5), "
                                      "gravity (6-8), template (9); aligned so the pattern starts at u=25")
    mlmodel.output_description["embedding"] = "float32 [1,128], L2-normalized"
    mlmodel.output_description["hand"] = "float32 [1,3] logits: right, left, twoHanded"
    for k, v in meta.items():
        mlmodel.user_defined_metadata[k] = str(v)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    return mlmodel


def lut_ops(mlmodel) -> dict:
    """Count palettized (lookup-table) constants in the ML Program."""
    prog = mlmodel._mil_program
    counts = {}
    if prog is None:
        return counts
    for fn in prog.functions.values():
        for op in fn.operations:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return {k: v for k, v in counts.items() if "lut" in k or "palett" in k}


def dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def reference_inputs(val_glob: str, n: int = 16, seed: int = 0) -> np.ndarray:
    """Real tensors from validation shards (falls back to random ones)."""
    import glob

    files = sorted(glob.glob(val_glob)) if val_glob else []
    if files:
        z = np.load(files[0], allow_pickle=True)
        x = z["x"].astype(np.float32)
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
        return x[np.sort(idx)]
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 10, 256)).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", default="", help="glob of validation shards for the reference inputs")
    ap.add_argument("--out-dir", default=ART)
    ap.add_argument("--work-ckpt", default="", help="where to save the palettized PyTorch student")
    ap.add_argument("--card", default="", help="JSON file with evaluation numbers for the model card")
    ap.add_argument("--version", default="1.0")
    args = ap.parse_args(argv)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    student = M.Student(**ck.get("model_kw", {}))
    student.load_state_dict(ck["state"])
    student.eval()
    ready, names = export_ready(student)
    x = reference_inputs(args.val)
    with torch.no_grad():
        e_ref, h_ref = M.ExportWrapper(ready)(torch.from_numpy(x))
        e_orig, h_orig = M.ExportWrapper(student)(torch.from_numpy(x))
    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "reference_io.npz")
    np.savez(npz, x=x, embedding=e_ref.numpy().astype(np.float32), hand=h_ref.numpy().astype(np.float32),
             embedding_unpalettized=e_orig.numpy().astype(np.float32),
             hand_unpalettized=h_orig.numpy().astype(np.float32))
    pal_delta = float(np.abs(e_ref.numpy() - e_orig.numpy()).max())
    print(f"palettized tensors: {len(names)}; max |embedding change| from palettization {pal_delta:.4f}; "
          f"hand argmax agreement {float((h_ref.argmax(1) == h_orig.argmax(1)).float().mean()):.3f}")
    if args.work_ckpt:
        torch.save({"kind": "student_folded", "model_kw": ck.get("model_kw", {}), "state": ready.state_dict(),
                    "source": args.ckpt}, args.work_ckpt)
    meta = {"source_checkpoint": os.path.basename(os.path.dirname(args.ckpt)), "params": M.count_params(student),
            "palettization": "8-bit lookup table (k-means), weights with more than 2048 elements", "version": args.version,
            "spec": "SPEC 14.1-14.4"}
    out = os.path.join(args.out_dir, "HandEncoder.mlpackage")
    mlmodel = convert(ready, out, meta)
    size = dir_size(out)
    luts = lut_ops(mlmodel)
    spec = mlmodel.get_spec()
    summary = {"path": out, "size_bytes": size, "lut_ops": luts, "palettized_tensors": len(names),
               "params": M.count_params(student), "palettization_max_delta": pal_delta,
               "inputs": [i.name for i in spec.description.input],
               "outputs": [o.name for o in spec.description.output],
               "spec_version": spec.specificationVersion}
    print(json.dumps(summary, indent=1))
    with open(os.path.join(args.out_dir, "export_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()

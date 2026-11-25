"""Self-supervised pretraining of the teacher (proposal step 2).

Two objectives on unlabelled recordings:

* masked reconstruction (as in MOMENT): ~30% of the 8-sample patches of the
  signal channels are zeroed (the challenge channel stays visible) and a light
  head reconstructs them from the last feature map (32 steps = one per patch);
  MSE on the masked patches only;
* contrastive views of the same recording (as in Mantis / SimCLR): two
  independently augmented views, NT-Xent on a projection of the pooled
  features.

Input: TensorSet shards (labels ignored) and/or app exports (JSON Lines,
any purpose, labelled or not). Output: a teacher checkpoint usable as
``--init`` for ``v1.train_teacher`` or ``v1.finetune``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import data as D
from . import losses as L
from . import models as M
from . import trainlib as TL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", default="", help="glob(s) of TensorSet shards")
    ap.add_argument("--jsonl", default="", help="comma-separated app exports (unlabelled is fine)")
    ap.add_argument("--val", default="", help="labelled shards for the cross-session probe")
    ap.add_argument("--arch", default="teacher", choices=["teacher", "teacher_small"])
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--contrast-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out_dir = TL.run_dir(args.run)
    logger = TL.Logger(out_dir)
    log = logger.log
    log(f"args {json.dumps(vars(args))}; gpu before: {TL.gpu_note()}")

    sets = []
    if args.train:
        sets.append(TL.load_sets(args.train, args.max_train or None))
    if args.jsonl:
        recs = [r for p in args.jsonl.split(",") for r in D.load_jsonl(p)]
        for r in recs:
            r.setdefault("label", {"personID": "unlabelled", "hand": "right"})
        sets.append(D.tensorset_from_recordings(recs, min_ncc=0.30))
    ts = D.TensorSet.concat(sets)
    log(f"samples {len(ts)}")
    dev = torch.device("cuda")
    dd = D.DeviceData(ts, dev)
    del ts
    vs = TL.load_sets(args.val) if args.val else None

    model = M.build_model(args.arch).to(dev)
    if args.init:
        from .train_teacher import load_init
        load_init(model, args.init, log)
    steps_feat = 256 // (2 ** (len(model.widths) - 1))
    if 256 // args.patch != steps_feat:
        raise SystemExit(f"patch {args.patch} must give {steps_feat} patches")
    recon = M.ReconHead(model.widths[-1], args.patch).to(dev)
    proj = M.ProjectionHead(model.feat_dim).to(dev)
    opt = torch.optim.AdamW(TL.make_param_groups(model, args.wd, [recon, proj]), lr=args.lr)
    aug = D.Augmenter(D.AugmentConfig(), seed=args.seed + 1, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 2)
    steps_per_epoch = dd.n // args.batch
    total = int(args.epochs * steps_per_epoch)
    log(f"params {M.count_params(model)}, steps {total}")
    step = 0
    acc = {"recon": 0.0, "contrast": 0.0, "n": 0}
    while step < total:
        for idx in dd.epoch_batches(args.batch, generator=gen):
            if step >= total:
                break
            TL.set_lr(opt, TL.cosine_lr(step, total, args.warmup, args.lr))
            model.train()
            x = dd.batch(idx)[0]
            v1 = aug(x)
            v2 = aug(x)
            mask = L.patch_mask(x.shape[0], args.patch, args.mask_ratio, device=dev, generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o1 = model(L.apply_mask(v1, mask), return_all=True)
                o2 = model(v2, return_all=True)
                pred = recon(o1["feature_map"])
                z1 = proj(o1["features"])
                z2 = proj(o2["features"])
            l_rec = L.masked_recon_loss(pred, v1, mask)
            l_con = L.nt_xent(z1, z2, args.temperature)
            loss = args.recon_weight * l_rec + args.contrast_weight * l_con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            acc["recon"] += float(l_rec)
            acc["contrast"] += float(l_con)
            acc["n"] += 1
            step += 1
            if step % 200 == 0:
                n = acc["n"]
                log(f"step {step}/{total} recon {acc['recon'] / n:.4f} contrast {acc['contrast'] / n:.4f}")
                logger.metrics(step=step, recon=acc["recon"] / n, contrast=acc["contrast"] / n)
                acc = {"recon": 0.0, "contrast": 0.0, "n": 0}
            if vs is not None and (step % args.eval_every == 0 or step == total):
                met = TL.quick_validate(model, vs, dev)
                log(f"VAL step {step}: {json.dumps(met)}")
                logger.metrics(step=step, val=met)
    TL.save_ckpt(f"{out_dir}/last.pt", model, args.arch, {}, {"step": step, "ssl": True})
    torch.save({"recon": recon.state_dict(), "proj": proj.state_dict()}, f"{out_dir}/ssl_heads.pt")
    log(f"done; gpu after: {TL.gpu_note()}")


if __name__ == "__main__":
    main()

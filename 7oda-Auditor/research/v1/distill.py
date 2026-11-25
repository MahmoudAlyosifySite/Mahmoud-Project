"""Distill one or more teachers into the student (proposal step 5).

Loss = w_cos * (1 - cos(student, teacher)) + w_rel * relational distances
     + w_arc * ArcFace(labels) + w_hand * CE(hand) + w_kd * KL(teacher hand).

The teacher target is the renormalized mean of the teachers' embeddings
(an ensemble when several checkpoints are given). The student's ArcFace head
starts from the first teacher's class centres, so the student is pulled into
the teacher's identity space. Mirrored copies (hand head only) take no part
in the embedding losses.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import losses as L
from . import models as M
from . import trainlib as TL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--teachers", required=True, help="comma-separated teacher checkpoints")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--w-rel", type=float, default=1.0)
    ap.add_argument("--w-arc", type=float, default=0.5)
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--w-kd", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--mirror-fraction", type=float, default=0.25)
    ap.add_argument("--mirror-mode", default="actuator", choices=["reflect", "actuator"],
                    help="actuator (default): physically consistent for a phone-fixed actuator; see docs/V1_MODEL.md")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", default="full", choices=["full", "light", "none"])
    args = ap.parse_args(argv)
    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out_dir = TL.run_dir(args.run)
    logger = TL.Logger(out_dir)
    log = logger.log
    log(f"args {json.dumps(vars(args))}; gpu before: {TL.gpu_note()}")
    dev = torch.device("cuda")

    teachers = []
    paths = [p for p in args.teachers.split(",") if p]
    for p in paths:
        t, _ = TL.load_model(p, dev)
        t.eval()
        for q in t.parameters():
            q.requires_grad_(False)
        teachers.append(t)
    ts = TL.load_sets(args.train, args.max_train or None, device=dev)
    ident = ts.identity()
    n_classes = int(ident.max()) + 1
    dd = D.DeviceData(ts, dev, identity=ident)
    log(f"teachers {len(teachers)}, train {len(ts)}, classes {n_classes}")
    del ts
    vs = TL.load_sets(args.val) if args.val else None

    student = M.Student().to(dev)
    arc = M.ArcFace(M.EMB_DIM, n_classes, margin=args.margin, scale=args.scale).to(dev)
    arc_path = os.path.join(os.path.dirname(paths[0]), "arcface.pt")
    if os.path.exists(arc_path):
        a = torch.load(arc_path, map_location="cpu", weights_only=False)
        if a["arcface"]["weight"].shape == arc.weight.shape:
            arc.load_state_dict(a["arcface"])
            log(f"ArcFace head initialized from {arc_path}")
    opt = torch.optim.AdamW(TL.make_param_groups(student, args.wd, [arc]), lr=args.lr)
    aug = D.Augmenter(D.AugmentConfig.named(args.aug), seed=args.seed + 1, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 2)
    steps_per_epoch = dd.n // args.batch
    total = int(args.epochs * steps_per_epoch)
    log(f"student params {M.count_params(student)}, steps {total}")
    best = -1.0
    step = 0
    state_path = f"{out_dir}/state.pt"
    if os.path.exists(state_path):
        st = TL.load_state(state_path, student=student, arc=arc, opt=opt)
        step, best = int(st["step"]), float(st["best"])
        log(f"resumed from {state_path} at step {step}")
    last_val = {}
    agg = {k: 0.0 for k in ("cos", "rel", "arc", "hand", "kd")}
    agg_n = 0
    while step < total:
        for idx in dd.epoch_batches(args.batch, generator=gen):
            if step >= total:
                break
            TL.set_lr(opt, TL.cosine_lr(step, total, args.warmup, args.lr))
            student.train()
            x, idt, hand, _ = dd.batch(idx)
            x = aug(x)
            x, idt, hand, is_m = D.add_mirrored(x, idt, hand, args.mirror_fraction, generator=gen,
                                               mode=args.mirror_mode)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs = [t(x) for t in teachers]
                t_emb = F.normalize(torch.stack([o["embedding"].float() for o in outs]).mean(0), dim=1)
                t_hand = torch.stack([o["hand"].float() for o in outs]).mean(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student(x)
            real = ~is_m
            l_cos = L.cosine_distill(out["embedding"][real], t_emb[real])
            l_rel = L.relational_distill(out["embedding"][real], t_emb[real])
            l_arc, _ = arc(out["embedding"], idt)
            l_hand = F.cross_entropy(out["hand"].float(), hand)
            l_kd = L.logit_distill(out["hand"], t_hand)
            loss = (args.w_cos * l_cos + args.w_rel * l_rel + args.w_arc * l_arc + args.w_hand * l_hand
                    + args.w_kd * l_kd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            for k, v in (("cos", l_cos), ("rel", l_rel), ("arc", l_arc), ("hand", l_hand), ("kd", l_kd)):
                agg[k] += float(v)
            agg_n += 1
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{total} " + " ".join(f"{k} {v / agg_n:.4f}" for k, v in agg.items()))
                logger.metrics(step=step, **{k: v / agg_n for k, v in agg.items()})
                agg = {k: 0.0 for k in agg}
                agg_n = 0
            if vs is not None and (step % args.eval_every == 0 or step == total):
                met = TL.quick_validate(student, vs, dev)
                last_val = met
                log(f"VAL step {step}: {json.dumps(met)}")
                logger.metrics(step=step, val=met)
                TL.write_results(args.run, out_dir)
                if met.get("top1_id", 0) > best:
                    best = met["top1_id"]
                    TL.save_ckpt(f"{out_dir}/best.pt", student, "student", {}, {"step": step, "val": met})
            if step % args.eval_every == 0 or step == total:
                TL.save_state(state_path, student=student, arc=arc, opt=opt, step=step, best=best)
    TL.save_ckpt(f"{out_dir}/last.pt", student, "student", {}, {"step": step, "teachers": paths})
    if vs is None:
        TL.save_ckpt(f"{out_dir}/best.pt", student, "student", {}, {"step": step})
    log(f"done, best val top1_id {best:.4f}; gpu after: {TL.gpu_note()}")
    TL.write_results(args.run, out_dir, {"run": args.run, "kind": "distill", "teachers": paths,
                                         "params": M.count_params(student), "steps": step,
                                         "best_val_top1_id": best, "final_val": last_val, "args": vars(args),
                                         "minutes": round((time.time() - t_start) / 60, 1)})
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()

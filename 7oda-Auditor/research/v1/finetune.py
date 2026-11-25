"""Fine-tune a teacher on study data (proposal step 4).

Data: app exports (JSON Lines, ``--jsonl``) or TensorSet shards
(``--train``), labelled with person, hand and session. The teacher keeps its
encoder (lower learning rate) and gets a fresh ArcFace head for the study
identities (person, hand); the hand head keeps training with mirrored
copies. Heavy augmentation (time jitter, gain, noise, case damping, sample
dropout).

People are split by ``--holdout-people`` (comma-separated ids, or a number
``k`` for the last k people in sorted order): fine-tuning never sees them,
so they play the role of people who enroll later, and the cross-session
probe runs on them. With 20 participants, run it as k-fold over people
(``--fold i --folds k``) and report the mean.

Example (real study)::

    python -m v1.finetune --run FT1 --init /tmp/.../T1/best.pt \
        --jsonl exports/p01.jsonl,exports/p02.jsonl,... --folds 4 --fold 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M
from . import trainlib as TL


def split_people(ts: D.TensorSet, holdout: str, fold: int, folds: int, seed: int = 0):
    ids = sorted(ts.person_ids)
    if folds > 1:
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(ids))
        hold = set(order[fold::folds])
    elif holdout and holdout.isdigit():
        hold = set(ids[-int(holdout):])
    else:
        hold = set(h for h in holdout.split(",") if h)
    is_hold = np.array([ts.person_ids[p] in hold for p in ts.person])
    return np.where(~is_hold)[0], np.where(is_hold)[0], sorted(hold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", required=True, help="teacher checkpoint")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--train", default="")
    ap.add_argument("--holdout-people", default="5")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--hand-weight", type=float, default=0.5)
    ap.add_argument("--mirror-fraction", type=float, default=0.25)
    ap.add_argument("--mirror-mode", default="actuator", choices=["reflect", "actuator"],
                    help="actuator (default): physically consistent for a phone-fixed actuator; see docs/V1_MODEL.md")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    out_dir = TL.run_dir(args.run)
    logger = TL.Logger(out_dir)
    log = logger.log
    log(f"args {json.dumps(vars(args))}; gpu before: {TL.gpu_note()}")
    dev = torch.device("cuda")

    sets = []
    if args.jsonl:
        sets.append(D.tensorset_from_jsonl(args.jsonl.split(",")))
    if args.train:
        sets.append(TL.load_sets(args.train))
    ts = D.TensorSet.concat(sets)
    tr_idx, ho_idx, hold = split_people(ts, args.holdout_people, args.fold, args.folds, args.seed)
    train = ts.subset(tr_idx)
    held = ts.subset(ho_idx)
    log(f"study samples {len(ts)}: fine-tune {len(train)} ({len(set(train.person.tolist()))} people), "
        f"held out {len(held)} ({hold})")
    # re-index identities of the fine-tuning people
    uniq = np.unique(train.person)
    remap = {p: i for i, p in enumerate(uniq)}
    ident = np.array([remap[p] * 2 + h if h < 2 else D.IGNORE for p, h in zip(train.person, train.hand)])
    dd = D.DeviceData(train, dev, identity=ident)

    model, ck = TL.load_model(args.init, dev)
    arc = M.ArcFace(M.EMB_DIM, 2 * len(uniq), margin=args.margin, scale=args.scale).to(dev)
    with torch.no_grad():   # start the class centres at the teacher's mean embeddings
        emb, _ = TL.embed(model, train.x, dev)
        for c in range(2 * len(uniq)):
            m = ident == c
            if m.any():
                arc.weight[c] = torch.from_numpy(emb[m].mean(0)).to(dev)
    groups = TL.make_param_groups(model, args.wd)
    groups.append({"params": list(arc.parameters()), "weight_decay": 0.0, "lr_mult": args.head_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr)
    aug = D.Augmenter(D.AugmentConfig(case_prob=0.5, dropout_prob=0.5, gain_sigma=0.2), seed=args.seed + 1,
                      device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 2)
    bs = min(args.batch, max(16, dd.n // 4))
    steps_per_epoch = max(1, dd.n // bs)
    total = int(args.epochs * steps_per_epoch)
    probe0 = TL.quick_validate(model, held, dev) if len(held) else {}
    log(f"held-out probe before fine-tuning: {json.dumps(probe0)}")
    logger.metrics(step=0, held=probe0)
    best, step = -1.0, 0
    while step < total:
        for idx in dd.epoch_batches(bs, generator=gen):
            if step >= total:
                break
            TL.set_lr(opt, TL.cosine_lr(step, total, max(1, total // 20), args.lr))
            model.train()
            x, idt, hand, _ = dd.batch(idx)
            x = aug(x)
            x, idt, hand, _ = D.add_mirrored(x, idt, hand, args.mirror_fraction, generator=gen,
                                               mode=args.mirror_mode)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            l_id, _ = arc(out["embedding"], idt)
            loss = l_id + args.hand_weight * F.cross_entropy(out["hand"].float(), hand)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            if (step % args.eval_every == 0 or step == total) and len(held):
                met = TL.quick_validate(model, held, dev)
                log(f"step {step}/{total} loss {float(loss):.4f} held-out probe {json.dumps(met)}")
                logger.metrics(step=step, held=met)
                if met.get("top1_id", 0) > best:
                    best = met["top1_id"]
                    TL.save_ckpt(f"{out_dir}/best.pt", model, ck["kind"], ck.get("model_kw", {}),
                                 {"step": step, "held": met, "holdout": hold})
    TL.save_ckpt(f"{out_dir}/last.pt", model, ck["kind"], ck.get("model_kw", {}), {"step": step, "holdout": hold})
    log(f"done: held-out top1_id before {probe0.get('top1_id')}, best after {best:.4f}")


if __name__ == "__main__":
    main()

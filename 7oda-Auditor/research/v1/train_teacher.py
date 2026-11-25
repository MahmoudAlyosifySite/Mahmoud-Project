"""Supervised training on labelled recordings (proposal step 1, and the
from-scratch student baseline).

Identity = (person, hand) with an ArcFace head; the hand head is trained
with cross entropy on right / left / twoHanded, plus mirrored copies of part
of each batch (hand labels swapped, identity ignored). Augmentations run on
the GPU. Data: TensorSet shards (``v1.simdata``) loaded into GPU memory.

Example::

    python -m v1.train_teacher --run T1 --train '/tmp/.../train/*.npz' \
        --val '/tmp/.../val/*.npz' --epochs 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import losses as L
from . import models as M
from . import trainlib as TL


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True, help="glob(s) of TensorSet shards, comma-separated")
    ap.add_argument("--val", default="", help="glob(s) of validation shards (held-out people)")
    ap.add_argument("--arch", default="teacher",
                    choices=["teacher", "teacher_small", "student", "mantis_pretrained", "mantis_scratch"])
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--margin-warmup-epochs", type=float, default=1.0)
    ap.add_argument("--hand-weight", type=float, default=0.5)
    ap.add_argument("--mirror-fraction", type=float, default=0.25)
    ap.add_argument("--mirror-mode", default="actuator", choices=["reflect", "actuator"],
                    help="actuator (default): physically consistent for a phone-fixed actuator; see docs/V1_MODEL.md")
    ap.add_argument("--ssl-weight", type=float, default=0.0, help="auxiliary masked reconstruction")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--aug", default="full", choices=["full", "light", "none"])
    ap.add_argument("--init", default="", help="checkpoint to initialize the encoder from")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap


def load_init(model, path: str, log):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("state", ck)
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in keep]
    model.load_state_dict(keep, strict=False)
    log(f"init from {path}: loaded {len(keep)} tensors, {len(missing)} left at random init")


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out_dir = TL.run_dir(args.run)
    logger = TL.Logger(out_dir)
    log = logger.log
    log(f"args {json.dumps(vars(args))}")
    log(f"gpu before: {TL.gpu_note()}")

    ts = TL.load_sets(args.train, args.max_train or None, device=args.device)
    ident = ts.identity()
    n_classes = int(ident.max()) + 1
    log(f"train samples {len(ts)}, people {len(ts.person_ids)}, classes {n_classes}, "
        f"hands {np.bincount(ts.hand, minlength=3).tolist()}")
    vs = TL.load_sets(args.val) if args.val else None
    if vs is not None:
        log(f"val samples {len(vs)}, people {len(vs.person_ids)}")
    dev = torch.device(args.device)
    dd = D.DeviceData(ts, dev, identity=ident)
    del ts

    model_kw = {}
    model = M.build_model(args.arch, **model_kw).to(dev)
    if args.init:
        load_init(model, args.init, log)
    arc = M.ArcFace(M.EMB_DIM, n_classes, margin=0.0, scale=args.scale).to(dev)
    recon = None
    if args.ssl_weight > 0:
        if args.arch not in ("teacher", "teacher_small"):
            raise SystemExit("--ssl-weight needs a ResNet teacher architecture")
        recon = M.ReconHead(model.widths[-1], patch=256 // 32).to(dev)
    extra = [arc] + ([recon] if recon is not None else [])
    opt = torch.optim.AdamW(TL.make_param_groups(model, args.wd, extra), lr=args.lr, betas=(0.9, 0.99))
    aug = D.Augmenter(D.AugmentConfig.named("none" if args.no_aug else args.aug), seed=args.seed + 1,
                      device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 2)

    steps_per_epoch = dd.n // args.batch
    total = int(args.epochs * steps_per_epoch)
    margin_steps = int(args.margin_warmup_epochs * steps_per_epoch)
    log(f"params {M.count_params(model)}, steps {total} ({steps_per_epoch}/epoch)")

    best = -1.0
    step = 0
    state_path = f"{out_dir}/state.pt"
    if os.path.exists(state_path):
        st = TL.load_state(state_path, model=model, arc=arc, opt=opt, **({"recon": recon} if recon else {}))
        step, best = int(st["step"]), float(st["best"])
        log(f"resumed from {state_path} at step {step} (best {best:.4f})")
    last_val = {}
    run_loss = run_id_acc = run_hand_acc = 0.0
    run_n = 0
    done = False
    while not done:
        for idx in dd.epoch_batches(args.batch, generator=gen):
            if step >= total:
                done = True
                break
            lr = TL.cosine_lr(step, total, args.warmup, args.lr)
            TL.set_lr(opt, lr)
            arc.set_margin(args.margin * min(1.0, step / max(1, margin_steps)))
            model.train()
            x, idt, hand, _ = dd.batch(idx)
            x = aug(x)
            x, idt, hand, _ = D.add_mirrored(x, idt, hand, args.mirror_fraction, generator=gen,
                                               mode=args.mirror_mode)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if recon is not None:
                    mask = L.patch_mask(x.shape[0], 8, 0.3, device=dev, generator=gen)
                    xin = L.apply_mask(x, mask)
                    out = model(xin, return_all=True)
                else:
                    out = model(x)
            loss_id, logits = arc(out["embedding"], idt)
            loss_hand = F.cross_entropy(out["hand"].float(), hand)
            loss = loss_id + args.hand_weight * loss_hand
            if recon is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = recon(out["feature_map"])
                loss = loss + args.ssl_weight * L.masked_recon_loss(pred, x, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                v = idt >= 0
                run_id_acc += float((logits[v].argmax(1) == idt[v]).float().mean()) if v.any() else 0.0
                run_hand_acc += float((out["hand"].argmax(1) == hand).float().mean())
                run_loss += float(loss)
                run_n += 1
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{total} lr {lr:.2e} m {arc.margin:.2f} loss {run_loss / run_n:.4f} "
                    f"id_acc {run_id_acc / run_n:.3f} hand_acc {run_hand_acc / run_n:.3f}")
                logger.metrics(step=step, loss=run_loss / run_n, id_acc=run_id_acc / run_n,
                               hand_acc=run_hand_acc / run_n, lr=lr)
                run_loss = run_id_acc = run_hand_acc = 0.0
                run_n = 0
            if vs is not None and (step % args.eval_every == 0 or step == total):
                met = TL.quick_validate(model, vs, dev)
                last_val = met
                log(f"VAL step {step}: {json.dumps(met)}")
                logger.metrics(step=step, val=met)
                TL.write_results(args.run, out_dir)
                score = met.get("top1_id", 0.0)
                if score > best:
                    best = score
                    TL.save_ckpt(f"{out_dir}/best.pt", model, args.arch, model_kw,
                                 {"step": step, "val": met, "n_classes": n_classes})
                TL.save_ckpt(f"{out_dir}/last.pt", model, args.arch, model_kw, {"step": step, "val": met})
            if step % args.eval_every == 0 or step == total:
                TL.save_state(state_path, model=model, arc=arc, opt=opt, step=step, best=best,
                              **({"recon": recon} if recon else {}))
    TL.save_ckpt(f"{out_dir}/last.pt", model, args.arch, model_kw, {"step": step})
    torch.save({"arcface": arc.state_dict(), "n_classes": n_classes}, f"{out_dir}/arcface.pt")
    if vs is None:
        TL.save_ckpt(f"{out_dir}/best.pt", model, args.arch, model_kw, {"step": step})
    log(f"done, best val top1_id {best:.4f}; gpu after: {TL.gpu_note()}")
    TL.write_results(args.run, out_dir, {"run": args.run, "kind": "supervised", "arch": args.arch,
                                         "params": M.count_params(model), "steps": step, "best_val_top1_id": best,
                                         "final_val": last_val, "args": vars(args),
                                         "train_samples": int(dd.n), "classes": n_classes,
                                         "minutes": round((time.time() - t_start) / 60, 1)})
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()

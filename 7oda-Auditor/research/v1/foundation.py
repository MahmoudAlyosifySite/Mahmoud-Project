"""Foundation-model teacher initialization (proposal step 3): Mantis-8M.

Mantis-8M (``paris-noah/Mantis-8M`` on Hugging Face, MIT licence, public
weights) is a contrastively pretrained transformer for univariate series of
512 samples. The teacher built on it:

* resizes each of the 10 channels of the SPEC 14.1 tensor from 256 to 512
  samples (linear interpolation) and encodes every channel independently
  (the template channel is one of them, so the challenge conditions the
  embedding by concatenation);
* concatenates the 10 x 256 channel embeddings and maps them to the 128-d
  L2-normalized embedding and the 3 hand logits, like ``models.Teacher``.

``kind = mantis_pretrained`` loads the public weights; ``mantis_scratch`` is
the same network from random initialization (the control that isolates the
value of pretraining). Needs the ``mantis-tsfm`` package (installed
``--no-deps`` in a separate virtualenv, see docs/V1_MODEL.md); downloads
need no token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import EMB_DIM, IN_CH, N_HAND, l2_normalize

HF_REPO = "paris-noah/Mantis-8M"


class MantisTeacher(nn.Module):
    def __init__(self, pretrained: bool = True, emb_dim: int = EMB_DIM, dropout: float = 0.1):
        super().__init__()
        from mantis.architecture import Mantis8M

        net = Mantis8M(device="cpu")
        if pretrained:
            net = Mantis8M.from_pretrained(HF_REPO, device="cpu")
        self.net = net
        self.widths = (256,)
        feat = IN_CH * 256
        self.feat_dim = feat
        self.emb = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, 512), nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True), nn.Linear(512, emb_dim), nn.BatchNorm1d(emb_dim))
        self.hand_head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Linear(256, N_HAND))

    def forward(self, x, return_all: bool = False):
        b, c, _ = x.shape
        z = F.interpolate(x, size=512, mode="linear", align_corners=False)
        f = self.net(z.reshape(b * c, 1, 512)).reshape(b, c * 256)
        emb_raw = self.emb(f)
        out = {"embedding": l2_normalize(emb_raw.float()), "hand": self.hand_head(f)}
        if return_all:
            out.update(features=f, embedding_raw=emb_raw)
        return out

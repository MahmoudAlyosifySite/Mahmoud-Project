"""Encoders and heads.

The encoders are torchvision ResNets with the CIFAR stem used throughout this
project (3x3 stride-1 first convolution, no max-pooling; Chen et al. 2020,
App. B.9).
"""

import torch
import torch.nn as nn
import torchvision


def _cifar_resnet(name):
    if name == "resnet18":
        net = torchvision.models.resnet18(weights=None)
        dim = 512
    elif name == "resnet50":
        net = torchvision.models.resnet50(weights=None)
        dim = 2048
    else:
        raise ValueError(name)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net, dim


class MLPHead(nn.Module):
    """Linear - BN - ReLU - Linear, optionally with BN on the output."""

    def __init__(self, in_dim, hidden_dim, out_dim, last_bn=False, layers=2):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        if last_bn:
            mods += [nn.Linear(d, out_dim, bias=False), nn.BatchNorm1d(out_dim)]
        else:
            mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """Encoder f and projection head g (hidden layer with BN, no BN on the output)."""

    def __init__(self, backbone="resnet18", proj_dim=128):
        super().__init__()
        self.encoder, self.feat_dim = _cifar_resnet(backbone)
        self.head = MLPHead(self.feat_dim, self.feat_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        return h, self.head(h)


class SimSiamModel(nn.Module):
    """SimSiam (Chen & He 2021), CIFAR variant with a 2-layer projector."""

    def __init__(self, backbone="resnet18", proj_dim=2048, pred_hidden=512):
        super().__init__()
        self.encoder, self.feat_dim = _cifar_resnet(backbone)
        self.head = MLPHead(self.feat_dim, proj_dim, proj_dim, last_bn=True)
        self.predictor = MLPHead(proj_dim, pred_hidden, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.head(h)
        return h, z, self.predictor(z)


class BYOLModel(nn.Module):
    """BYOL (Grill et al. 2020): online network + predictor; the target is an EMA copy."""

    def __init__(self, backbone="resnet18", proj_hidden=4096, proj_dim=256):
        super().__init__()
        self.encoder, self.feat_dim = _cifar_resnet(backbone)
        self.head = MLPHead(self.feat_dim, proj_hidden, proj_dim)
        self.predictor = MLPHead(proj_dim, proj_hidden, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.head(h)
        return h, z, self.predictor(z)


class Classifier(nn.Module):
    """Encoder + linear classifier (supervised and semi-supervised baselines, fine-tuning)."""

    def __init__(self, backbone="resnet18", num_classes=10):
        super().__init__()
        self.encoder, self.feat_dim = _cifar_resnet(backbone)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


@torch.no_grad()
def extract_features(encoder, x_uint8, normalize_fn, batch_size=2000):
    """Encoder outputs h for un-augmented images (eval mode, mixed precision)."""
    was_training = encoder.training
    encoder.eval()
    out = []
    for i in range(0, x_uint8.shape[0], batch_size):
        xb = normalize_fn(x_uint8[i:i + batch_size]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            h = encoder(xb)
        out.append(h.float())
    encoder.train(was_training)
    return torch.cat(out)


def load_encoder(path, device="cuda"):
    """Rebuild the encoder of any checkpoint written by dlt.train / dlt.semisup."""
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    backbone = a.get("backbone", "resnet18")
    method = a.get("method")
    if method in ("simclr", "supcon"):
        model = ContrastiveModel(backbone)
    elif method == "simsiam":
        model = SimSiamModel(backbone)
    elif method == "byol":
        model = BYOLModel(backbone)
    else:
        n_cls = sd["fc.weight"].shape[0]
        model = Classifier(backbone, n_cls)
    model.load_state_dict(sd)
    enc = model.encoder.to(device).to(memory_format=torch.channels_last).eval()
    return enc, a

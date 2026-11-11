"""Réseaux PyTorch du carnet (CNN contraint, DDPM, générateur pix2pix), séparés du détecteur pour que
celui-ci puisse fonctionner sans PyTorch, avec les mêmes réseaux exportés en ONNX (export_onnx.py)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import features as FZ


class BayarConv(nn.Module):
    def __init__(self, cin=1, cout=3, k=5):
        super().__init__(); self.k = k
        self.w = nn.Parameter(torch.randn(cout, cin, k, k) * 0.01)
        m = torch.ones(cout, cin, k, k); m[:, :, k // 2, k // 2] = 0
        c = torch.zeros(cout, cin, k, k); c[:, :, k // 2, k // 2] = -1
        self.register_buffer("mask", m); self.register_buffer("center", c)

    def forward(self, x):
        w = self.w * self.mask
        w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-8) + self.center
        return F.conv2d(x, w, padding=self.k // 2)


def srm_bank():
    kv = torch.tensor(FZ.K_KV); s3 = torch.zeros(5, 5); s3[1:4, 1:4] = torch.tensor(FZ.K_SRM3)
    d1 = torch.zeros(5, 5); d1[2, 2], d1[2, 3] = -1, 1
    return torch.stack([kv, s3, d1])[:, None]


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                  nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ForensicUNet(nn.Module):
    def __init__(self, encoder="efficientnet_b0"):
        super().__init__()
        import timm
        self.bayar = BayarConv(1, 3, 5)
        self.register_buffer("srm", srm_bank())
        # pretrained=False : les poids viennent de cnn_best.pt (aucun téléchargement)
        self.enc = timm.create_model(encoder, features_only=True, pretrained=False, in_chans=7)
        ch = self.enc.feature_info.channels()
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch[-1], 1))
        self.ups = nn.ModuleList()
        cur = ch[-1]
        for c in reversed(ch[:-1]):
            self.ups.append(Up(cur, c, max(32, c))); cur = max(32, c)
        self.seg = nn.Conv2d(cur, 1, 1)

    def forward(self, x):
        x7 = torch.cat([x, self.bayar(x), F.conv2d(x, self.srm, padding=2)], 1)
        feats = self.enc(x7)
        logit = self.cls(feats[-1]).squeeze(1)
        d = feats[-1]
        for up, sk in zip(self.ups, reversed(feats[:-1])):
            d = up(d, sk)
        seg = F.interpolate(self.seg(d), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logit, seg[:, 0]


# ------------------------------------------------------------------ branche diffusion (§8)
class TEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, 4 * d))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class ResB(nn.Module):
    def __init__(self, i, o, td=512):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, i), nn.Conv2d(i, o, 3, padding=1)
        self.t = nn.Linear(td, o)
        self.n2, self.c2 = nn.GroupNorm(8, o), nn.Conv2d(o, o, 3, padding=1)
        self.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        return self.c2(F.silu(self.n2(h))) + self.sk(x)


class DiffUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = TEmb(128); self.inc = nn.Conv2d(1, 64, 3, padding=1)
        self.d1, self.d2, self.d3, self.mid = ResB(64, 64), ResB(64, 128), ResB(128, 256), ResB(256, 256)
        self.u3, self.u2, self.u1 = ResB(512, 128), ResB(256, 64), ResB(128, 64)
        self.no, self.out = nn.GroupNorm(8, 64), nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        te = self.te(t); x0 = self.inc(x)
        h1 = self.d1(x0, te); h2 = self.d2(F.avg_pool2d(h1, 2), te); h3 = self.d3(F.avg_pool2d(h2, 2), te)
        m = self.mid(F.avg_pool2d(h3, 2), te)
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), h3], 1), te)
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(F.silu(self.no(u)))


# ---- générateur pix2pix (démonstration)
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


class UNetG(nn.Module):
    def __init__(self, cin=3, b=48):
        super().__init__()
        self.e1, self.e2, self.e3, self.bt = cbr(cin, b), cbr(b, 2 * b), cbr(2 * b, 4 * b), cbr(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), cbr(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), cbr(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), cbr(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); bt = self.bt(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(bt), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.tanh(self.out(d1))

"""Détecteur hybride de faux scans CT — reprise fidèle du carnet (detection_faux_scans_LIDC, §4 à §10).

Charge les modèles entraînés lors de la 4e exécution (dossier MODELES, par défaut Documents\\GPU_ROCm\\out) :
  modeles_fusion.joblib  classifieur forensique des patchs, fusion locale, méta-classifieur HGB, fusion robuste
  cnn_best.pt            branche CNN (couche de Bayar + SRM, EfficientNet-B0, décodeur U-Net)
  ddpm_attaquant.pt      DDPM (sert aussi de défenseur : SEPARATE_DEFENDER_DDPM = False dans le carnet)
  gan_G.pt               générateur pix2pix (uniquement pour fabriquer des falsifications de démonstration)

Aucun réentraînement : les sorties sont celles du modèle évalué dans le mémoire (tableau 3.6).
"""
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import joblib
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.stats import kurtosis

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import features as FZ  # noqa: E402  (copie de features.py écrit par le carnet, §5)

def _dossier_modeles():
    """Dossier des modèles : variable DETECTEUR_MODELES, sinon sorties du carnet, sinon copie du dossier complet."""
    if os.environ.get("DETECTEUR_MODELES"):
        return Path(os.environ["DETECTEUR_MODELES"])
    for p in (ICI / "modeles", Path.home() / "Documents" / "GPU_ROCm" / "out",
              ICI.parent / "3_Resultats" / "modeles_entraines"):
        if (p / "modeles_fusion.joblib").exists():
            return p
    raise FileNotFoundError("modeles_fusion.joblib introuvable : définir DETECTEUR_MODELES")


MODELES = _dossier_modeles()
DEVICE = torch.device(os.environ.get("DETECTEUR_DEVICE", "cpu"))
P, S = 64, 32                       # patchs 64 × 64 au pas de 32 (CFG du carnet)
T_DIFF, T_STAR = 1000, 150
HU_MIN, HU_MAX = -1024, 3071
WIN_LO, WIN_HI = -1100.0, 300.0
GLOBAL_ATTACKS = {"bruit", "fgsm"}


# ------------------------------------------------------------------ prétraitement (§2)
def lung_mask_slice(hu):
    m = hu < -400
    lab, n = ndi.label(m)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = m & ~np.isin(lab, border)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1][:2]
    keep = [i + 1 for i in order if sizes[i] > 1500]
    m = np.isin(lab, keep)
    m = ndi.binary_closing(m, iterations=3)
    return ndi.binary_fill_holes(m)


def dicom_to_hu(ds):
    return (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1))
            + float(getattr(ds, "RescaleIntercept", 0))).astype(np.int16)


def hu_to_net(x):
    return x / 1000.0


def to_gan(x):
    return (np.clip((x - WIN_LO) / (WIN_HI - WIN_LO), 0, 1) * 2 - 1).astype(np.float32)


def from_gan(y):
    return (y + 1) / 2 * (WIN_HI - WIN_LO) + WIN_LO


# ------------------------------------------------------------------ branche forensique (§6)
def grid_coords(lung):
    out = []
    for y in range(0, lung.shape[0] - P + 1, S):
        for x in range(0, lung.shape[1] - P + 1, S):
            if lung[y:y + P, x:x + P].mean() > 0.5:
                out.append((y, x))
    return out


def patch_desc(R, img, y, x):
    r = R["med"][y:y + P, x:x + P]; k = R["kv"][y:y + P, x:x + P]
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    a1x, a1y = FZ.ac1(k); e = FZ.band_energy(k, 6)
    blk = img[y:y + P, x:x + P].astype(np.float32)
    return [sig, a1x, a1y, e[5] / (e[2] + 1e-9), k.std(), kurtosis(k.ravel()), *e, blk.mean(), blk.std()]


def slice_patch_features(img, lung, coords=None):
    coords = coords if coords is not None else grid_coords(lung)
    if not coords:
        return coords, np.zeros((0, 27), np.float32)
    R = FZ.residuals(img)
    f = np.array([patch_desc(R, img, y, x) for y, x in coords], np.float32)
    z = np.abs(FZ.robust_z(f[:, :12]))
    dup = FZ.duplicate_map(img, img > -900)
    d = np.array([[dup[y:y + P, x:x + P].mean()] for y, x in coords], np.float32)
    return coords, np.nan_to_num(np.hstack([f, z, d]))


# ------------------------------------------------------------------ branche CNN (§7)
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


# ------------------------------------------------------------------ fusion robuste (§9, ajout du 09/11/2026)
class PositiveLogit:
    """Régression logistique à poids positifs ou nuls (classe du carnet, nécessaire pour relire la fusion)."""
    def __init__(self, cols, C=1.0):
        self.cols, self.C = list(cols), C

    def fit(self, V, y):
        X = V[:, self.cols]; self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd; y = np.asarray(y, float); n, d = Z.shape
        sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9)) / n
        lam = 1.0 / (self.C * n)

        def f(t):
            z = Z @ t[:-1] + t[-1]; p = 1 / (1 + np.exp(-z)); g = sw * (p - y)
            return (np.sum(sw * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * t[:-1] @ t[:-1],
                    np.r_[Z.T @ g + lam * t[:-1], g.sum()])
        self.t = minimize(f, np.r_[np.full(d, 0.1), 0.0], jac=True, method="L-BFGS-B",
                          bounds=[(0, None)] * d + [(None, None)]).x
        return self

    def predict_proba(self, V):
        p = 1 / (1 + np.exp(-(((V[:, self.cols] - self.mu) / self.sd) @ self.t[:-1] + self.t[-1])))
        return np.c_[1 - p, p]


def agg(p):
    p = np.sort(p)[::-1]
    return [p[0], p[:3].mean(), p[:10].mean(), (p > 0.5).sum()] if len(p) else [0, 0, 0, 0]


def top3(p):
    return float(np.mean(np.sort(p)[::-1][:3])) if len(p) else 0.0


def pixel_map(coords, vals, shape=(512, 512)):
    acc = np.zeros(shape); cnt = np.zeros(shape)
    for (y, x), v in zip(coords, vals):
        acc[y:y + P, x:x + P] += v; cnt[y:y + P, x:x + P] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)


# ------------------------------------------------------------------ chargement des modèles
class Detecteur:
    def __init__(self, dossier=MODELES, device=DEVICE):
        self.dossier, self.device = Path(dossier), device
        import __main__
        __main__.PositiveLogit = PositiveLogit      # la fusion a été enregistrée depuis le carnet (__main__)
        m = joblib.load(self.dossier / "modeles_fusion.joblib")
        self.forensic_clf, self.local_fuser, self.meta = m["forensic_clf"], m["local_fuser"], m["meta"]
        self.fusion, self.fusion_choix = m["fusion"], m["fusion_choix"]
        self.cnn = ForensicUNet().to(device)
        self.cnn.load_state_dict(torch.load(self.dossier / "cnn_best.pt", map_location=device)); self.cnn.eval()
        self.ddpm = DiffUNet().to(device)
        self.ddpm.load_state_dict(torch.load(self.dossier / "ddpm_attaquant.pt", map_location=device)); self.ddpm.eval()
        betas = torch.linspace(1e-4, 0.02, T_DIFF, device=device)
        self.betas, self.alphas = betas, 1 - betas
        self.abar = torch.cumprod(self.alphas, 0)
        sel = self.dossier / "selection_fusion.json"
        if not sel.exists():                     # dossier complet : les JSON sont dans scores_et_tableaux
            sel = self.dossier.parent / "scores_et_tableaux" / "selection_fusion.json"
        self.selection = json.load(open(sel, encoding="utf-8")) if sel.exists() else None
        self._gan = None

    # ---------------------------------------------------------- branches
    @torch.no_grad()
    def cnn_patch(self, img, coords):
        if not coords:
            return np.zeros(0), np.zeros((0, P, P))
        X = np.stack([img[y:y + P, x:x + P] for y, x in coords]).astype(np.float32)
        pr, sg = [], []
        for b in range(0, len(X), 128):
            lo, se = self.cnn(torch.tensor(hu_to_net(X[b:b + 128]))[:, None].to(self.device))
            pr.append(torch.sigmoid(lo).cpu().numpy()); sg.append(torch.sigmoid(se).cpu().numpy())
        return np.concatenate(pr), np.concatenate(sg)

    @torch.no_grad()
    def diffusion_score(self, patches_hu, n_draws=3, bs=128, seed=0):
        g = torch.Generator(device=self.device).manual_seed(seed)   # tirages reproductibles d'une analyse à l'autre
        ab = self.abar[T_STAR]
        out = []
        for b in range(0, len(patches_hu), bs):
            x0 = torch.tensor(to_gan(patches_hu[b:b + bs])[:, None], device=self.device)
            t = torch.full((len(x0),), T_STAR, device=self.device, dtype=torch.long)
            acc = torch.zeros(len(x0), device=self.device)
            for _ in range(n_draws):
                eps = torch.randn(x0.shape, generator=g, device=self.device)
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                x0h = (xt - (1 - ab).sqrt() * self.ddpm(xt, t)) / ab.sqrt()
                acc += ((x0h - x0) ** 2).mean((1, 2, 3))
            out.append((acc / n_draws).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def branch_outputs(self, img, lung):
        coords, f = slice_patch_features(img, lung)
        if not coords:
            return None
        p_for = self.forensic_clf.predict_proba(f)[:, 1]
        p_cnn, seg = self.cnn_patch(img, coords)
        X = np.stack([img[y:y + P, x:x + P] for y, x in coords])
        s = self.diffusion_score(X)
        s_z = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
        return dict(coords=coords, p_for=p_for, p_cnn=p_cnn, seg=seg, s_diff=s_z,
                    gfeat=FZ.image_features(img, lung)[0], dup=float(f[:, -1].max()))

    def fused_local(self, bo):
        return self.local_fuser.predict_proba(np.c_[bo["p_for"], bo["p_cnn"], bo["s_diff"]])[:, 1]

    def meta_features(self, bo):
        return np.r_[bo["gfeat"], agg(self.fused_local(bo)), agg(bo["p_for"]), agg(bo["p_cnn"]), agg(bo["s_diff"])]

    def branch_vector(self, bo):
        return np.array([top3(bo["p_for"]), top3(bo["p_cnn"]), top3(bo["s_diff"]), top3(self.fused_local(bo))])

    # ---------------------------------------------------------- analyse complète d'une coupe
    def analyser(self, img):
        img = np.asarray(img, np.int16)
        lung = lung_mask_slice(img)
        bo = self.branch_outputs(img, lung)
        if bo is None:
            return dict(ok=False, lung=lung, message="Aucun patch pulmonaire exploitable dans cette coupe "
                                                     "(poumon absent ou trop petit) : la coupe ne peut pas être analysée.")
        v = self.branch_vector(bo)
        m = np.nan_to_num(self.meta_features(bo))[None]
        loc = self.fused_local(bo)
        seg_map = np.zeros(img.shape); cnt = np.zeros(img.shape)
        for (y, x), sgm in zip(bo["coords"], bo["seg"]):
            seg_map[y:y + P, x:x + P] += sgm; cnt[y:y + P, x:x + P] += 1
        seg_map = np.where(cnt > 0, seg_map / np.maximum(cnt, 1), 0)
        return dict(
            ok=True, lung=lung, n_patchs=len(bo["coords"]),
            score=float(self.fusion.predict_proba(v[None])[0, 1]),
            score_hgb=float(self.meta.predict_proba(m)[0, 1]),
            branches=dict(forensique=float(v[0]), cnn=float(v[1]), diffusion_z=float(v[2]), fusion_locale=float(v[3])),
            carte=pixel_map(bo["coords"], loc, img.shape), carte_cnn=seg_map,
            duplication=bo["dup"])

    # ---------------------------------------------------------- générateur GAN (démonstration uniquement)
    def gan(self):
        if self._gan is None:
            self._gan = UNetG().to(self.device)
            self._gan.load_state_dict(torch.load(self.dossier / "gan_G.pt", map_location=self.device))
            self._gan.eval()
        return self._gan


# ------------------------------------------------------------------ attaques de démonstration (§3)
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


CENTER = np.zeros((P, P), np.float32); CENTER[P // 4:3 * P // 4, P // 4:3 * P // 4] = 1


def mad_sigma(img, mask):
    f = img.astype(np.float32)
    r = f - ndi.median_filter(f, 3)
    v = r[mask] if mask.any() else r.ravel()
    return 1.4826 * np.median(np.abs(v - np.median(v))) * 1.2


def blend_patch(img, patch, cy, cx, sigma, rng):
    out = img.astype(np.float32).copy()
    h, w = patch.shape
    y0, x0 = cy - h // 2, cx - w // 2
    yy, xx = np.mgrid[:h, :w]
    win = np.exp(-(((yy - h / 2) / (h / 3.2)) ** 2 + ((xx - w / 2) / (w / 3.2)) ** 2))
    reg = out[y0:y0 + h, x0:x0 + w]
    out[y0:y0 + h, x0:x0 + w] = reg * (1 - win) + (patch + rng.normal(0, sigma, patch.shape)) * win
    m = np.zeros(img.shape, bool); m[y0:y0 + h, x0:x0 + w] = win > 0.25
    return np.clip(np.round(out), HU_MIN, HU_MAX).astype(np.int16), m


def random_lung_point(lung, margin, rng):
    er = ndi.binary_erosion(lung, iterations=margin)
    idx = np.argwhere(er)
    if len(idx) == 0:
        idx = np.argwhere(lung)
    return idx[rng.integers(len(idx))]


def atk_copy_move(img, lung, rng):
    s = int(rng.integers(24, 40))
    a = random_lung_point(lung, s, rng)
    for _ in range(100):
        b = random_lung_point(lung, s, rng)
        if np.abs(a - b).max() > s + 4:
            break
    out = img.copy()
    out[b[0] - s // 2:b[0] - s // 2 + s, b[1] - s // 2:b[1] - s // 2 + s] = \
        img[a[0] - s // 2:a[0] - s // 2 + s, a[1] - s // 2:a[1] - s // 2 + s]
    m = np.zeros(img.shape, bool); m[b[0] - s // 2:b[0] - s // 2 + s, b[1] - s // 2:b[1] - s // 2 + s] = True
    return out, m


def atk_noise(img, rng):
    sig = rng.uniform(4, 12)
    body = img > -900
    out = img.astype(np.float32) + rng.normal(0, sig, img.shape) * body
    return np.clip(np.round(out), HU_MIN, HU_MAX).astype(np.int16), body


def atk_inpaint(img, lung, nod, rng):
    cy, cx, r = nod
    h = int(2 * r + 16) | 1
    mk = np.zeros(img.shape, np.uint8); cv2.circle(mk, (int(cx), int(cy)), int(r + 3), 1, -1)
    lo, hi = -1024.0, 400.0
    u8 = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(u8, mk, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
    patch = ndi.gaussian_filter(inp, 1.0)[cy - h // 2:cy - h // 2 + h, cx - h // 2:cx - h // 2 + h]
    return blend_patch(img, patch, cy, cx, mad_sigma(img, lung), rng)


@torch.no_grad()
def gan_fill(det, patch_hu, cond):
    G = det.gan()
    y = torch.tensor(to_gan(patch_hu)[None, None], device=det.device)
    m = torch.tensor(CENTER, device=det.device)[None, None].expand_as(y)
    c = torch.tensor([cond], device=det.device).view(-1, 1, 1, 1).float().expand_as(y)
    x = torch.cat([y * (1 - m), m, c], 1)
    return from_gan((y * (1 - m) + G(x) * m)[0, 0].cpu().numpy())


def paste_center(img, lung, new_patch, cy, cx, rng):
    c = new_patch[P // 2 - 20:P // 2 + 20, P // 2 - 20:P // 2 + 20]
    return blend_patch(img, c, cy, cx, mad_sigma(img, lung), rng)


def atk_gan_inject(det, img, lung, rng):
    cy, cx = random_lung_point(lung, P // 2 + 2, rng)
    return paste_center(img, lung, gan_fill(det, img[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2], 1), cy, cx, rng)


def atk_gan_remove(det, img, lung, nod, rng):
    cy, cx, _ = nod
    return paste_center(img, lung, gan_fill(det, img[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2], 0), cy, cx, rng)


ATTAQUES = {
    "copier_deplacer": "Copier-déplacer (une région du poumon recopiée ailleurs)",
    "bruit": "Ajout de bruit gaussien (σ de 4 à 12 HU) sur tout le corps",
    "gan_injection": "Injection d’un nodule par GAN (pix2pix entraîné dans le carnet)",
    "gan_suppression": "Suppression d’un nodule par GAN (coupe avec nodule annoté)",
    "inpainting": "Suppression d’un nodule par inpainting de Telea (coupe avec nodule annoté)",
}


def appliquer_attaque(det, nom, img, nods, graine=0):
    """Attaques du carnet. Différence volontaire : les pixels hors du champ de vue (valeur de remplissage du
    scanner, < -1024 HU) sont conservés. Dans le carnet, l'écrêtage à [-1024, 3071] HU les modifiait sur toutes les
    coupes falsifiées et jamais sur les authentiques : un artefact qu'un classifieur peut apprendre (le
    méta-classifieur HGB y est sensible, la fusion robuste non)."""
    rng = np.random.default_rng(graine)
    lung = lung_mask_slice(img)
    if nom == "copier_deplacer":
        out, m = atk_copy_move(img, lung, rng)
    elif nom == "bruit":
        out, m = atk_noise(img, rng)
    elif nom == "gan_injection":
        out, m = atk_gan_inject(det, img, lung, rng)
    elif nom in ("gan_suppression", "inpainting"):
        if not nods:
            raise ValueError("Cette attaque supprime un nodule : choisissez une coupe qui en contient un.")
        nod = tuple(int(round(v)) for v in nods[0])
        out, m = (atk_gan_remove(det, img, lung, nod, rng) if nom == "gan_suppression"
                  else atk_inpaint(img, lung, nod, rng))
    else:
        raise ValueError(nom)
    return np.where(img < HU_MIN, img, out).astype(np.int16), m


# ------------------------------------------------------------------ module de sécurité (§10)
CRITICAL_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "RescaleSlope", "RescaleIntercept",
                 "Rows", "Columns"]


def dicom_fingerprint(ds):
    h = hashlib.sha256(ds.PixelData)
    for t in CRITICAL_TAGS:
        h.update(str(getattr(ds, t, "")).encode())
    return h.hexdigest()


def empreinte_image(img):
    return hashlib.sha256(np.ascontiguousarray(img, np.int16).tobytes()).hexdigest()


class AuditLog:
    """Journal d'audit chaîné : chaque entrée contient l'empreinte de la précédente."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self.path.touch(exist_ok=True)

    def entrees(self):
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    def _last(self):
        e = self.entrees()
        return e[-1]["entry_hash"] if e else "0" * 64

    def append(self, record):
        rec = dict(record, prev_hash=self._last(), time=datetime.datetime.now().isoformat(timespec="seconds"))
        rec["entry_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def verify(self):
        prev = "0" * 64
        for i, rec in enumerate(self.entrees()):
            h = rec.pop("entry_hash")
            if rec["prev_hash"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != h:
                return False, i
            prev = h
        return True, None


def verdict(score, tau1, tau2, hash_ok=True):
    if not hash_ok or score > tau2:
        return "ALERTE"
    return "A VERIFIER" if score > tau1 else "CONFORME"

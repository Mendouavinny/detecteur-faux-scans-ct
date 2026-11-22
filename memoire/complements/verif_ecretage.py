# --- Vérification de l'écrêtage à -1024 HU : coupes réelles et internes écrêtées comme les coupes falsifiées ---
# Coupes falsifiées de M3Dsynth et coupes réelles d'origine (même coupe LIDC-IDRI), patients d'entraînement / validation exclus.
import os, sys, io, json, glob, time, zipfile, subprocess, urllib.request, urllib.parse, math, random, re
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc", "pydicom", "timm"], check=False)
import configparser
configparser.SafeConfigParser = getattr(configparser, "SafeConfigParser", configparser.ConfigParser)
import numpy as np, pandas as pd, pydicom, torch, torch.nn as nn, torch.nn.functional as F, timm, joblib
from scipy import ndimage as ndi
from scipy.stats import kurtosis
from PIL import Image
from sklearn.metrics import roc_auc_score
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
(Path.home() / ".pylidcrc").write_text("[dicom]\npath = /tmp/lidc\nwarn = True\n")
import pylidc as pl
T0 = time.time(); DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ch(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

# ---------- modèles de l'exécution v5 (sortie du carnet principal ajoutée en entrée) ----------
SRC = Path(next(iter(glob.glob("/kaggle/input/**/modeles_sklearn.joblib", recursive=True)))).parent
sys.path.insert(0, str(SRC.parent)); sys.path.insert(0, str(SRC))
FEAT = next(iter(glob.glob("/kaggle/input/**/features.py", recursive=True)))
sys.path.insert(0, str(Path(FEAT).parent)); import features as FZ
SK = joblib.load(SRC / "modeles_sklearn.joblib")
P, S, T_STAR = 64, 32, 150


def lung_mask_slice(hu):
    m = hu < -400
    lab, n = ndi.label(m)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = m & ~np.isin(lab, border)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1)); order = np.argsort(sizes)[::-1][:2]
    keep = [i + 1 for i in order if sizes[i] > 1500]
    m = np.isin(lab, keep); m = ndi.binary_closing(m, iterations=3)
    return ndi.binary_fill_holes(m)


def hu_to_net(x):
    return x / 1000.0


WIN_LO, WIN_HI = -1100.0, 300.0
def to_gan(x):
    return (np.clip((x - WIN_LO) / (WIN_HI - WIN_LO), 0, 1) * 2 - 1).astype(np.float32)


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
    def __init__(self, encoder="efficientnet_b0", pretrained=False):
        super().__init__()
        self.bayar = BayarConv(1, 3, 5)
        self.register_buffer("srm", srm_bank())
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=7)
        ch_ = self.enc.feature_info.channels()
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch_[-1], 1))
        self.ups = nn.ModuleList(); cur = ch_[-1]
        for c in reversed(ch_[:-1]):
            self.ups.append(Up(cur, c, max(32, c))); cur = max(32, c)
        self.seg = nn.Conv2d(cur, 1, 1)

    def forward(self, x):
        x7 = torch.cat([x, self.bayar(x), F.conv2d(x, self.srm, padding=2)], 1)
        feats = self.enc(x7); logit = self.cls(feats[-1]).squeeze(1); d = feats[-1]
        for up, sk in zip(self.ups, reversed(feats[:-1])):
            d = up(d, sk)
        return logit, F.interpolate(self.seg(d), size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


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


cnn = ForensicUNet().to(DEVICE); cnn.load_state_dict(torch.load(SRC / "cnn_best.pt", map_location=DEVICE)); cnn.eval()
ddpm = DiffUNet().to(DEVICE); ddpm.load_state_dict(torch.load(SRC / "ddpm_defenseur.pt", map_location=DEVICE)); ddpm.eval()
betas = torch.linspace(1e-4, 0.02, 1000, device=DEVICE); abar = torch.cumprod(1 - betas, 0)
ch("modèles chargés")


def grid_coords(lung):
    return [(y, x) for y in range(0, lung.shape[0] - P + 1, S) for x in range(0, lung.shape[1] - P + 1, S) if lung[y:y + P, x:x + P].mean() > 0.5]


def patch_desc(R, img, y, x):
    r = R["med"][y:y + P, x:x + P]; k = R["kv"][y:y + P, x:x + P]
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    a1x, a1y = FZ.ac1(k); e = FZ.band_energy(k, 6)
    blk = img[y:y + P, x:x + P].astype(np.float32)
    return [sig, a1x, a1y, e[5] / (e[2] + 1e-9), k.std(), kurtosis(k.ravel()), *e, blk.mean(), blk.std()]


def slice_patch_features(img, lung):
    coords = grid_coords(lung)
    if not coords:
        return coords, np.zeros((0, 27), np.float32)
    R = FZ.residuals(img)
    f = np.array([patch_desc(R, img, y, x) for y, x in coords], np.float32)
    z = np.abs(FZ.robust_z(f[:, :12]))
    dup = FZ.duplicate_map(img, img > -900)
    d = np.array([[dup[y:y + P, x:x + P].mean()] for y, x in coords], np.float32)
    return coords, np.nan_to_num(np.hstack([f, z, d]))


def agg(p):
    p = np.sort(p)[::-1]
    return [p[0], p[:3].mean(), p[:10].mean(), (p > 0.5).sum()] if len(p) else [0, 0, 0, 0]


@torch.no_grad()
def branch(img):
    lung = lung_mask_slice(img); coords, f = slice_patch_features(img, lung)
    if not coords:
        return None
    g = FZ.image_features(img, lung)[0]
    p_for = SK["forensic_clf"].predict_proba(f)[:, 1]
    X = np.stack([img[y:y + P, x:x + P] for y, x in coords]).astype(np.float32)
    p_cnn = np.concatenate([torch.sigmoid(cnn(torch.tensor(hu_to_net(X[b:b + 256]))[:, None].to(DEVICE))[0]).float().cpu().numpy() for b in range(0, len(X), 256)])
    s = []
    for b in range(0, len(X), 256):
        x0 = torch.tensor(to_gan(X[b:b + 256])[:, None], device=DEVICE); t = torch.full((len(x0),), T_STAR, device=DEVICE, dtype=torch.long); acc = torch.zeros(len(x0), device=DEVICE)
        for _ in range(3):
            eps = torch.randn_like(x0); xt = abar[T_STAR].sqrt() * x0 + (1 - abar[T_STAR]).sqrt() * eps
            x0h = (xt - (1 - abar[T_STAR]).sqrt() * ddpm(xt, t)) / abar[T_STAR].sqrt(); acc += ((x0h - x0) ** 2).mean((1, 2, 3))
        s.append((acc / 3).cpu().numpy())
    s = np.concatenate(s); sz = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
    fl = SK["local_fuser"].predict_proba(np.c_[p_for, p_cnn, sz, np.abs(sz)])[:, 1]
    mf = np.r_[g, agg(fl), agg(p_for), agg(p_cnn), agg(sz), agg(np.abs(sz))]
    f2 = SK["loc2"].predict_proba(np.c_[p_for, p_cnn])[:, 1]
    mf2 = np.r_[g, agg(f2), agg(p_for), agg(p_cnn)]
    top = lambda v: float(np.mean(np.sort(v)[::-1][:3]))
    return dict(forensique=top(p_for), cnn=top(p_cnn), diffusion=top(np.abs(sz)),
                hybride=float(SK["meta"].predict_proba(np.nan_to_num(mf)[None])[0, 1]),
                sans_diffusion=float(SK["meta2"].predict_proba(np.nan_to_num(mf2)[None])[0, 1]),
                _g=np.nan_to_num(np.asarray(g, float)), _agg=np.nan_to_num(np.r_[agg(f2), agg(p_for), agg(p_cnn)]),
                _int=np.nan_to_num(np.asarray(FZ.intensity_features(img, lung), float)))


# ---------- échantillons extraits par le carnet CPU (m3d_extract.py) ----------
SELF = next(iter(glob.glob("/kaggle/input/**/m3d_selection.csv", recursive=True)))
SEL = pd.read_csv(SELF); M3 = Path(SELF).parent / "m3d"
L = pd.read_csv("https://raw.githubusercontent.com/grip-unina/M3Dsynth/main/data/LIDC.csv")
scans = pl.query(pl.Scan).all()
mods = ["pix2pix", "cycle", "diffusion"]
STAT = {mod: len(list((M3 / mod / "scan").glob("*/*.tiff"))) if (M3 / mod / "scan").exists() else 0 for mod in mods}
EXCL = set()
ch(f"échantillons : {SEL.groupby(['mod', 'ty']).size().to_dict()} ; coupes extraites {STAT}")

# ---------- coupes réelles d'origine (LIDC-IDRI, TCIA) ----------
NBIA = "https://services.cancerimagingarchive.net/nbia-api"; _TOK = {}
def tok():
    if "v" not in _TOK:
        d = urllib.parse.urlencode({"username": "nbia_guest", "password": "", "client_id": "NBIA", "grant_type": "password"}).encode()
        with urllib.request.urlopen(urllib.request.Request(NBIA + "/oauth/token", data=d), timeout=60) as r:
            _TOK["v"] = json.load(r)["access_token"]
    return _TOK["v"]
def fetch(uid):
    last = None
    for a in range(4):
        for url, auth in ((NBIA + "/services/v1/getImage?SeriesInstanceUID=" + uid, False), (NBIA + "/services/v2/getImage?SeriesInstanceUID=" + uid, True)):
            try:
                rq = urllib.request.Request(url, headers={"User-Agent": "lidc-tampering-research"})
                if auth:
                    rq.add_header("Authorization", "Bearer " + tok())
                with urllib.request.urlopen(rq, timeout=300) as r:
                    return r.read()
            except Exception as e:
                last = e
        time.sleep(5 * (a + 1))
    raise RuntimeError(str(last)[:80])
by_pid = {}
for s in scans:
    by_pid.setdefault(s.patient_id, []).append(s)
REAL, ROWS, MISS = {}, [], []
for pid, grp in SEL.groupby("orig_id"):
    Lp = L[L.orig_id == pid]
    cand = by_pid.get(pid, [])
    if len(Lp):
        cand = sorted(cand, key=lambda s: abs(float(s.slice_thickness) - float(Lp.spacing_z.iloc[0])) + abs(float(s.pixel_spacing) - float(Lp.spacing_x.iloc[0])))
    if not cand:
        MISS.append(pid); continue
    try:
        data_ = fetch(cand[0].series_instance_uid)
    except Exception:
        MISS.append(pid); continue
    with zipfile.ZipFile(io.BytesIO(data_)) as z:
        ds = [pydicom.dcmread(io.BytesIO(z.read(m))) for m in z.namelist() if m.lower().endswith(".dcm")]
    ds.sort(key=lambda d: -float(d.ImagePositionPatient[2]))                 # même ordre que M3Dsynth (z décroissant)
    for r in grp.itertuples():
        zi = int(r.coord_z)
        if not 0 <= zi < len(ds):
            continue
        d = ds[zi]; raw = d.pixel_array.astype(np.int32)
        slope, icpt = float(getattr(d, "RescaleSlope", 1)), float(getattr(d, "RescaleIntercept", 0))
        hu_raw = np.round(raw * slope + icpt)
        hu_real = np.clip(hu_raw, -1024, 3071).astype(np.int16)          # écrêtage identique à celui des coupes falsifiées
        if r.mod == "real":
            REAL[r.img_id] = hu_real
            ROWS.append(dict(img=r.img_id, pid=pid, mod="real", ty=r.ty, y=0, key=r.img_id, min_brut=float(hu_raw.min()))); continue
        fs = [str(M3 / r.mod / "scan" / r.img_id / f"slide{zi:04d}.tiff")]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            continue
        man = np.asarray(Image.open(fs[0]), dtype=np.int32)
        if man.shape != raw.shape:
            continue
        off = int(np.median(man - raw))
        hu_m = np.clip(np.round((man - off) * slope + icpt), -1024, 3071).astype(np.int16)
        frac = float((man - off == raw).mean())
        REAL[r.img_id + "|" + r.mod] = hu_m
        ROWS.append(dict(img=r.img_id, pid=pid, mod=r.mod, ty=r.ty, y=1, key=r.img_id + "|" + r.mod, identiques=round(frac, 4),
                         min_brut=float(np.round((man - off) * slope + icpt).min())))
ch(f"coupes prêtes : {len(ROWS)} ; patients manquants : {len(MISS)}")

# ---------- scores et AUC par générateur et par type ----------
D = pd.DataFrame(ROWS)
SC = {}
for k in D.key:
    try:
        SC[k] = branch(REAL[k])
    except Exception as e:
        SC[k] = None
D = D[D.key.map(lambda k: SC.get(k) is not None)].reset_index(drop=True)
for b in ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion"):
    D[b] = [SC[k][b] for k in D.key]
ch("scores M3Dsynth calculés")


def diff_ci(y, a, b, g, n=2000, seed=0):
    y, a, b, g = map(np.asarray, (y, a, b, g)); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); Dd = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() != y[ii].max():
            Dd.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 3), round(float(np.percentile(Dd, 2.5)), 3), round(float(np.percentile(Dd, 97.5)), 3)]


RES = dict(extraction=STAT, patients=int(D.pid.nunique()), patients_manquants=MISS,
           identiques_hors_zone=round(float(D[D.y == 1].identiques.median()), 4) if (D.y == 1).any() else None, groupes={})
for mod in mods:
    for ty in ("inj", "rem"):
        m = ((D["mod"] == mod) & (D.ty == ty)) | ((D["mod"] == "real") & (D.ty == ty) & D.img.isin(D[(D["mod"] == mod) & (D.ty == ty)].img))
        sub = D[m]
        if sub.y.nunique() < 2:
            continue
        RES["groupes"][f"{mod}_{ty}"] = dict(falsifiees=int(sub.y.sum()), reelles=int((1 - sub.y).sum()),
                                            auc={b: round(float(roc_auc_score(sub.y, sub[b])), 3) for b in ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion")},
                                            fusion_moins_forensique=diff_ci(sub.y, sub.hybride, sub.forensique, sub.pid),
                                            fusion2_moins_forensique=diff_ci(sub.y, sub.sans_diffusion, sub.forensique, sub.pid))
# coupes réelles portant un nodule (origine des suppressions) contre coupes réelles sans nodule (origine des injections)
r = D[D["mod"] == "real"]
if r.ty.nunique() == 2:
    RES["reelles_nodule_vs_sans"] = {b: round(float(roc_auc_score((r.ty == "rem").astype(int), r[b])), 3) for b in ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion")}
for ty in ("inj", "rem"):
    sub = D[(D.ty == ty)]
    if sub.y.nunique() == 2:
        RES["groupes"][f"tous_{ty}"] = dict(falsifiees=int(sub.y.sum()), reelles=int((1 - sub.y).sum()),
                                           auc={b: round(float(roc_auc_score(sub.y, sub[b])), 3) for b in ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion")},
                                           fusion_moins_forensique=diff_ci(sub.y, sub.hybride, sub.forensique, sub.pid))
D.to_csv("/kaggle/working/m3dsynth_scores_ecretees.csv", index=False)
RES["m3d_min_inf_1024"] = {"reelles": round(float((D[D.y == 0].min_brut < -1024).mean()), 3), "falsifiees_avant_ecretage": round(float((D[D.y == 1].min_brut < -1024).mean()), 3)}

# ---------- transfert des descripteurs globaux : appris sur le test interne, appliqués à CT-GAN et M3Dsynth ----------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
def auc_ci(y, a, g, n=2000, seed=0):
    y, a, g = map(np.asarray, (y, a, g)); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); A = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() != y[ii].max():
            A.append(roc_auc_score(y[ii], a[ii]))
    return [round(float(roc_auc_score(y, a)), 3), round(float(np.percentile(A, 2.5)), 3), round(float(np.percentile(A, 97.5)), 3)]


BR = ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion")
clipv = lambda a: np.clip(a, -1024, 3071).astype(np.int16)
try:
    CT = np.load(next(iter(glob.glob("/kaggle/input/**/coupes_test.npz", recursive=True))), allow_pickle=True)
    te, ex = list(CT["te"]), list(CT["ex"])
    yte0, gte0, ate0 = CT["te_y"], CT["te_pid"], CT["te_attaque"]; yex0, gex0, tex0 = CT["ex_y"], CT["ex_uuid"], CT["ex_type"]
    mte = np.array([float(x.min()) for x in te]); mex = np.array([float(x.min()) for x in ex])
    RES["minimum"] = dict(interne_authentiques_inf_1024=round(float((mte[yte0 == 0] < -1024).mean()), 3),
                          interne_falsifiees_inf_1024=round(float((mte[yte0 == 1] < -1024).mean()), 3),
                          ctgan_authentiques_inf_1024=round(float((mex[yex0 == 0] < -1024).mean()), 3),
                          ctgan_falsifiees_inf_1024=round(float((mex[yex0 == 1] < -1024).mean()), 3),
                          interne_n=[int((yte0 == 0).sum()), int((yte0 == 1).sum())])
    # détecteur trivial : « la coupe ne descend pas sous -1024 HU »
    RES["indice_trivial_auc"] = dict(interne=round(float(roc_auc_score(yte0, (mte >= -1024).astype(float))), 3),
                                     ctgan=round(float(roc_auc_score(yex0, (mex >= -1024).astype(float))), 3),
                                     m3d_avant_ecretage=round(float(roc_auc_score(D.y, (D.min_brut >= -1024).astype(float))), 3))
    ch(f"minimums : {RES['minimum']} ; indice trivial : {RES['indice_trivial_auc']}")
    TEu = [branch(x) for x in te]; ch("interne brut rescoré")
    TEc = [branch(clipv(x)) for x in te]; ch("interne écrêté scoré")
    EXu = [branch(x) for x in ex]; EXc = [branch(clipv(x)) for x in ex]; ch("CT-GAN brut et écrêté scorés")
    okt = np.array([a is not None and b is not None for a, b in zip(TEu, TEc)]); oke = np.array([a is not None and b is not None for a, b in zip(EXu, EXc)])
    yte, gte, ate, mte_ = yte0[okt], gte0[okt], ate0[okt], mte[okt]; yex, gex, tex = yex0[oke], gex0[oke], tex0[oke]
    TEu = [o for o, k in zip(TEu, okt) if k]; TEc = [o for o, k in zip(TEc, okt) if k]
    EXu = [o for o, k in zip(EXu, oke) if k]; EXc = [o for o, k in zip(EXc, oke) if k]
    su = {b: np.array([o[b] for o in TEu]) for b in BR}; sc = {b: np.array([o[b] for o in TEc]) for b in BR}
    eu = {b: np.array([o[b] for o in EXu]) for b in BR}; ec = {b: np.array([o[b] for o in EXc]) for b in BR}
    fmtm = np.isin(tex, ["FM", "TM"])
    sansremb = (yte == 1) | (mte_ >= -1024)        # authentiques sans valeurs sous -1024 HU, toutes les falsifiées
    FR = {}
    for b in BR:
        FR[b] = dict(interne_brut=auc_ci(yte, su[b], gte), interne_ecrete=auc_ci(yte, sc[b], gte),
                     ecrete_moins_brut=diff_ci(yte, sc[b], su[b], gte),
                     interne_brut_authentiques_sans_rembourrage=round(float(roc_auc_score(yte[sansremb], su[b][sansremb])), 3),
                     ctgan_brut=round(float(roc_auc_score(yex, eu[b])), 3), ctgan_ecrete=round(float(roc_auc_score(yex, ec[b])), 3),
                     ctgan_FM_TM_ecrete=round(float(roc_auc_score(yex[fmtm], ec[b][fmtm])), 3))
    FR["fusion_moins_forensique_interne_ecrete"] = diff_ci(yte, sc["hybride"], sc["forensique"], gte)
    FR["fusion2_moins_forensique_interne_ecrete"] = diff_ci(yte, sc["sans_diffusion"], sc["forensique"], gte)
    FR["fusion_moins_forensique_ctgan_ecrete"] = diff_ci(yex, ec["hybride"], ec["forensique"], gex)
    FR["n_interne"] = [int((yte == 0).sum()), int((yte == 1).sum()), int((sansremb & (yte == 0)).sum())]
    fam = {}
    for a in sorted(set(ate[yte == 1])):
        m = (yte == 0) | (ate == a)
        fam[str(a)] = dict(n=int((ate == a).sum()), **{b: [round(float(roc_auc_score(yte[m], su[b][m])), 3), round(float(roc_auc_score(yte[m], sc[b][m])), 3)] for b in ("forensique", "cnn", "hybride", "sans_diffusion")})
    FR["par_famille_brut_ecrete"] = fam
    RES["modeles_figes"] = FR
    ch("AUC des modèles figés calculées")
    M = [SC[k] for k in D.key]; ym = D.y.values
    feats = {"globaux_40": lambda o: o["_g"], "bruit_global_38": lambda o: o["_g"][:-2], "intensite_21": lambda o: o["_int"],
             "agregats_branches_12": lambda o: o["_agg"], "globaux_plus_agregats": lambda o: np.r_[o["_g"], o["_agg"]]}
    mk = {"RL": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
          "HGB": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)}
    TR = {}
    for fn, f in feats.items():
        Xu = np.array([f(o) for o in TEu]); Xt = np.array([f(o) for o in TEc]); Xe = np.array([f(o) for o in EXc]); Xm = np.array([f(o) for o in M])
        for mn, m in mk.items():
            cv = np.zeros(len(yte)); cvu = np.zeros(len(yte))
            for tr, va in GroupKFold(5).split(Xt, yte, gte):
                cv[va] = m().fit(Xt[tr], yte[tr]).predict_proba(Xt[va])[:, 1]
                cvu[va] = m().fit(Xu[tr], yte[tr]).predict_proba(Xu[va])[:, 1]
            mod = m().fit(Xt, yte); pe = mod.predict_proba(Xe)[:, 1]; pm = mod.predict_proba(Xm)[:, 1]
            r = dict(interne_hors_pli_brut=round(float(roc_auc_score(yte, cvu)), 3), interne_hors_pli_ecrete=auc_ci(yte, cv, gte),
                     ctgan=round(float(roc_auc_score(yex, pe)), 3), ctgan_FM_vs_TM=round(float(roc_auc_score(yex[fmtm], pe[fmtm])), 3),
                     m3dsynth=round(float(roc_auc_score(ym, pm)), 3))
            for mod_ in mods:
                mm = (D["mod"].values == mod_) | ((D["mod"].values == "real") & D.img.isin(D[D["mod"] == mod_].img).values)
                if len(set(ym[mm])) == 2:
                    r["m3d_" + mod_] = round(float(roc_auc_score(ym[mm], pm[mm])), 3)
            TR[f"{fn}_{mn}"] = r
    RES["transfert_ecrete"] = TR
except Exception as e:
    import traceback; traceback.print_exc(); RES["verif_erreur"] = repr(e)[:300]
ch("transfert terminé")

# ---------- TruFor sur M3Dsynth ----------
try:
    TFD = Path("/kaggle/working/trufor_m3d"); (TFD / "in").mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(D.key):
        Image.fromarray((np.clip((REAL[k].astype(np.float32) + 1350) / 1500, 0, 1) * 255).round().astype(np.uint8)).convert("RGB").save(TFD / "in" / f"{i:04d}.png")
    subprocess.run("cd /tmp && rm -rf TruFor && git clone -q --depth 1 https://github.com/grip-unina/TruFor.git && "
                   "wget -q -O /tmp/tfw.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip && pip install -q yacs", shell=True, check=True)
    with zipfile.ZipFile("/tmp/tfw.zip") as z:
        z.extractall("/tmp/TruFor/TruFor_train_test/pretrained_models")
    w = next(iter(glob.glob("/tmp/TruFor/TruFor_train_test/pretrained_models/**/*.pth.tar", recursive=True)))
    tp = "/tmp/TruFor/TruFor_train_test/test.py"; src_ = open(tp).read()
    open(tp, "w").write(src_.replace("map_location=torch.device(device))", "map_location=torch.device(device), weights_only=False)"))
    r = subprocess.run(f'cd /tmp/TruFor/TruFor_train_test && python test.py -g 0 -in {TFD}/in -out {TFD}/out -exp trufor_ph3 TEST.MODEL_FILE "{w}"', shell=True, capture_output=True, text=True)
    print("TruFor", r.returncode, r.stderr[-400:])
    D["trufor"] = [float(np.load(TFD / "out" / f"{i:04d}.png.npz")["score"]) for i in range(len(D))]
    RES["trufor"] = {}
    for mod_ in mods:
        for ty in ("inj", "rem"):
            m = ((D["mod"] == mod_) & (D.ty == ty)) | ((D["mod"] == "real") & (D.ty == ty) & D.img.isin(D[(D["mod"] == mod_) & (D.ty == ty)].img))
            sub = D[m]
            if sub.y.nunique() == 2:
                RES["trufor"][f"{mod_}_{ty}"] = dict(auc=round(float(roc_auc_score(sub.y, sub.trufor)), 3),
                                                     forensique_moins_trufor=diff_ci(sub.y, sub.forensique, sub.trufor, sub.pid))
    D.to_csv("/kaggle/working/m3dsynth_scores_ecretees.csv", index=False)
except Exception as e:
    import traceback; traceback.print_exc(); RES["trufor_erreur"] = repr(e)[:200]
print("ECR_DEBUT"); print(json.dumps(RES, separators=(",", ":"))); print("ECR_FIN")
ch("fin")

"""Exporte les modèles de l'exécution finale (v7, 300 patients) pour l'application web.

À lancer dans un carnet Kaggle qui a en entrée la sortie du carnet principal (dossier out/ avec
modeles_sklearn.joblib, cnn_best.pt, ddpm_defenseur.pt, gan_G.pt et features.py), internet activé :

    !git clone -q --depth 1 https://github.com/Mendouavinny/detecteur-faux-scans-ct /kaggle/working/depot
    !python /kaggle/working/depot/outils_site/export_kaggle_v7.py

Produit /kaggle/working/export_web_v7/ :
  cnn.onnx, ddpm.onnx, gan.onnx           réseaux (ONNX, opset 17), contrôlés contre PyTorch
  hgb_forensique.json, hgb_meta.json,      classifieurs à gradient boosting (arbres)
  hgb_meta2.json
  fusion.json                              fusions locales (régressions logistiques), constantes, versions
  reference.json                           sorties Python sur les 10 coupes d'exemple du dépôt (contrôle de parité)
  morceaux/ et morceaux.json               réseaux découpés en morceaux de 4 Mo avec empreinte SHA-256
Si un secret Kaggle HF_TOKEN est attaché au carnet, le dossier est aussi envoyé dans le dépôt Hugging Face
MENDOUA/detecteur-faux-scans-ct-modeles, sous v7/, d'où l'application web le lit.
"""
import glob
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "timm", "pydicom", "huggingface_hub"],
               check=False)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import joblib  # noqa: E402
import sklearn  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pydicom  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.stats import kurtosis  # noqa: E402

T0 = time.time()
def ch(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

DEPOT = Path(__file__).resolve().parent.parent
SRC = Path(sorted(glob.glob("/kaggle/input/**/out/modeles_sklearn.joblib", recursive=True))[0]).parent
cands = [SRC.parent / "features.py"] + [Path(p) for p in glob.glob("/kaggle/input/**/features.py", recursive=True)]
FEAT = next(p for p in cands if p.exists())
OUTD = Path("/kaggle/working/export_web_v7"); OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FEAT.parent))
import features as FZ  # noqa: E402  descripteurs de l'exécution v7
sys.path.insert(1, str(DEPOT))
import reseaux_torch as RT  # noqa: E402  mêmes classes de réseaux que le carnet
ch(f"modèles : {SRC} ; descripteurs : {FEAT}")

# contrôle : les descripteurs de la coupe (image_features) sont-ils ceux du dépôt, que l'application reprend ?
src_v7 = inspect.getsource(FZ.image_features)
src_depot = (DEPOT / "features.py").read_text(encoding="utf-8")
print("image_features identique au dépôt :", src_v7.strip() in src_depot, flush=True)

P, S, T_STAR = 64, 32, 150
HU_MIN, HU_MAX = -1024, 3071
WIN_LO, WIN_HI = -1100.0, 300.0
torch.manual_seed(0)


# ------------------------------------------------------------------ réseaux → ONNX
def charger(cls, nom):
    m = cls(); m.load_state_dict(torch.load(SRC / nom, map_location="cpu")); return m.eval()


class DDPMEps(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, t):
        return self.m(x, t)


cnn = charger(RT.ForensicUNet, "cnn_best.pt")
ddpm = charger(RT.DiffUNet, "ddpm_defenseur.pt" if (SRC / "ddpm_defenseur.pt").exists() else "ddpm_attaquant.pt")
gan = charger(RT.UNetG, "gan_G.pt")
print("paramètres : CNN", sum(p.numel() for p in cnn.parameters()), "DDPM", sum(p.numel() for p in ddpm.parameters()), flush=True)

RES = {
    "cnn.onnx": (cnn, (torch.randn(8, 1, P, P),), ["x"], ["logit", "seg"]),
    "ddpm.onnx": (DDPMEps(ddpm), (torch.randn(8, 1, P, P), torch.full((8,), T_STAR, dtype=torch.long)), ["x", "t"], ["eps"]),
    "gan.onnx": (gan, (torch.randn(1, 3, P, P),), ["x"], ["y"]),
}
for nom, (m, ex, ins, outs) in RES.items():
    axes = {k: {0: "n"} for k in ins + outs}
    try:
        torch.onnx.export(m, ex, str(OUTD / nom), input_names=ins, output_names=outs, dynamic_axes=axes, opset_version=17,
                          dynamo=False)
    except TypeError:                                   # PyTorch < 2.5 : pas d'argument dynamo
        torch.onnx.export(m, ex, str(OUTD / nom), input_names=ins, output_names=outs, dynamic_axes=axes, opset_version=17)
    ex2 = tuple(torch.cat([e, e[:3]]) for e in ex)
    with torch.no_grad():
        ref = m(*ex2)
    ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    got = ort.InferenceSession(str(OUTD / nom), providers=["CPUExecutionProvider"]).run(None, {k: e.numpy() for k, e in zip(ins, ex2)})
    err = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    print(f"{nom}: {(OUTD / nom).stat().st_size / 1e6:.1f} Mo, écart ONNX/PyTorch {err:.2e}", flush=True)
ch("réseaux exportés")

# ------------------------------------------------------------------ classifieurs → JSON
SK = joblib.load(SRC / "modeles_sklearn.joblib")
print("clés :", list(SK), "scikit-learn", sklearn.__version__, flush=True)


def hgb_json(clf, nom):
    assert getattr(clf, "n_trees_per_iteration_", 1) == 1
    arbres = []
    for it in clf._predictors:
        nd = it[0].nodes
        assert not nd["is_categorical"].any()
        arbres.append([nd["feature_idx"].tolist(), [float(x) for x in nd["num_threshold"]], nd["left"].tolist(),
                       nd["right"].tolist(), nd["is_leaf"].tolist(), [float(x) for x in nd["value"]],
                       nd["missing_go_to_left"].tolist()])
    (OUTD / nom).write_text(json.dumps(dict(baseline=float(np.ravel(clf._baseline_prediction)[0]),
                                             n_features=int(clf.n_features_in_), trees=arbres), separators=(",", ":")))
    print(f"{nom}: {len(arbres)} arbres, {clf.n_features_in_} entrées", flush=True)


def lr(m):
    return dict(coef=[float(c) for c in m.coef_[0]], intercept=float(m.intercept_[0]))


hgb_json(SK["forensic_clf"], "hgb_forensique.json")
hgb_json(SK["meta"], "hgb_meta.json")
hgb_json(SK["meta2"], "hgb_meta2.json")
abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), 0)
cfg = dict(version="v7", patients=300, ecretage=[HU_MIN, HU_MAX],
           local=dict(lr(SK["local_fuser"]), entrees=["p_forensique", "p_cnn", "s_diffusion", "|s_diffusion|"]),
           loc2=dict(lr(SK["loc2"]), entrees=["p_forensique", "p_cnn"]),
           meta=dict(entrees="40 descripteurs de la coupe + agg(fusion locale, forensique, CNN, s, |s|)"),
           meta2=dict(entrees="40 descripteurs de la coupe + agg(fusion locale sans diffusion, forensique, CNN)"),
           ddpm=dict(t_star=T_STAR, abar=float(abar[T_STAR]), win_lo=WIN_LO, win_hi=WIN_HI, tirages=3),
           seuils=dict(tau1=0.5, tau2=0.85),
           versions=dict(scikit_learn=sklearn.__version__, torch=torch.__version__, numpy=np.__version__))
(OUTD / "fusion.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ références sur les coupes d'exemple
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


def grid_coords(lung):
    return [(y, x) for y in range(0, lung.shape[0] - P + 1, S) for x in range(0, lung.shape[1] - P + 1, S)
            if lung[y:y + P, x:x + P].mean() > 0.5]


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


s_cnn = ort.InferenceSession(str(OUTD / "cnn.onnx"), providers=["CPUExecutionProvider"])
s_ddpm = ort.InferenceSession(str(OUTD / "ddpm.onnx"), providers=["CPUExecutionProvider"])
sig = lambda z: 1 / (1 + np.exp(-z))
top3 = lambda v: float(np.mean(np.sort(v)[::-1][:3]))
ref = {}
for f in sorted((DEPOT / "site" / "exemples").glob("*.dcm")):
    ds = pydicom.dcmread(f)
    hu = (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))).astype(np.int16)
    img = np.clip(hu, HU_MIN, HU_MAX).astype(np.int16)                 # lecture écrêtée de l'exécution v7
    lung = lung_mask_slice(img); coords, feats = slice_patch_features(img, lung)
    if not coords:
        continue
    g = np.nan_to_num(np.asarray(FZ.image_features(img, lung)[0], np.float64))
    p_for = SK["forensic_clf"].predict_proba(feats)[:, 1]
    X = np.stack([img[y:y + P, x:x + P] for y, x in coords]).astype(np.float32)
    lo, _ = s_cnn.run(None, {"x": np.ascontiguousarray((X / 1000.0)[:, None], np.float32)})
    p_cnn = sig(lo.astype(np.float64))
    with torch.no_grad():
        lo_t = cnn(torch.tensor(X / 1000.0)[:, None])[0].numpy()
    # diffusion : 3 tirages (graine fixée) ; le navigateur utilise un autre générateur aléatoire
    rng = np.random.default_rng(0); ab = float(abar[T_STAR]); x0 = ((np.clip((X - WIN_LO) / (WIN_HI - WIN_LO), 0, 1) * 2 - 1)[:, None]).astype(np.float32)
    acc = np.zeros(len(X))
    for _ in range(3):
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        xt = (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32)
        e = s_ddpm.run(None, {"x": xt, "t": np.full(len(X), T_STAR, np.int64)})[0]
        acc += (((xt - np.sqrt(1 - ab) * e) / np.sqrt(ab) - x0) ** 2).mean((1, 2, 3))
    s = acc / 3; sz = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
    fl = SK["local_fuser"].predict_proba(np.c_[p_for, p_cnn, sz, np.abs(sz)])[:, 1]
    mf = np.nan_to_num(np.r_[g, agg(fl), agg(p_for), agg(p_cnn), agg(sz), agg(np.abs(sz))])
    f2 = SK["loc2"].predict_proba(np.c_[p_for, p_cnn])[:, 1]
    mf2 = np.nan_to_num(np.r_[g, agg(f2), agg(p_for), agg(p_cnn)])
    ref[f.name] = dict(n=len(coords), p_for=p_for.tolist(), p_cnn=p_cnn.tolist(), ecart_cnn_onnx_torch=float(np.abs(lo - lo_t).max()),
                       sz=sz.tolist(), g=g.tolist(), mf2=mf2.tolist(),
                       meta=float(SK["meta"].predict_proba(mf[None])[0, 1]), meta2=float(SK["meta2"].predict_proba(mf2[None])[0, 1]),
                       forensique=top3(p_for), cnn=top3(p_cnn))
    print(f"{f.name}: {len(coords)} patchs, bruit {top3(p_for):.3f}, CNN {top3(p_cnn):.3f}, "
          f"bruit+CNN {ref[f.name]['meta2']:.3f}, trois branches {ref[f.name]['meta']:.3f}", flush=True)
(OUTD / "reference.json").write_text(json.dumps(ref))
ch("références calculées")

# ------------------------------------------------------------------ morceaux de 4 Mo (comme le dépôt de l'application)
(OUTD / "morceaux").mkdir(exist_ok=True)
man = {}
for nom in ["cnn.onnx", "ddpm.onnx", "gan.onnx"]:
    b = (OUTD / nom).read_bytes(); parts = []
    for i in range(0, len(b), 4 * 1024 * 1024):
        p = f"morceaux/{nom}.part{i // (4 * 1024 * 1024):02d}"
        (OUTD / p).write_bytes(b[i:i + 4 * 1024 * 1024]); parts.append(p)
    man[nom] = dict(morceaux=parts, sha256=hashlib.sha256(b).hexdigest(), taille=len(b))
(OUTD / "morceaux.json").write_text(json.dumps(man, indent=1))
for p in sorted(OUTD.rglob("*")):
    if p.is_file() and "morceaux/" not in str(p):
        print(f"  {p.relative_to(OUTD)}  {p.stat().st_size / 1e6:.2f} Mo  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")

# ------------------------------------------------------------------ envoi vers Hugging Face (si un jeton est attaché)
jeton = None
try:
    from kaggle_secrets import UserSecretsClient
    for nom in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF", "hf_token"):
        try:
            jeton = UserSecretsClient().get_secret(nom)
            if jeton:
                break
        except Exception:
            pass
except ImportError:
    pass
if jeton:
    from huggingface_hub import HfApi
    HfApi(token=jeton).upload_folder(folder_path=str(OUTD), repo_id="MENDOUA/detecteur-faux-scans-ct-modeles", path_in_repo="v7",
                                     commit_message="Modèles de l'exécution v7 (300 patients) pour l'application web",
                                     allow_patterns=["*.json", "morceaux/*"])
    ch("ENVOI_HF_OK : dossier v7/ publié sur Hugging Face")
else:
    ch("ENVOI_HF_ABSENT : aucun secret HF_TOKEN attaché ; fichiers disponibles dans la sortie du carnet")

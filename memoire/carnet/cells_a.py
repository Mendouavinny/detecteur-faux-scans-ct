# Cellules du carnet (partie A : configuration, données, attaques). Chaque élément = (type, source).
CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# Détection automatique de faux scans et d'anomalies dans les bases d'imagerie médicale
### Approche hybride apprentissage profond + forensique numérique — carnet d'expérimentation (LIDC-IDRI)

**Auteur :** Ing. Mendoua Vinny Jordan — Master Recherche SDIA, ENSPD, 2025-2026

Ce carnet implémente le **protocole complet** décrit au chapitre 2 du mémoire :

| § | Contenu |
|---|---|
| 0 | Installation, configuration, graines, GPU |
| 1 | Lecture de LIDC-IDRI (pydicom / pylidc), conversion en HU, masque pulmonaire |
| 2 | Partition 70/15/15 **par patient**, inventaire des coupes |
| 3 | Attaques classiques : copier-déplacer, inpainting, bruit, FGSM (+ camouflage du bruit) |
| 4 | GAN d'inpainting conditionnel (pix2pix) : injection / suppression de nodules |
| 5 | DDPM + inpainting de type RePaint |
| 5 bis | Construction du jeu falsifié (coupes + masques de vérité terrain) |
| 6 | Branche forensique (résidus SRM/KV, autocorrélation, spectre, incohérence locale, duplication) |
| 7 | Branche CNN : couche contrainte de Bayar + SRM → EfficientNet-B0 → U-Net, perte BCE + Dice |
| 8 | Branche diffusion : score d'erreur de reconstruction |
| 9 | Fusion local → global, évaluation (test interne, CT-GAN externe, robustesse) et **génération du tableau 3.6** |
| 10 | Module de sécurité hospitalier : SHA-256, journal d'audit chaîné, politique d'alerte, interface Gradio |
| 11 | Scripts de la preuve de concept SimCT-Tamper (reproduction des chiffres du chapitre 3) |
| 12 | Figures calculées (résidus, bruit, copier-déplacer), sur une coupe réelle LIDC-IDRI si disponible |

**Matériel conseillé :** GPU ≥ 8 Go : Kaggle / Colab (T4, P100) ou GPU local, y compris AMD Radeon RX 7600 sous Windows avec PyTorch ROCm. Sur Kaggle : activer **GPU** et **Internet**, puis « Save & Run All ». Les scanners LIDC-IDRI retenus sont **téléchargés automatiquement depuis le TCIA**, lus puis supprimés au fur et à mesure (l'espace disque reste sous quelques Go). L'ensemble tourne en 3 à 5 heures avec 300 patients.

> Règle forensique fondamentale : **ne jamais rééchantillonner ni exporter en 8 bits** les images avant analyse (cf. §3.1.4 du mémoire). Toutes les opérations travaillent sur les HU natifs 16 bits.
""")

md("## 0. Installation et configuration")
code(r"""
# Installation automatique des paquets manquants (Kaggle / Colab / local)
import importlib.util, subprocess, sys
_need = {"pydicom": "pydicom", "pylidc": "pylidc", "SimpleITK": "SimpleITK", "timm": "timm", "skimage": "scikit-image",
         "cv2": "opencv-python-headless", "tabulate": "tabulate", "gdown": "gdown"}
_missing = [pkg for mod, pkg in _need.items() if importlib.util.find_spec(mod) is None]
if _missing:
    print("installation :", _missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], check=False)
""")
code(r"""
import os, json, math, time, random, hashlib, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import kurtosis
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

# Racine des données : /kaggle/working sur Kaggle, /content sur Colab, Documents\GPU_ROCm sur un PC Windows local
if Path("/kaggle/working").exists():   BASE = Path("/kaggle/working")
elif Path("/content").exists():        BASE = Path("/content")
else:                                  BASE = Path.home() / "Documents" / "GPU_ROCm"
CFG = dict(
    LIDC_ROOT=str(BASE / "LIDC-IDRI"),   # DICOM LIDC-IDRI (téléchargés automatiquement depuis le TCIA si absents)
    CTGAN_ROOT=str(BASE / "CT-GAN"),     # jeu UCI id=520 (test externe), téléchargé si possible
    OUT=str(BASE / "out"),
    FAST=True,                           # sous-ensemble de patients
    MAX_PATIENTS=300,                    # utilisé si FAST (300 patients : 210 entraînement, 45 validation, 45 test)
    DOWNLOAD_LIDC=True,                  # télécharger depuis le TCIA les scanners absents
    KEEP_DICOM=False,                    # supprimer chaque scanner après lecture (économie de disque)
    DOWNLOAD_CTGAN=True,                 # tenter de télécharger le jeu CT-GAN (6 Go) pour le test externe
    N_JOBS=max(1, (os.cpu_count() or 2)),
    SLICES_PER_PATIENT=12,
    P_AUTH=0.5,                          # proportion de coupes authentiques
    PATCH=64, STRIDE=32,
    SEED=42,
    GAN_EPOCHS=30, DDPM_EPOCHS=40, CNN_EPOCHS=30, NODCLS_EPOCHS=3,
    BATCH=128, LR=3e-4,
    T_DIFF=1000, T_STAR=150,
)
OUT = Path(CFG["OUT"]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "slices").mkdir(exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 2   # sous Windows, les workers DataLoader ne voient pas les classes du notebook
T_START = time.time()
def chrono(msg): print(f"[{(time.time() - T_START) / 60:6.1f} min] {msg}", flush=True)

def pmap(fn, items, chunksize=4):
    # Parallélisation CPU (fork sous Linux) ; repli séquentiel sous Windows ou en cas d'erreur.
    items = list(items)
    if os.name != "nt" and CFG["N_JOBS"] > 1 and len(items) > 8:
        try:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(CFG["N_JOBS"]) as pool:
                return pool.map_async(fn, items, chunksize=chunksize).get(timeout=4 * 3600)
        except Exception as e:
            print("parallélisation indisponible, passage séquentiel :", e)
    return [fn(it) for it in items]
print("Périphérique :", DEVICE, "| PyTorch", torch.__version__,
      "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")

# Radeon sous Windows (ROCm) : la BatchNorm de MIOpen doit être compilée à la volée et cette
# compilation échoue (« type_traits file not found »). On force la BatchNorm native de PyTorch
# et on garde MIOpen pour les convolutions. Le premier passage est lent (recherche des noyaux
# MIOpen, quelques minutes), les suivants utilisent le cache.
if DEVICE.type == "cuda" and getattr(torch.version, "hip", None) and os.name == "nt":
    _bn_miopen = F.batch_norm
    def _bn_native(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return _bn_miopen(*args, **kwargs)
    F.batch_norm = _bn_native
    print("Correctif ROCm Windows actif : BatchNorm native, convolutions MIOpen.")
""")

md(r"""
## 1. Lecture de LIDC-IDRI et prétraitement

Les volumes sont lus avec **pylidc**, qui gère l'ordre des coupes et la conversion en unités Hounsfield (pente/ordonnée DICOM).
Les nodules retenus sont ceux annotés par **au moins 3 radiologues** (regroupement `cluster_annotations`).
""")
code(r"""
# Compatibilité pylidc / NumPy récents
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
rc = Path.home() / (".pylidcrc" if os.name != "nt" else "pylidc.conf")
rc.write_text(f"[dicom]\npath = {CFG['LIDC_ROOT']}\nwarn = True\n")
import pylidc as pl

scans = pl.query(pl.Scan).all()
patients = sorted({s.patient_id for s in scans})
if CFG["FAST"]:
    rng = np.random.default_rng(CFG["SEED"])
    patients = sorted(rng.choice(patients, size=min(CFG["MAX_PATIENTS"], len(patients)), replace=False).tolist())
scan_of = {}
for s in scans:
    if s.patient_id in patients and s.patient_id not in scan_of:
        scan_of[s.patient_id] = s
patients = [p for p in patients if p in scan_of]
print(len(scan_of), "patients retenus")

# --- Téléchargement à la demande depuis le TCIA (API publique NBIA) ---
import io, zipfile, shutil, urllib.request, urllib.parse
NBIA = "https://services.cancerimagingarchive.net/nbia-api"
_TOKEN = {}
def _nbia_token():
    if "v" not in _TOKEN:
        data = urllib.parse.urlencode({"username": "nbia_guest", "password": "", "client_id": "NBIA", "grant_type": "password"}).encode()
        with urllib.request.urlopen(urllib.request.Request(NBIA + "/oauth/token", data=data), timeout=60) as r:
            _TOKEN["v"] = json.load(r)["access_token"]
    return _TOKEN["v"]

def series_dir(pid):
    return Path(CFG["LIDC_ROOT"]) / pid / scan_of[pid].series_instance_uid

def ensure_dicom(pid):
    d = series_dir(pid)
    if (d / ".ok").exists() or not CFG["DOWNLOAD_LIDC"]:
        return d
    uid, last = scan_of[pid].series_instance_uid, None
    for attempt in range(4):
        for url, auth in ((NBIA + "/services/v1/getImage?SeriesInstanceUID=" + uid, False),
                          (NBIA + "/services/v2/getImage?SeriesInstanceUID=" + uid, True)):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "lidc-tampering-research"})
                if auth: req.add_header("Authorization", "Bearer " + _nbia_token())
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                d.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for m in z.infolist():
                        if m.filename.lower().endswith(".dcm"):
                            (d / Path(m.filename).name).write_bytes(z.read(m))
                (d / ".ok").write_text("ok")
                return d
            except Exception as e:
                last = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"téléchargement impossible pour {pid} : {last}")

def drop_dicom(pid):
    if not CFG["KEEP_DICOM"]:
        shutil.rmtree(Path(CFG["LIDC_ROOT"]) / pid, ignore_errors=True)
""")
code(r"""
def lung_mask_slice(hu):
    # Segmentation pulmonaire simple : air interne (< -400 HU) non connecté au bord, 2 plus grandes composantes.
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

def load_patient(pid):
    ensure_dicom(pid)
    s = scan_of[pid]
    vol = s.to_volume(verbose=False).astype(np.int16)          # (512, 512, Z) en HU
    nods = []
    for cl in s.cluster_annotations():
        if len(cl) >= 3:
            c = np.mean([a.centroid for a in cl], axis=0)       # (ligne, colonne, coupe)
            d = float(np.mean([a.diameter for a in cl]))
            nods.append((int(round(c[0])), int(round(c[1])), int(round(c[2])), d / s.pixel_spacing / 2))
    return vol, nods, float(s.pixel_spacing), float(s.slice_thickness)

vol, nods, ps, st = load_patient(patients[0])
z = nods[0][2] if nods else vol.shape[2] // 2
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
ax[0].imshow(np.clip((vol[:, :, z] + 1350) / 1500, 0, 1), cmap="gray"); ax[0].set_title(f"{patients[0]} coupe {z}")
ax[1].imshow(lung_mask_slice(vol[:, :, z]), cmap="gray"); ax[1].set_title("Masque pulmonaire")
for a in ax: a.axis("off")
plt.show(); print("nodules (≥3 lecteurs) :", nods)
""")

md(r"""
## 2. Partition par patient et inventaire des coupes

Aucune coupe d'un patient du test n'est vue à l'entraînement (coupes d'un même patient = même dose, même noyau, même anatomie).
""")
code(r"""
rng = np.random.default_rng(CFG["SEED"])
perm = rng.permutation(patients)
n = len(perm); n_tr = int(0.70 * n); n_va = int(0.15 * n)
SPLIT = {p: "train" for p in perm[:n_tr]}
SPLIT.update({p: "val" for p in perm[n_tr:n_tr + n_va]})
SPLIT.update({p: "test" for p in perm[n_tr + n_va:]})
print(pd.Series(SPLIT).value_counts())

def candidate_slices(vol, nods, k, rng):
    # Coupes riches en poumon ; on garantit la présence des coupes à nodule (pour les attaques de suppression).
    Z = vol.shape[2]
    area = np.array([(vol[::4, ::4, z] < -400).mean() for z in range(Z)])
    good = np.where(area > 0.12)[0]
    nod_z = sorted({c[2] for c in nods if 0 <= c[2] < Z})
    rest = [z for z in good if z not in nod_z]
    pick = list(nod_z[: k // 2]) + list(rng.choice(rest, size=min(len(rest), k - min(len(nod_z), k // 2)), replace=False))
    return sorted(int(z) for z in pick)
""")

md(r"""
## 3. Attaques classiques et camouflage du bruit

L'attaquant estime le niveau de bruit local (MAD du résidu médian, éq. 2.1) et ajoute un bruit blanc de même écart-type au contenu inséré, fondu par une fenêtre gaussienne (éq. 2.2).
""")
code(r"""
HU_MIN, HU_MAX = -1024, 3071

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

def atk_copy_move(img, lung, rng, s=None):
    s = s or int(rng.integers(24, 40))
    a = random_lung_point(lung, s, rng)
    for _ in range(100):
        b = random_lung_point(lung, s, rng)
        if np.abs(a - b).max() > s + 4: break
    out = img.copy()
    out[b[0] - s // 2:b[0] - s // 2 + s, b[1] - s // 2:b[1] - s // 2 + s] = img[a[0] - s // 2:a[0] - s // 2 + s, a[1] - s // 2:a[1] - s // 2 + s]
    m = np.zeros(img.shape, bool); m[b[0] - s // 2:b[0] - s // 2 + s, b[1] - s // 2:b[1] - s // 2 + s] = True
    return out, m

def atk_inpaint(img, lung, nod, rng):
    cy, cx, r = nod
    h = int(2 * r + 16) | 1
    mk = np.zeros(img.shape, np.uint8); cv2.circle(mk, (cx, cy), int(r + 3), 1, -1)
    lo, hi = -1024.0, 400.0
    u8 = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(u8, mk, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
    patch = ndi.gaussian_filter(inp, 1.0)[cy - h // 2:cy - h // 2 + h, cx - h // 2:cx - h // 2 + h]
    return blend_patch(img, patch, cy, cx, mad_sigma(img, lung), rng)

def atk_noise(img, rng):
    sig = rng.uniform(4, 12)
    body = img > -900
    out = img.astype(np.float32) + rng.normal(0, sig, img.shape) * body
    return np.clip(np.round(out), HU_MIN, HU_MAX).astype(np.int16), body
""")
code(r"""
# --- FGSM : nécessite un classifieur de nodules (cible de l'attaque adversariale) ---
import timm

def hu_to_net(x):  # normalisation utilisée par tous les réseaux « forensiques » (pas d'écrêtage des poumons)
    return x / 1000.0

class NoduleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model("resnet18", pretrained=True, in_chans=1, num_classes=1)
    def forward(self, x):
        return self.net(x).squeeze(1)

def atk_fgsm(img, lung, model, eps_hu=8.0, P=64):
    # Perturbation FGSM (éq. 2.4) calculée patch par patch sur le poumon, poussant vers « nodule ».
    model.eval()
    out = img.astype(np.float32).copy()
    ys, xs = np.where(lung)
    if len(ys) == 0:
        return img, np.zeros(img.shape, bool)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    coords = [(y, x) for y in range(y0, max(y0 + 1, y1 - P), P) for x in range(x0, max(x0 + 1, x1 - P), P)]
    batch = np.stack([out[y:y + P, x:x + P] for y, x in coords])[:, None]
    t = torch.tensor(hu_to_net(batch).astype(np.float32), device=DEVICE, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(model(t), torch.ones(len(coords), device=DEVICE))
    loss.backward()
    g = -t.grad.sign().cpu().numpy()[:, 0]   # descente sur la perte « nodule »
    for (y, x), gi in zip(coords, g):
        out[y:y + P, x:x + P] += eps_hu * gi * lung[y:y + P, x:x + P]
    return np.clip(np.round(out), HU_MIN, HU_MAX).astype(np.int16), lung.copy()
""")

# Banc d'essai local de la cellule v5 : environnement simulé (sans GPU ni torch), blocs A-D et G.
import sys, types, time, random, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "../simulateur")
import features as FZ
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- faux torch ---
torch = types.SimpleNamespace(no_grad=lambda: (lambda f: f), cuda=types.SimpleNamespace(device_count=lambda: 0))
class _M: pass
nn = types.SimpleNamespace(Module=_M)
Dataset = object
F = None
rng = np.random.default_rng(0)
T_START = time.time()
CFG = dict(SEED=42, CNN_EPOCHS=2, LR=1e-3, BATCH=8)
P, S = 64, 32
GLOBAL_ATTACKS = {"bruit", "fgsm"}
ATT = ["authentique", "copier_deplacer", "bruit", "gan_injection", "fgsm"]
def seed_all(s): np.random.seed(s); random.seed(s)
def pmap(fn, items): return [fn(i) for i in items]
def agg(p):
    p = np.sort(p)[::-1]
    return [p[0], p[:3].mean(), p[:10].mean(), (p > 0.5).sum()] if len(p) else [0, 0, 0, 0]
def tpr_at(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(fpr_target, fpr, tpr))

pids = [f"LIDC-IDRI-{i:04d}" for i in range(60)]
SPLIT = {p: ("train" if i < 42 else "val" if i < 51 else "test") for i, p in enumerate(pids)}
DATA, PF = {}, {}
yy, xx = np.mgrid[:128, :128]
for p in pids:
    for z in range(6):
        atk = ATT[rng.integers(len(ATT))] if rng.random() < 0.5 else "authentique"
        img = (rng.normal(-700, 40, (128, 128))).astype(np.int16)
        lung = ((yy - 64) ** 2 + (xx - 64) ** 2) < 50 ** 2
        mask = np.zeros_like(lung)
        if atk != "authentique": mask[40:70, 40:70] = True
        DATA[(p, z)] = dict(img=img, lung=lung, mask=mask, attack=atk, split=SPLIT[p])
        nb = 12; lab = int(atk != "authentique")
        f = rng.normal(0, 1, (nb, 27)).astype(np.float32); yb = (rng.random(nb) < 0.3 * lab).astype(int); f[:, 0] += yb * 1.5
        PF[(p, z)] = dict(coords=[(i, i) for i in range(nb)], f=f, y=yb, g=rng.normal(lab * 0.3, 1, 40).astype(np.float32))
BO = {}
for k, d in DATA.items():
    if d["split"] in ("val", "test"):
        BO[k] = dict(p_for=rng.random(12), p_cnn=rng.random(12), s_diff=rng.normal(0, 1, 12), gfeat=PF[k]["g"], coords=PF[k]["coords"])
def meta_features(bo): return np.r_[bo["gfeat"], agg(bo["p_for"]), agg(bo["p_cnn"])]
def _meta2_feat(bo): return np.r_[bo["gfeat"], agg(bo["p_for"])]
def slice_scores(bo): return dict(forensique=bo["p_for"].max(), cnn=bo["p_cnn"].max(), diffusion=abs(bo["s_diff"]).max(), hybride=0.5)
tkeys = [k for k in BO if DATA[k]["split"] == "test"]
yt = np.array([int(DATA[k]["attack"] != "authentique") for k in tkeys])
_s = {b: rng.random(len(tkeys)) + 0.3 * yt for b in ("forensique", "cnn", "diffusion", "hybride", "sans_diffusion")}
# CT-GAN simulé
Path("/tmp/mockct").mkdir(exist_ok=True)
lab = pd.DataFrame(dict(type=rng.choice(["FM", "FB", "TM", "TB"], 40), uuid=rng.integers(1000, 1010, 40), slice=rng.integers(0, 50, 40)))
lab.to_csv("/tmp/mockct/labels_exp1.csv", index=False); labs = ["/tmp/mockct/labels_exp1.csv"]
_ek = []
for uuid, grp in lab.groupby("uuid"):
    for j, r in grp.reset_index().iterrows():
        _ek.append((labs[0], uuid, j))
_ety = {k: None for k in _ek}
_ey = np.array([int(lab[lab.uuid == k[1]].reset_index().loc[k[2], "type"] in ("FM", "FB")) for k in _ek])
_eg = np.array([str(k[1]) for k in _ek])
_es = {b: rng.random(len(_ek)) + 0.2 * _ey for b in _s}
class _Scan:
    def cluster_annotations(self): raise RuntimeError("pas de DICOM")
scan_of = {p: _Scan() for p in pids}
def load_patient(pid): return None, [(60, 60, 2, 5.0)], 0.7, 2.5
def drop_dicom(pid): pass
OUT = Path("/tmp")
ex_imgs = {k: (DATA[tkeys[0]]["img"], DATA[tkeys[0]]["lung"]) for k in _ek}
def _cpu_job(k): return ([(0, 0)], np.zeros((1, 27)), np.zeros(40))

src = open("./v5_cell.py").read()
exec(compile(src, "v5", "exec"))

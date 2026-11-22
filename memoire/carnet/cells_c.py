from cells_a import md, code, CELLS

md(r"""
## 6. Branche forensique

Le module `features.py` (identique à celui de la preuve de concept) fournit les résidus SRM/KV, l'autocorrélation (éq. 2.6), l'énergie spectrale par anneaux (éq. 2.7), la normalisation robuste intra-coupe (éq. 2.8) et la détection de duplication par hachage.
On l'applique ici sur des **patchs $64\times64$ au pas de 32** couvrant les poumons.
""")
FEATURES_SRC = open("../simulateur/features.py").read()
code("%%writefile features.py\n" + FEATURES_SRC)
code(r"""
import importlib, features as FZ
importlib.reload(FZ)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score, precision_score, recall_score

S = CFG["STRIDE"]

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

def patch_labels(mask, coords, attack):
    if attack in GLOBAL_ATTACKS:
        return np.ones(len(coords), int)
    return np.array([mask[y:y + P, x:x + P].mean() > 0.3 for y, x in coords], int)

# Extraction pour toutes les coupes (parallélisée sur les cœurs CPU), avec descripteurs globaux de la coupe
def _pf_job(key):
    d = DATA[key]
    coords, f = slice_patch_features(d["img"], d["lung"])
    return coords, f, FZ.image_features(d["img"], d["lung"])[0]

t0 = time.time()
_keys = list(DATA)
PF = {}
for key, (coords, f, g) in zip(_keys, pmap(_pf_job, _keys)):
    PF[key] = dict(coords=coords, f=f, g=g, y=patch_labels(DATA[key]["mask"], coords, DATA[key]["attack"]))
print(f"descripteurs de patchs : {time.time() - t0:.0f}s pour {len(PF)} coupes")

def stack_split(split):
    keys = [k for k in PF if DATA[k]["split"] == split and len(PF[k]["coords"])]
    X = np.vstack([PF[k]["f"] for k in keys]); y = np.concatenate([PF[k]["y"] for k in keys])
    owner = np.concatenate([[i] * len(PF[k]["coords"]) for i, k in enumerate(keys)])
    return keys, X, y, owner

trk, Xtr, ytr, _ = stack_split("train")
# sous-échantillonnage des négatifs (3:1)
neg = np.where(ytr == 0)[0]; pos = np.where(ytr == 1)[0]
sel = np.concatenate([pos, np.random.choice(neg, min(len(neg), 3 * len(pos)), replace=False)])
forensic_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight="balanced", random_state=0)
forensic_clf.fit(Xtr[sel], ytr[sel])
vak, Xva, yva, _ = stack_split("val")
print("AUC patchs (val) — branche forensique :", round(roc_auc_score(yva, forensic_clf.predict_proba(Xva)[:, 1]), 4))
chrono("branche forensique entraînée")
""")

md(r"""
## 7. Branche CNN : couche contrainte de Bayar + SRM → EfficientNet-B0 → décodeur U-Net

* Couche de Bayar (éq. 1.9) : poids centraux fixés à −1, somme des voisins = 1, reprojection à chaque passe avant.
* 3 résidus SRM fixes (KV $5\times5$, SRM $3\times3$, différence horizontale) + l'image (HU/1000) → 7 canaux.
* Perte : BCE (patch) + Dice (carte), éq. 2.9. Augmentation : D4, flou, ré-échantillonnage, bruit, export 8 bits.
""")
code(r"""
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
    def __init__(self, encoder="efficientnet_b0", pretrained=True):
        super().__init__()
        self.bayar = BayarConv(1, 3, 5)
        self.register_buffer("srm", srm_bank())
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=7)
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

def postprocess(img, kind):
    f = img.astype(np.float32)
    if kind == "blur":    f = ndi.gaussian_filter(f, 0.5)
    elif kind == "resample":
        h, w = f.shape; f = cv2.resize(cv2.resize(f, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_LINEAR), (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "noise": f = f + np.random.normal(0, 2, f.shape)
    elif kind == "8bit":
        lo, hi = -1350, 150; f = np.round(np.clip((f - lo) / (hi - lo), 0, 1) * 255) / 255 * (hi - lo) + lo
    return np.clip(np.round(f), HU_MIN, HU_MAX).astype(np.int16)

class PatchSet(Dataset):
    def __init__(self, split, neg_ratio=2, augment=False):
        self.items = []
        for k, pf in PF.items():
            if DATA[k]["split"] != split: continue
            for (y, x), lab in zip(pf["coords"], pf["y"]):
                self.items.append((k, y, x, int(lab)))
        self.augment = augment; self.neg_ratio = neg_ratio; self.resample()
    def resample(self):
        pos = [i for i in self.items if i[3] == 1]; neg = [i for i in self.items if i[3] == 0]
        random.shuffle(neg); self.cur = pos + neg[:self.neg_ratio * max(1, len(pos))]; random.shuffle(self.cur)
    def __len__(self): return len(self.cur)
    def __getitem__(self, i):
        k, y, x, lab = self.cur[i]; d = DATA[k]
        y0, x0 = max(0, y - 8), max(0, x - 8)          # marge pour que flou / ré-échantillonnage n'aient pas d'effet de bord
        big = d["img"][y0:y + P + 8, x0:x + P + 8]; mbig = d["mask"][y0:y + P + 8, x0:x + P + 8]
        if self.augment and random.random() < 0.4:
            big = postprocess(big, random.choice(["blur", "resample", "noise", "8bit"]))
        oy, ox = y - y0, x - x0
        pt = big[oy:oy + P, ox:ox + P].astype(np.float32); m = mbig[oy:oy + P, ox:ox + P].astype(np.float32)
        if d["attack"] in GLOBAL_ATTACKS: m = np.ones_like(m) * lab
        if self.augment:
            kk = random.randint(0, 3); pt, m = np.rot90(pt, kk), np.rot90(m, kk)
            if random.random() < 0.5: pt, m = pt[:, ::-1], m[:, ::-1]
        return torch.tensor(hu_to_net(pt.copy()))[None], torch.tensor(float(lab)), torch.tensor(m.copy())

def dice_loss(seg_logit, m, eps=1.0):
    p = torch.sigmoid(seg_logit)
    num = 2 * (p * m).sum((1, 2)) + eps; den = p.sum((1, 2)) + m.sum((1, 2)) + eps
    return (1 - num / den).mean()

cnn = ForensicUNet().to(DEVICE)
opt = torch.optim.Adam(cnn.parameters(), CFG["LR"])
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
tr_set, va_set = PatchSet("train", augment=True), PatchSet("val", neg_ratio=3)
best, patience = -1, 0
for ep in range(CFG["CNN_EPOCHS"]):
    cnn.train(); tr_set.resample()
    for xb, yb, mb in DataLoader(tr_set, CFG["BATCH"], shuffle=True, num_workers=NUM_WORKERS, drop_last=True):
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logit, seg = cnn(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb) + dice_loss(seg.float(), mb)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    cnn.eval(); ps_, ys_ = [], []
    with torch.no_grad():
        for xb, yb, _ in DataLoader(va_set, 256, num_workers=NUM_WORKERS):
            ps_.append(torch.sigmoid(cnn(xb.to(DEVICE))[0]).cpu().numpy()); ys_.append(yb.numpy())
    auc = roc_auc_score(np.concatenate(ys_), np.concatenate(ps_))
    print(f"CNN époque {ep}: perte {loss.item():.4f} | AUC patchs val {auc:.4f}")
    if auc > best:
        best, patience = auc, 0; torch.save(cnn.state_dict(), OUT / "cnn_best.pt")
    else:
        patience += 1
        if patience >= 6: print("arrêt précoce"); break
cnn.load_state_dict(torch.load(OUT / "cnn_best.pt")); cnn.eval()
chrono(f"branche CNN entraînée (meilleure AUC patchs val {best:.4f})")
""")

md(r"""
## 8. Branche diffusion : score d'erreur de reconstruction (éq. 2.10)

Pour chaque patch : bruitage jusqu'à $t^\ast$, prédiction de $\hat x_0$ en un pas, erreur quadratique moyenne. Branche **non supervisée** (aucun exemple falsifié utilisé).
L'erreur est normalisée au sein de la coupe (score z robuste) ; une région falsifiée peut se reconstruire **moins bien** (contenu étranger) ou **trop bien** (contenu généré, cf. DIRE) : le score de coupe utilise donc l'écart absolu $|z|$.
""")
code(r"""
@torch.no_grad()
def diffusion_score(patches_hu, net=None, t_star=None, n_draws=3, bs=256):
    net = net or ddpm_def; t_star = t_star or CFG["T_STAR"]
    out = []
    for b in range(0, len(patches_hu), bs):
        x0 = torch.tensor(to_gan(patches_hu[b:b + bs])[:, None], device=DEVICE)
        t = torch.full((len(x0),), t_star, device=DEVICE, dtype=torch.long)
        acc = torch.zeros(len(x0), device=DEVICE)
        for _ in range(n_draws):
            eps = torch.randn_like(x0)
            xt = abar[t_star].sqrt() * x0 + (1 - abar[t_star]).sqrt() * eps
            x0h = (xt - (1 - abar[t_star]).sqrt() * net(xt, t)) / abar[t_star].sqrt()
            acc += ((x0h - x0) ** 2).mean((1, 2, 3))
        out.append((acc / n_draws).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
""")

md(r"""
## 9. Fusion local → global et évaluation

* **Local** : pour chaque patch, $[p_{for}, p_{cnn}, z(s_{diff})]$ → régression logistique (apprise sur la validation).
* **Global** : agrégation (max, moyenne top-3, top-10, nombre > 0,5) + descripteurs globaux de la coupe → HGB (appris sur la validation), éq. 2.11.
* Test interne (patients jamais vus), **test externe CT-GAN** (générateur jamais vu) et robustesse aux post-traitements.
""")
code(r"""
@torch.no_grad()
def cnn_patch(img, coords):
    if not coords: return np.zeros(0), np.zeros((0, P, P))
    X = np.stack([img[y:y + P, x:x + P] for y, x in coords]).astype(np.float32)
    pr, sg = [], []
    for b in range(0, len(X), 256):
        lo, se = cnn(torch.tensor(hu_to_net(X[b:b + 256]))[:, None].to(DEVICE))
        pr.append(torch.sigmoid(lo).cpu().numpy()); sg.append(torch.sigmoid(se).cpu().numpy())
    return np.concatenate(pr), np.concatenate(sg)

def cpu_part(img, lung):
    coords, f = slice_patch_features(img, lung)
    return coords, f, FZ.image_features(img, lung)[0]

def gpu_part(img, coords, f, gfeat, keep_seg=True):
    if not coords:
        return None
    p_for = forensic_clf.predict_proba(f)[:, 1]
    p_cnn, seg = cnn_patch(img, coords)
    X = np.stack([img[y:y + P, x:x + P] for y, x in coords])
    s = diffusion_score(X)
    s_z = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))) + 1e-9)
    return dict(coords=coords, p_for=p_for, p_cnn=p_cnn, seg=seg if keep_seg else None, s_diff=s_z, gfeat=gfeat)

def branch_outputs(img, lung):
    return gpu_part(img, *cpu_part(img, lung))

_JOB = {}
def _cpu_job(key):
    img, lung = _JOB[key]
    return cpu_part(img, lung)

def branch_outputs_many(images, keep_seg=True):
    # images : dict clé -> (img, lung). Partie CPU en parallèle, partie GPU séquentielle.
    global _JOB
    _JOB = images; keys = list(images)
    cpu = pmap(_cpu_job, keys)
    out = {k: gpu_part(images[k][0], *c, keep_seg=keep_seg) for k, c in zip(keys, cpu)}
    _JOB = {}
    return {k: v for k, v in out.items() if v is not None}

def agg(p):
    p = np.sort(p)[::-1]
    return [p[0], p[:3].mean(), p[:10].mean(), (p > 0.5).sum()] if len(p) else [0, 0, 0, 0]

def pixel_map(coords, vals, shape=(512, 512)):
    acc = np.zeros(shape); cnt = np.zeros(shape)
    for (y, x), v in zip(coords, vals):
        acc[y:y + P, x:x + P] += v; cnt[y:y + P, x:x + P] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)

# descripteurs CPU déjà calculés au §6 : seule la partie GPU est exécutée ici
BO = {}
for k, d in DATA.items():
    if d["split"] in ("val", "test"):
        bo = gpu_part(d["img"], PF[k]["coords"], PF[k]["f"], PF[k]["g"])
        if bo is not None: BO[k] = bo
chrono("sorties des trois branches calculées (validation + test)")

# --- apprentissage de la fusion sur la VALIDATION uniquement ---
vkeys = [k for k in BO if DATA[k]["split"] == "val"]
L_X = np.vstack([np.c_[BO[k]["p_for"], BO[k]["p_cnn"], BO[k]["s_diff"], np.abs(BO[k]["s_diff"])] for k in vkeys])
L_y = np.concatenate([PF[k]["y"] for k in vkeys])
local_fuser = LogisticRegression(max_iter=2000, class_weight="balanced").fit(L_X, L_y)

def fused_local(bo):
    return local_fuser.predict_proba(np.c_[bo["p_for"], bo["p_cnn"], bo["s_diff"], np.abs(bo["s_diff"])])[:, 1]

def meta_features(bo):
    return np.r_[bo["gfeat"], agg(fused_local(bo)), agg(bo["p_for"]), agg(bo["p_cnn"]), agg(bo["s_diff"]), agg(np.abs(bo["s_diff"]))]

meta = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
meta.fit(np.nan_to_num(np.array([meta_features(BO[k]) for k in vkeys])), np.array([int(DATA[k]["attack"] != "authentique") for k in vkeys]))

def slice_scores(bo):
    return dict(forensique=np.mean(np.sort(bo["p_for"])[::-1][:3]), cnn=np.mean(np.sort(bo["p_cnn"])[::-1][:3]),
                diffusion=np.mean(np.sort(np.abs(bo["s_diff"]))[::-1][:3]),
                hybride=meta.predict_proba(np.nan_to_num(meta_features(bo))[None])[0, 1])

def tpr_at(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(fpr_target, fpr, tpr))

def dice(pred, gt):
    return 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-9)
""")
code(r"""
# --- Test interne ---
tkeys = [k for k in BO if DATA[k]["split"] == "test"]
yt = np.array([int(DATA[k]["attack"] != "authentique") for k in tkeys])
SC = {b: np.array([slice_scores(BO[k])[b] for k in tkeys]) for b in ("forensique", "cnn", "diffusion", "hybride")}
RES = {}
for b, s in SC.items():
    RES[b] = dict(auc=roc_auc_score(yt, s), tpr1=tpr_at(yt, s))
# seuil hybride = 0,5 ; F1 / exactitude
pred_h = (SC["hybride"] >= 0.5).astype(int)
RES["hybride"].update(acc=accuracy_score(yt, pred_h), prec=precision_score(yt, pred_h), rec=recall_score(yt, pred_h), f1=f1_score(yt, pred_h))
# Dice par branche (attaques locales)
for b, getter in (("forensique", lambda bo: bo["p_for"]), ("cnn", None), ("hybride", lambda bo: fused_local(bo))):
    ds = []
    for k in tkeys:
        d = DATA[k]
        if d["attack"] in GLOBAL_ATTACKS or d["attack"] == "authentique": continue
        bo = BO[k]
        if b == "cnn":
            acc = np.zeros((512, 512)); cnt = np.zeros((512, 512))
            for (y, x), sg in zip(bo["coords"], bo["seg"]):
                acc[y:y + P, x:x + P] += sg; cnt[y:y + P, x:x + P] += 1
            pm = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
        else:
            pm = pixel_map(bo["coords"], getter(bo))
        ds.append(dice(pm >= 0.5, d["mask"] & d["lung"]))
    RES[b]["dice"] = float(np.mean(ds)) if ds else float("nan")
# détection par type d'attaque (hybride)
per_attack = pd.Series({a: float(((SC["hybride"] >= 0.5) == (a != "authentique"))[np.array([DATA[k]["attack"] == a for k in tkeys])].mean())
                        for a in ATTACKS if any(DATA[k]["attack"] == a for k in tkeys)})
# AUC par type d'attaque (authentiques du test + coupes de ce type), pour chaque branche
AUC_ATK = {}
auth_idx = np.array([DATA[k]["attack"] == "authentique" for k in tkeys])
for a in ATTACKS[1:]:
    sel = np.array([DATA[k]["attack"] == a for k in tkeys])
    if sel.sum() >= 3 and auth_idx.sum() >= 3:
        m_ = sel | auth_idx
        AUC_ATK[a] = {b: float(roc_auc_score(yt[m_], SC[b][m_])) for b in SC}
print(pd.DataFrame(RES).T.round(3)); print(per_attack.round(3)); print(pd.DataFrame(AUC_ATK).T.round(3))
chrono("test interne terminé")
""")
code(r"""
# --- Test externe : jeu CT-GAN de l'UCI (générateur jamais vu) ---
# Le jeu fournit des scanners DICOM et un CSV de vérité terrain (type, uuid, slice, x, y).
# Types : FM = cancer injecté (faux malin), FB = cancer retiré (faux bénin), TM/TB = authentiques.
# Vérifiez les noms de colonnes et la convention d'indice de coupe en affichant quelques cas.
import pydicom
CT_ROOTS = [Path(CFG["CTGAN_ROOT"])] + [Path(p).parent for p in glob.glob("/kaggle/input/**/labels_exp1.csv", recursive=True)]
def _find_labels():
    out = []
    for root in CT_ROOTS:
        out += glob.glob(str(root / "**" / "labels_exp*.csv"), recursive=True)
    return sorted(set(out))

if not _find_labels() and CFG["DOWNLOAD_CTGAN"]:
    # Archive officielle (UCI id=520, Google Drive, ~6 Go) : téléchargée, extraite puis supprimée.
    try:
        import gdown, zipfile
        root = Path(CFG["CTGAN_ROOT"]); root.mkdir(parents=True, exist_ok=True)
        zp = root / "data.zip"
        gdown.download(id="1R0WD_5IZ3NlyCiOPf1Ex74nBnZYQegwr", output=str(zp), quiet=True)
        with zipfile.ZipFile(zp) as z: z.extractall(root)
        zp.unlink()
    except Exception as e:
        print("téléchargement CT-GAN impossible :", e)

def load_ctgan_scan(folder):
    # Même convention que le code de CT-GAN : coupes triées par ImagePositionPatient[2] croissant, valeurs en HU.
    sl = []
    for f in glob.glob(str(Path(folder) / "*")):
        try:
            sl.append(pydicom.dcmread(f))
        except Exception:
            pass
    sl = [s for s in sl if hasattr(s, "ImagePositionPatient") and hasattr(s, "PixelData")]
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return np.stack([dicom_to_hu_raw(s) for s in sl])

def dicom_to_hu_raw(ds):
    return (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))).astype(np.int16)

EXT = None
labs = _find_labels()
if labs:
    ys_, ss_, ex_imgs, ex_y = [], {b: [] for b in SC}, {}, {}
    for lf in labs:
        lab = pd.read_csv(lf); lab.columns = [c.strip().lower() for c in lab.columns]
        base = Path(lf).parent
        for uuid, grp in lab.groupby("uuid"):
            folder = next(iter(glob.glob(str(base / "**" / str(uuid)), recursive=True)), None)
            if folder is None: continue
            try:
                arr = load_ctgan_scan(folder)
            except Exception as e:
                print("scan illisible", uuid, e); continue
            for j, r in grp.reset_index().iterrows():
                z = int(r["slice"])
                if not 0 <= z < len(arr): continue
                img = arr[z]
                ex_imgs[(lf, uuid, j)] = (img, lung_mask_slice(img))
                ex_y[(lf, uuid, j)] = int(str(r["type"]).strip().upper() in ("FM", "FB"))
    EXB = branch_outputs_many(ex_imgs, keep_seg=False)
    for k, bo in EXB.items():
        sc = slice_scores(bo)
        for b in SC: ss_[b].append(sc[b])
        ys_.append(ex_y[k])
    ys_ = np.array(ys_)
    if len(set(ys_)) == 2:
        EXT = {b: dict(auc=roc_auc_score(ys_, np.array(v)), tpr1=tpr_at(ys_, np.array(v)), n=len(ys_)) for b, v in ss_.items()}
        print(pd.DataFrame(EXT).T.round(3))
else:
    print("Jeu CT-GAN indisponible : test externe non réalisé.")
chrono("test externe CT-GAN terminé")
""")
code(r"""
# --- Robustesse aux post-traitements (hybride, test interne) ---
ROB = {"aucun": RES["hybride"]["auc"]}
for kind in ("blur", "resample", "noise", "8bit"):
    np.random.seed(0)
    imgs = {k: (postprocess(DATA[k]["img"], kind), DATA[k]["lung"]) for k in tkeys}
    bos = branch_outputs_many(imgs, keep_seg=False)
    s = [slice_scores(bos[k])["hybride"] if k in bos else 0.0 for k in tkeys]
    ROB[kind] = roc_auc_score(yt, s)
print(pd.Series(ROB).round(3))
chrono("robustesse évaluée")
""")
code(r"""
# --- Génération automatique du tableau 3.6 du mémoire ---
names = {"forensique": "Forensique (blocs + agrégation)", "cnn": "CNN contraint + EfficientNet-B0 / U-Net",
         "diffusion": "Diffusion (erreur de reconstruction)", "hybride": "Hybride (fusion des trois branches)"}
fmt = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}".replace(".", ",")
T36 = pd.DataFrame([{
    "Branche": names[b],
    "AUC test interne": fmt(RES[b]["auc"]),
    "AUC CT-GAN (externe)": fmt(EXT[b]["auc"]) if EXT else "non disponible",
    "TPR à FPR = 1 %": fmt(RES[b]["tpr1"]),
    "Dice moyen": fmt(RES[b].get("dice")) if b != "diffusion" else "—"} for b in names])
print(T36.to_markdown(index=False))
T36.to_csv(OUT / "tableau_3_6.csv", index=False)
INFO = dict(patients=len(patients), coupes=len(DATA), coupes_test=len(tkeys), falsifiees_test=int(yt.sum()),
            separate_defender_ddpm=SEPARATE_DEFENDER_DDPM, gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU",
            duree_min=round((time.time() - T_START) / 60, 1))
json.dump(dict(INFO=INFO, RES=RES, EXT=EXT, ROB=ROB, per_attack=per_attack.to_dict(), auc_par_attaque=AUC_ATK,
               attaques_echouees=ATK_FAIL), open(OUT / "resultats_lidc.json", "w"), indent=1, default=float)
print(INFO)
""")

md(r"""
### Visualisation : cartes de localisation sur le test
""")
code(r"""
ex = [k for k in tkeys if DATA[k]["attack"] not in GLOBAL_ATTACKS | {"authentique"}][:4]
fig, ax = plt.subplots(len(ex), 3, figsize=(10, 3.4 * len(ex)))
for i, k in enumerate(ex):
    d, bo = DATA[k], BO[k]; w = np.clip((d["img"] + 1350) / 1500, 0, 1)
    ax[i, 0].imshow(w, cmap="gray"); ax[i, 0].set_title(f"{d['attack']} — score {slice_scores(bo)['hybride']:.2f}")
    ax[i, 1].imshow(w, cmap="gray"); ax[i, 1].contour(d["mask"], [0.5], colors="orange"); ax[i, 1].set_title("vérité terrain")
    ax[i, 2].imshow(w, cmap="gray"); ax[i, 2].imshow(np.ma.masked_less(pixel_map(bo["coords"], fused_local(bo)), 0.05), cmap="magma", vmin=0, vmax=1, alpha=0.7)
    ax[i, 2].set_title("carte fusionnée")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "localisation_lidc.png", dpi=150); plt.show()
""")

md(r"""
## 10. Module de sécurité pour les hôpitaux

1. Empreinte **SHA-256** des pixels + attributs critiques, comparée à l'empreinte enregistrée à l'acquisition.
2. Détecteur hybride → score, type d'attaque probable, carte.
3. Politique : **conforme** (< τ₁, choisi pour 1 % de faux positifs sur la validation), **à vérifier** (τ₁–τ₂), **alerte** (> τ₂ ou empreinte discordante).
4. **Journal d'audit chaîné** (chaque entrée contient l'empreinte de la précédente).
5. Notification radiologue / RSSI.
""")
code(r"""
import pydicom, datetime, getpass
CRITICAL_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "RescaleSlope", "RescaleIntercept", "Rows", "Columns"]

def dicom_fingerprint(ds):
    h = hashlib.sha256(ds.PixelData)
    for t in CRITICAL_TAGS:
        h.update(str(getattr(ds, t, "")).encode())
    return h.hexdigest()

def dicom_to_hu(ds):
    return (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))).astype(np.int16)

class AuditLog:
    def __init__(self, path):
        self.path = Path(path); self.path.touch(exist_ok=True)
    def _last(self):
        lines = self.path.read_text().strip().splitlines()
        return json.loads(lines[-1])["entry_hash"] if lines else "0" * 64
    def append(self, record):
        rec = dict(record, prev_hash=self._last(), time=datetime.datetime.now().isoformat(timespec="seconds"))
        rec["entry_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        with self.path.open("a") as f: f.write(json.dumps(rec) + "\n")
        return rec
    def verify(self):
        prev = "0" * 64
        for i, line in enumerate(self.path.read_text().strip().splitlines()):
            rec = json.loads(line); h = rec.pop("entry_hash")
            if rec["prev_hash"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != h:
                return False, i
            prev = h
        return True, None

# Seuils calibrés sur la validation : τ1 -> 1 % de FP, τ2 -> 0,1 % de FP (ou 0,9 si trop peu de données)
val_auth = np.array([slice_scores(BO[k])["hybride"] for k in vkeys if DATA[k]["attack"] == "authentique"])
TAU1 = float(np.quantile(val_auth, 0.99)); TAU2 = float(max(np.quantile(val_auth, 0.999), min(0.9, TAU1 + 0.2)))
AUDIT = AuditLog(OUT / "audit_log.jsonl")
REGISTRY = {}   # empreintes enregistrées à l'acquisition (SOPInstanceUID -> sha256), à brancher sur la base de l'établissement

def verify_dicom(path, user="service-verif"):
    ds = pydicom.dcmread(path)
    fp = dicom_fingerprint(ds); ref = REGISTRY.get(str(getattr(ds, "SOPInstanceUID", "")))
    img = dicom_to_hu(ds); lung = lung_mask_slice(img)
    bo = branch_outputs(img, lung)
    score = slice_scores(bo)["hybride"] if bo else 0.0
    pmap = pixel_map(bo["coords"], fused_local(bo), img.shape) if bo else np.zeros(img.shape)
    hash_ok = (ref is None) or (ref == fp)
    verdict = "ALERTE" if (not hash_ok or score > TAU2) else ("A VERIFIER" if score > TAU1 else "CONFORME")
    rec = AUDIT.append(dict(sop=str(getattr(ds, "SOPInstanceUID", "")), sha256=fp, hash_ok=hash_ok, score=round(float(score), 4),
                            verdict=verdict, user=user, tau1=round(TAU1, 4), tau2=round(TAU2, 4)))
    return verdict, score, img, pmap, rec

print("τ1 =", round(TAU1, 3), "| τ2 =", round(TAU2, 3)); print("Journal intègre :", AUDIT.verify())
""")
code(r"""
# Interface de démonstration (Gradio) — à exécuter sur un poste du réseau interne uniquement
try:
    import gradio as gr
except ImportError:
    gr = None
    print("Gradio non installé : interface de démonstration ignorée (pip install gradio).")
def ui(file):
    verdict, score, img, pmap, rec = verify_dicom(file.name)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    w = np.clip((img + 1350) / 1500, 0, 1)
    ax[0].imshow(w, cmap="gray"); ax[0].set_title("Coupe"); ax[1].imshow(w, cmap="gray")
    ax[1].imshow(np.ma.masked_less(pmap, 0.05), cmap="magma", vmin=0, vmax=1, alpha=0.7); ax[1].set_title("Carte de falsification")
    for a in ax: a.axis("off")
    return f"{verdict} (score {score:.3f})", fig, json.dumps(rec, indent=1)
if gr is not None:
    demo = gr.Interface(fn=ui, inputs=gr.File(label="Fichier DICOM"), outputs=[gr.Textbox(label="Verdict"), gr.Plot(), gr.Code(label="Entrée du journal d'audit")],
                        title="Vérification d'authenticité des images CT")
# demo.launch(share=False)
""")
md(r"""
**Intégration PACS (Orthanc).** Orthanc déclenche un script Python à chaque instance reçue (`OnStoredInstance`) ; le script récupère le fichier via `GET /instances/{id}/file`, appelle `verify_dicom`, puis écrit le verdict dans les métadonnées de l'instance (`PUT /instances/{id}/metadata/...`) et notifie en cas d'alerte. Les analyses restent **sur site** : aucune image ne quitte l'établissement (loi n° 2024/017).

```python
# greffon Python d'Orthanc (extrait)
import orthanc, io
def OnChange(changeType, level, resource):
    if changeType == orthanc.ChangeType.NEW_INSTANCE:
        dcm = orthanc.RestApiGet(f"/instances/{resource}/file")
        verdict, score, *_ = verify_dicom(io.BytesIO(dcm))
        orthanc.RestApiPut(f"/instances/{resource}/metadata/1024", verdict.encode())
orthanc.RegisterOnChangeCallback(OnChange)
```
""")

md(r"""
## 11. Preuve de concept SimCT-Tamper (reproduction des chiffres du chapitre 3)

Ces cellules écrivent puis exécutent les scripts de la preuve de concept (sans GPU, ~10 min sur 2 cœurs). `features.py` a été écrit en §6.
""")
for fn in ("simct.py", "generate.py", "experiments.py", "hybrid.py", "figures.py"):
    src = open(f"../simulateur/{fn}").read()
    src = src.replace("../simulateur/", "").replace('"figures/"', '"fig/"')
    code(f"%%writefile {fn}\n" + src)
code(r"""
!python generate.py
!python experiments.py
!python hybrid.py
!python figures.py
""")
code(r"""
r1, r2 = json.load(open("results.json")), json.load(open("results_hybrid.json"))
print("AUC forensique globale (meilleur) :", round(r1["E1"][r1["best"]]["mean"][4], 3))
print("AUC hybride local→global        :", round(r2["hybrid"]["mean"][4], 3), "±", round(r2["hybrid"]["std"][4], 3))
""")

md(r"""
## 12. Figures illustratives calculées (chapitre 2)

Ce script produit les figures calculées du mémoire : résidus SRM (figure 2.4), noyaux et bruit — NPS, loi en $1/\sqrt{I_0}$ — (figure 2.5) et détection du copier-déplacer (figure 2.6), ainsi que des illustrations complémentaires (principe CT, fenêtrage HU, manipulations classiques, FGSM, diffusion directe, traces spectrales) utiles pour la soutenance. Les figures de la revue de littérature (chapitre 1) sont, elles, reprises de publications et citées avec leur source.
Si une coupe LIDC-IDRI est disponible (cache du §3 bis), elle est sauvegardée dans `real_slice.npy` et utilisée à la place de la coupe simulée.
""")
code(r"""
# real_slice.npy a été enregistré au §3 bis (coupe réelle LIDC-IDRI contenant un nodule)
print("coupe réelle disponible :", Path("real_slice.npy").exists())
""")
code("%%writefile fig_ch1.py\n" + open("../simulateur/fig_ch1.py").read())
code(r"""
!python fig_ch1.py fig_ch1/
from IPython.display import Image, display
for f in sorted(glob.glob("fig_ch1/*.png")):
    display(Image(f, width=900))
""")

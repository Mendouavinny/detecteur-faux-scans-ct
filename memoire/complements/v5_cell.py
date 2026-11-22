# ===== Exécution v5 : compléments demandés par l'audit =====
# Exécutée après le carnet complet et la cellule d'ablation (réutilise DATA, PF, BO, EXB, SC, _s, _es, _meta2...).
# Blocs : A nodules réels -> B test contrôlé -> C seuils fixés sur la validation -> D décomposition H1/H2 sur LIDC-IDRI
#         -> E cascade de Zhang et al. réimplémentée -> F partitions supplémentaires -> G scores par coupe.
import json, time, traceback, gc
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
V5_T0 = time.time()
V5_LIMIT_H = globals().get("V5_LIMIT_H", 9.0)       # pas de nouvelle partition au-delà de cette durée totale
V5_SEEDS = globals().get("V5_SEEDS", [1, 2, 3, 4])
# scanners CT-GAN identifiés pixel à pixel (notebook99d3b07677 v2) comme provenant de patients d'entraînement ou de validation
ID_TRV = ["1009", "1201", "1937", "2031", "2220", "2366", "2484", "2590", "2703", "3246", "4632", "5614", "7668", "7794", "8972",
          "1871", "2024", "2508", "3378"]


def v5out(tag, obj):
    print(f"V5_{tag}_DEBUT")
    print(json.dumps(obj, default=lambda o: float(o) if np.isscalar(o) else str(o), separators=(",", ":")))
    print(f"V5_{tag}_FIN", flush=True)


def v5h():
    return (time.time() - T_START) / 3600


def v5err(tag):
    print(f"V5_ERREUR_{tag}"); traceback.print_exc()


def auc_ci(y, a, g, n=2000, seed=0):
    y, a, g = np.asarray(y), np.asarray(a, float), np.asarray(g)
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); D = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() != y[ii].max():
            D.append(roc_auc_score(y[ii], a[ii]))
    return [round(float(roc_auc_score(y, a)), 3), round(float(np.percentile(D, 2.5)), 3), round(float(np.percentile(D, 97.5)), 3)]


def diff_ci(y, a, b, g, n=2000, seed=0):
    y, a, b, g = np.asarray(y), np.asarray(a, float), np.asarray(b, float), np.asarray(g)
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); D = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() != y[ii].max():
            D.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 3), round(float(np.percentile(D, 2.5)), 3),
            round(float(np.percentile(D, 97.5)), 3)]


def thr_eval(y_te, s_te, s_val_auth, q=99):
    t = float(np.percentile(np.asarray(s_val_auth, float), q))
    y_te, s_te = np.asarray(y_te), np.asarray(s_te, float)
    return dict(seuil=round(t, 5), tpr=round(float((s_te[y_te == 1] > t).mean()), 3), fpr=round(float((s_te[y_te == 0] > t).mean()), 3),
                n_val_auth=int(len(s_val_auth)))


def top3(p):
    return float(np.mean(np.sort(p)[::-1][:3])) if len(p) else 0.0


def oof_scores(X, y, g, mk, k=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        p[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def lab_of(k):
    return int(DATA[k]["attack"] != "authentique")


# ---------- A. Nodules réels de chaque coupe (mêmes règles que le cache du §3 bis) ----------
NOD = {}


def nod_list(pid):
    if pid in NOD:
        return NOD[pid]
    s = scan_of[pid]
    try:
        out = []
        for cl in s.cluster_annotations():
            if len(cl) >= 3:
                c = np.mean([a.centroid for a in cl], axis=0); d = float(np.mean([a.diameter for a in cl]))
                out.append((int(round(c[0])), int(round(c[1])), int(round(c[2])), d))
        NOD[pid] = ("pylidc", out)
    except Exception:
        try:
            _v, nods, ps_, _st = load_patient(pid); drop_dicom(pid); del _v
            NOD[pid] = ("dicom", [(r, c, z, rad * 2 * ps_) for (r, c, z, rad) in nods])
        except Exception:
            NOD[pid] = ("echec", None)
    return NOD[pid]


def nod_mm(key):
    src, L = nod_list(key[0])
    if L is None:
        return None
    ds = [d for (r, c, zz, d) in L if abs(zz - key[1]) <= 1 and 40 < r < 472 and 40 < c < 472]
    return max(ds) if ds else 0.0


try:
    _atk = np.array([DATA[k]["attack"] for k in tkeys]); _g = np.array([str(k[0]) for k in tkeys])
    NODMM = [nod_mm(k) for k in tkeys]
    v5out("NODULES", dict(source=pd.Series([NOD[p][0] for p in set(_g)]).value_counts().to_dict(),
                          auth_avec_nodule=int(sum(1 for v, a in zip(NODMM, _atk) if a == "authentique" and v)),
                          auth=int((_atk == "authentique").sum()),
                          diam_mm=[round(float(np.percentile([v for v in NODMM if v], q)), 1) for q in (10, 50, 90)] if any(NODMM) else None))
except Exception:
    v5err("A"); NODMM = [None] * len(tkeys)

# ---------- B. Test contrôlé : coupes authentiques portant un nodule réel ----------
try:
    known = np.array([v is not None for v in NODMM]); nod = np.array([bool(v) for v in NODMM])
    auth = _atk == "authentique"; inj = _atk == "gan_injection"
    CTRL = {}
    for name, m in {"injGAN_vs_auth_nodule": inj | (auth & nod), "injGAN_vs_auth_sans_nodule": inj | (auth & ~nod & known),
                    "toutes_vs_auth_nodule": (~auth) | (auth & nod), "toutes_vs_auth_sans_nodule": (~auth) | (auth & ~nod & known)}.items():
        yy = yt[m]
        if yy.min() == yy.max():
            continue
        CTRL[name] = dict(pos=int(yy.sum()), neg=int((1 - yy).sum()), auc={b: round(float(roc_auc_score(yy, v[m])), 3) for b, v in _s.items()},
                          fusion_moins_forensique=diff_ci(yy, _s["hybride"][m], _s["forensique"][m], _g[m]),
                          fusion2_moins_forensique=diff_ci(yy, _s["sans_diffusion"][m], _s["forensique"][m], _g[m]))
    m = auth & known
    if nod[m].min() != nod[m].max():
        CTRL["auc_nodule_parmi_authentiques"] = {b: round(float(roc_auc_score(nod[m], v[m])), 3) for b, v in _s.items()}
    # CT-GAN : types par coupe, dans l'ordre de EXB
    TYPE = {}
    for lf in labs:
        lab = pd.read_csv(lf); lab.columns = [c.strip().lower() for c in lab.columns]
        for uuid, grp in lab.groupby("uuid"):
            for j, r in grp.reset_index().iterrows():
                TYPE[(lf, uuid, j)] = str(r["type"]).strip().upper()
    _et = np.array([TYPE.get(k, "?") for k in _ek])
    CTRL["ctgan_types"] = pd.Series(_et).value_counts().to_dict()
    for name, m in {"ctgan_FM_vs_TM": np.isin(_et, ["FM", "TM"]), "ctgan_FB_vs_TM": np.isin(_et, ["FB", "TM"])}.items():
        yy = _ey[m]
        CTRL[name] = dict(pos=int(yy.sum()), neg=int((1 - yy).sum()), auc={b: round(float(roc_auc_score(yy, v[m])), 3) for b, v in _es.items()},
                          fusion_moins_forensique=diff_ci(yy, _es["hybride"][m], _es["forensique"][m], _eg[m]),
                          fusion2_moins_forensique=diff_ci(yy, _es["sans_diffusion"][m], _es["forensique"][m], _eg[m]))
    mx = ~np.isin(_eg, ID_TRV)
    for name, m in {"ctgan_sans_patients_trv": mx, "ctgan_FM_vs_TM_sans_patients_trv": mx & np.isin(_et, ["FM", "TM"])}.items():
        yy = _ey[m]
        CTRL[name] = dict(pos=int(yy.sum()), neg=int((1 - yy).sum()), scanners=int(len(set(_eg[m]))),
                          auc={b: round(float(roc_auc_score(yy, v[m])), 3) for b, v in _es.items()},
                          fusion_moins_forensique=diff_ci(yy, _es["hybride"][m], _es["forensique"][m], _eg[m]))
    v5out("CONTROLE", CTRL)
except Exception:
    v5err("B")

# ---------- C. Seuils fixés sur la validation (99e centile des authentiques), appliqués au test ----------
try:
    _vk = [k for k in BO if DATA[k]["split"] == "val"]
    _yv = np.array([lab_of(k) for k in _vk]); _gv = np.array([str(k[0]) for k in _vk])
    VAL = {b: np.array([slice_scores(BO[k])[b] for k in _vk], float) for b in ("forensique", "cnn", "diffusion")}
    _mk3 = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    VAL["hybride"] = oof_scores(np.nan_to_num(np.array([meta_features(BO[k]) for k in _vk])), _yv, _gv, _mk3)
    VAL["sans_diffusion"] = oof_scores(np.nan_to_num(np.array([_meta2_feat(BO[k]) for k in _vk])), _yv, _gv, _mk3)
    THR = {b: thr_eval(yt, _s[b], VAL[b][_yv == 0]) for b in VAL}
    THR["note"] = "fusion : scores de validation hors pli (5 plis par patient) pour fixer le seuil"
    THR["auc_val_hors_pli"] = {b: round(float(roc_auc_score(_yv, VAL[b])), 3) for b in VAL}
    v5out("SEUILS", THR)
except Exception:
    v5err("C")

# ---------- D. Décomposition H1 / H2 sur LIDC-IDRI (partition principale) ----------
NOISE_COLS = list(range(12)) + list(range(14, 26)); NI_COLS = list(range(26)); ALL_COLS = list(range(27))


def _ix_job(key):
    return FZ.intensity_features(DATA[key]["img"], DATA[key]["lung"])


def block_aggs(cols, keys_fit, keys_apply, seed=0, folds=5):
    """Probabilités de blocs agrégées : hors pli (par patient) pour keys_fit, modèle complet pour keys_apply."""
    pts = sorted({k[0] for k in keys_fit}); fo = {p: i % folds for i, p in enumerate(np.random.default_rng(seed).permutation(pts))}

    def fit(keys):
        X = np.vstack([PF[k]["f"][:, cols] for k in keys]); y = np.concatenate([PF[k]["y"] for k in keys])
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        sel = np.concatenate([pos, np.random.default_rng(seed + 1).choice(neg, min(len(neg), 3 * len(pos)), replace=False)])
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight="balanced", random_state=0).fit(X[sel], y[sel])
    A = {}
    for f in range(folds):
        mdl = fit([k for k in keys_fit if fo[k[0]] != f])
        for k in keys_fit:
            if fo[k[0]] == f:
                A[k] = agg(mdl.predict_proba(PF[k]["f"][:, cols])[:, 1])
    mdl = fit(keys_fit)
    for k in keys_apply:
        A[k] = agg(mdl.predict_proba(PF[k]["f"][:, cols])[:, 1])
    return A


try:
    _ixk = list(DATA)
    IXF = dict(zip(_ixk, pmap(_ix_job, _ixk)))
    trv = [k for k in PF if DATA[k]["split"] in ("train", "val") and len(PF[k]["coords"])]
    te = [k for k in PF if DATA[k]["split"] == "test" and len(PF[k]["coords"])]
    ytr = np.array([lab_of(k) for k in trv]); yte = np.array([lab_of(k) for k in te]); gte = np.array([str(k[0]) for k in te])
    atte = np.array([DATA[k]["attack"] for k in te])
    G_ = lambda ks: np.nan_to_num(np.array([PF[k]["g"] for k in ks], float))
    I_ = lambda ks: np.nan_to_num(np.array([IXF[k] for k in ks], float))
    A_ = lambda AA, ks: np.array([AA[k] for k in ks], float)
    lrm = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    hgbm = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)
    rfm = lambda: RandomForestClassifier(400, n_jobs=-1, random_state=0)
    fp = lambda Xa, Xb, mk: mk().fit(Xa, ytr).predict_proba(Xb)[:, 1]
    DS = {}
    DS["intensite_RF"] = fp(I_(trv), I_(te), rfm)
    DS["bruit_seul_RL"] = fp(G_(trv)[:, :-2], G_(te)[:, :-2], lrm)
    DS["duplication_seule_RL"] = fp(G_(trv)[:, -2:], G_(te)[:, -2:], lrm)
    DS["forensique_globale_RL"] = fp(G_(trv), G_(te), lrm)
    DS["forensique_globale_HGB"] = fp(G_(trv), G_(te), hgbm)
    Ab = block_aggs(NOISE_COLS, trv, te); Abi = block_aggs(NI_COLS, trv, te); Abc = block_aggs(ALL_COLS, trv, te)
    DS["agregation_locale_seule_complets_HGB"] = fp(A_(Abc, trv), A_(Abc, te), hgbm)
    DS["globale_plus_locale_bruit_HGB"] = fp(np.hstack([G_(trv), A_(Ab, trv)]), np.hstack([G_(te), A_(Ab, te)]), hgbm)
    DS["globale_plus_locale_bruit_intensite_HGB"] = fp(np.hstack([G_(trv), A_(Abi, trv)]), np.hstack([G_(te), A_(Abi, te)]), hgbm)
    DS["globale_plus_locale_complets_HGB"] = fp(np.hstack([G_(trv), A_(Abc, trv)]), np.hstack([G_(te), A_(Abc, te)]), hgbm)
    ref = DS["forensique_globale_RL"]; ncm = atte != "copier_deplacer"
    DEC = dict(n_test=int(len(te)), falsifiees=int(yte.sum()), patients=int(len(set(gte))),
               auc={k: auc_ci(yte, v, gte) for k, v in DS.items()},
               ecart_vs_forensique_RL={k: diff_ci(yte, v, ref, gte) for k, v in DS.items() if k != "forensique_globale_RL"},
               H1_bruit_moins_intensite=diff_ci(yte, DS["bruit_seul_RL"], DS["intensite_RF"], gte),
               H1_bruit_moins_intensite_sans_copier_deplacer=diff_ci(yte[ncm], DS["bruit_seul_RL"][ncm], DS["intensite_RF"][ncm], gte[ncm]),
               H1_forensique_moins_bruit=diff_ci(yte, ref, DS["bruit_seul_RL"], gte),
               H2_classifieur=diff_ci(yte, DS["forensique_globale_HGB"], ref, gte),
               H2_ajout_agregation=diff_ci(yte, DS["globale_plus_locale_complets_HGB"], DS["forensique_globale_HGB"], gte),
               H2_ajout_globaux=diff_ci(yte, DS["globale_plus_locale_complets_HGB"], DS["agregation_locale_seule_complets_HGB"], gte),
               H2_intensite_locale=diff_ci(yte, DS["globale_plus_locale_bruit_intensite_HGB"], DS["globale_plus_locale_bruit_HGB"], gte),
               par_attaque={k: {a: round(float(roc_auc_score(yte[(atte == a) | (atte == "authentique")], v[(atte == a) | (atte == "authentique")])), 3)
                                for a in sorted(set(atte) - {"authentique"}) if ((atte == a).sum() >= 3)} for k, v in DS.items()})
    v5out("DECOMPOSITION", DEC)
except Exception:
    v5err("D")

# ---------- E. Cascade de Zhang et al. réimplémentée (sous-images 32 x 32 -> carte -> GLCM -> ACP -> SVM) ----------
ZP, ZS = 32, 16


def z_coords(lung):
    out = []
    for y in range(0, lung.shape[0] - ZP + 1, ZS):
        for x in range(0, lung.shape[1] - ZP + 1, ZS):
            if lung[y:y + ZP, x:x + ZP].mean() > 0.5:
                out.append((y, x))
    return out


def z_labels(key, coords):
    d = DATA[key]
    if d["attack"] in GLOBAL_ATTACKS:
        return np.full(len(coords), int(d["attack"] != "authentique"))
    return np.array([d["mask"][y:y + ZP, x:x + ZP].mean() > 0.3 for y, x in coords], int)


class ZNet(nn.Module):
    def __init__(self):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


class ZSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, y, x, lab = self.items[i]
        pt = DATA[k]["img"][y:y + ZP, x:x + ZP].astype(np.float32)
        kk = random.randint(0, 3); pt = np.rot90(pt, kk)
        if random.random() < 0.5:
            pt = pt[:, ::-1]
        return torch.tensor(hu_to_net(pt.copy()))[None], torch.tensor(float(lab))


@torch.no_grad()
def z_heat(model, img, lung):
    coords = z_coords(lung); H = np.zeros(((img.shape[0] - ZP) // ZS + 1, (img.shape[1] - ZP) // ZS + 1), np.float32)
    if not coords:
        return H
    X = np.stack([img[y:y + ZP, x:x + ZP] for y, x in coords]).astype(np.float32); pr = []
    for b in range(0, len(X), 1024):
        pr.append(torch.sigmoid(model(torch.tensor(hu_to_net(X[b:b + 1024]))[:, None].to(DEVICE))).float().cpu().numpy())
    for (y, x), p in zip(coords, np.concatenate(pr)):
        H[y // ZS, x // ZS] = p
    return H


def z_glcm(H):
    from skimage.feature import graycomatrix, graycoprops
    q = np.clip(np.round(H * 15), 0, 15).astype(np.uint8)
    M = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True)
    return np.concatenate([graycoprops(M, p).ravel() for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")])


def zhang_run(split_of, seed=0, epochs=6):
    seed_all(seed)
    items = []
    for k in DATA:
        if split_of(k) == "train":
            cs = z_coords(DATA[k]["lung"]); ls = z_labels(k, cs)
            items += [(k, y, x, int(l)) for (y, x), l in zip(cs, ls)]
    pos = [i for i in items if i[3] == 1]; neg = [i for i in items if i[3] == 0]
    model = ZNet().to(DEVICE); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(epochs):
        random.shuffle(neg); cur = pos + neg[:2 * len(pos)]; model.train()
        for xb, yb in DataLoader(ZSet(cur), 512, shuffle=True, num_workers=NUM_WORKERS, drop_last=True):
            loss = F.binary_cross_entropy_with_logits(model(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    feats = lambda img, lung: np.nan_to_num(z_glcm(z_heat(model, img, lung)))
    vk = [k for k in DATA if split_of(k) == "val"]; tk = [k for k in DATA if split_of(k) == "test"]
    Xv = np.array([feats(DATA[k]["img"], DATA[k]["lung"]) for k in vk]); yv = np.array([lab_of(k) for k in vk])
    svm = make_pipeline(StandardScaler(), PCA(n_components=10, random_state=0), SVC(C=1.0, probability=True, class_weight="balanced", random_state=0)).fit(Xv, yv)
    st = {k: float(svm.predict_proba(feats(DATA[k]["img"], DATA[k]["lung"])[None])[0, 1]) for k in tk}
    se = {k: float(svm.predict_proba(feats(*ex_imgs[k])[None])[0, 1]) for k in _ek}
    sv = svm.predict_proba(Xv)[:, 1]
    return st, se, dict(patchs_pos=len(pos), patchs_neg=len(neg), loss=float(loss.item())), (vk, yv, sv)


try:
    _zt, _ze, _zinfo, _zval = zhang_run(lambda k: DATA[k]["split"], seed=CFG["SEED"])
    ZS_te = np.array([_zt[k] for k in tkeys]); ZS_ex = np.array([_ze[k] for k in _ek])
    ZH = dict(info=_zinfo, interne=auc_ci(yt, ZS_te, _g), externe=auc_ci(_ey, ZS_ex, _eg),
              interne_fusion_moins_zhang=diff_ci(yt, _s["hybride"], ZS_te, _g), interne_forensique_moins_zhang=diff_ci(yt, _s["forensique"], ZS_te, _g),
              externe_forensique_moins_zhang=diff_ci(_ey, _es["forensique"], ZS_ex, _eg), externe_fusion_moins_zhang=diff_ci(_ey, _es["hybride"], ZS_ex, _eg),
              tpr1_interne=round(float(tpr_at(yt, ZS_te)), 3),
              par_attaque={a: round(float(roc_auc_score(yt[(_atk == a) | (_atk == "authentique")], ZS_te[(_atk == a) | (_atk == "authentique")])), 3)
                           for a in sorted(set(_atk) - {"authentique"}) if (_atk == a).sum() >= 3},
              seuil_validation=thr_eval(yt, ZS_te, _zval[2][_zval[1] == 0]))
    _m = np.isin(_et, ["FM", "TM"]); ZH["ctgan_FM_vs_TM"] = round(float(roc_auc_score(_ey[_m], ZS_ex[_m])), 3)
    _m = ~np.isin(_eg, ID_TRV); ZH["externe_sans_patients_trv"] = round(float(roc_auc_score(_ey[_m], ZS_ex[_m])), 3)
    v5out("ZHANG", ZH)
except Exception:
    v5err("E"); ZS_te = ZS_ex = None

# ---------- I. TruFor (détecteur pour photographies, poids publiés) sur les mêmes coupes ----------
def to_png8(img, lo=-1350.0, hi=150.0):
    return (np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


TF_te = TF_ex = None
try:
    import subprocess, zipfile as _zf
    from PIL import Image as _Im
    TFD = Path("/kaggle/working/trufor_run"); (TFD / "in" / "te").mkdir(parents=True, exist_ok=True); (TFD / "in" / "ex").mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(tkeys):
        _Im.fromarray(to_png8(DATA[k]["img"])).convert("RGB").save(TFD / "in" / "te" / f"{i:04d}.png")
    for i, k in enumerate(_ek):
        _Im.fromarray(to_png8(ex_imgs[k][0])).convert("RGB").save(TFD / "in" / "ex" / f"{i:04d}.png")
    subprocess.run("cd /tmp && rm -rf TruFor && git clone -q --depth 1 https://github.com/grip-unina/TruFor.git && "
                   "wget -q -O /tmp/tfw.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip && "
                   "pip install -q yacs", shell=True, check=True)
    with _zf.ZipFile("/tmp/tfw.zip") as z:
        z.extractall("/tmp/TruFor/TruFor_train_test/pretrained_models")
    _w = next(iter(glob.glob("/tmp/TruFor/TruFor_train_test/pretrained_models/**/*.pth.tar", recursive=True)))
    _tp = "/tmp/TruFor/TruFor_train_test/test.py"; _src = open(_tp).read()
    open(_tp, "w").write(_src.replace("map_location=torch.device(device))", "map_location=torch.device(device), weights_only=False)"))
    for part in ("te", "ex"):
        r = subprocess.run(f'cd /tmp/TruFor/TruFor_train_test && python test.py -g {1 if torch.cuda.device_count() > 1 else 0} -in {TFD}/in/{part} -out {TFD}/out/{part} -exp trufor_ph3 TEST.MODEL_FILE "{_w}"',
                           shell=True, capture_output=True, text=True)
        print("TruFor", part, r.returncode, r.stderr[-600:])
    _sc = lambda part, n: np.array([float(np.load(TFD / "out" / part / f"{i:04d}.png.npz")["score"]) for i in range(n)])
    TF_te = _sc("te", len(tkeys)); TF_ex = _sc("ex", len(_ek))
    _m = np.isin(_et, ["FM", "TM"]); _mx = ~np.isin(_eg, ID_TRV)
    TFR = dict(interne=auc_ci(yt, TF_te, _g), externe=auc_ci(_ey, TF_ex, _eg), ctgan_FM_vs_TM=round(float(roc_auc_score(_ey[_m], TF_ex[_m])), 3),
               externe_sans_patients_trv=round(float(roc_auc_score(_ey[_mx], TF_ex[_mx])), 3),
               interne_fusion_moins_trufor=diff_ci(yt, _s["hybride"], TF_te, _g), externe_forensique_moins_trufor=diff_ci(_ey, _es["forensique"], TF_ex, _eg),
               tpr1_interne=round(float(tpr_at(yt, TF_te)), 3),
               par_attaque={a: round(float(roc_auc_score(yt[(_atk == a) | (_atk == "authentique")], TF_te[(_atk == a) | (_atk == "authentique")])), 3)
                            for a in sorted(set(_atk) - {"authentique"}) if (_atk == a).sum() >= 3},
               fenetre_png="[-1350 ; 150] HU sur 8 bits, recopiée sur 3 canaux")
    v5out("TRUFOR", TFR)
except Exception:
    v5err("I")

# ---------- J. Coupes du test interne et du test externe enregistrées (réutilisables par d'autres détecteurs) ----------
try:
    np.savez_compressed(OUT / "coupes_test.npz", te=np.stack([DATA[k]["img"] for k in tkeys]), te_mask=np.stack([DATA[k]["mask"] for k in tkeys]),
                        te_y=yt, te_attaque=_atk, te_pid=_g, ex=np.stack([ex_imgs[k][0] for k in _ek]), ex_y=_ey, ex_uuid=_eg, ex_type=_et)
    print("V5 : coupes enregistrées")
except Exception:
    v5err("J")

# ---------- F. Partitions supplémentaires (fusion à deux branches, même protocole) ----------
@torch.no_grad()
def cnn_probs(model, img, coords):
    if not coords:
        return np.zeros(0)
    X = np.stack([img[y:y + P, x:x + P] for y, x in coords]).astype(np.float32); pr = []
    for b in range(0, len(X), 256):
        pr.append(torch.sigmoid(model(torch.tensor(hu_to_net(X[b:b + 256]))[:, None].to(DEVICE))[0]).float().cpu().numpy())
    return np.concatenate(pr)


def train_cnn_seed(seed):
    seed_all(seed)
    m = ForensicUNet().to(DEVICE); opt = torch.optim.Adam(m.parameters(), CFG["LR"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tr_set, va_set = PatchSet("train", augment=True), PatchSet("val", neg_ratio=3)
    best, pat, path, ep = -1, 0, OUT / f"cnn_seed{seed}.pt", 0
    for ep in range(CFG["CNN_EPOCHS"]):
        m.train(); tr_set.resample()
        for xb, yb, mb in DataLoader(tr_set, CFG["BATCH"], shuffle=True, num_workers=NUM_WORKERS, drop_last=True):
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                logit, seg = m(xb)
                loss = F.binary_cross_entropy_with_logits(logit, yb) + dice_loss(seg.float(), mb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        m.eval(); ps_, ys_ = [], []
        with torch.no_grad():
            for xb, yb, _ in DataLoader(va_set, 256, num_workers=NUM_WORKERS):
                ps_.append(torch.sigmoid(m(xb.to(DEVICE))[0]).float().cpu().numpy()); ys_.append(yb.numpy())
        auc = roc_auc_score(np.concatenate(ys_), np.concatenate(ps_))
        if auc > best:
            best, pat = auc, 0; torch.save(m.state_dict(), path)
        else:
            pat += 1
            if pat >= 6:
                break
    m.load_state_dict(torch.load(path)); m.eval()
    return m, float(best), ep + 1


def eval_partition(SPL, fclf, model, seed, with_nod=True):
    vk = [k for k in PF if SPL[k[0]] == "val" and len(PF[k]["coords"])]; tk = [k for k in PF if SPL[k[0]] == "test" and len(PF[k]["coords"])]
    O = {}
    for k in vk + tk:
        O[k] = dict(p_for=fclf.predict_proba(PF[k]["f"])[:, 1], p_cnn=cnn_probs(model, DATA[k]["img"], PF[k]["coords"]), gfeat=PF[k]["g"])
    for k in _ek:
        c, f, gg = EXF[k]
        O[k] = dict(p_for=fclf.predict_proba(f)[:, 1], p_cnn=cnn_probs(model, ex_imgs[k][0], c), gfeat=gg)
    loc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(np.vstack([np.c_[O[k]["p_for"], O[k]["p_cnn"]] for k in vk]),
                                                                        np.concatenate([PF[k]["y"] for k in vk]))
    mf = lambda o: np.r_[o["gfeat"], agg(loc.predict_proba(np.c_[o["p_for"], o["p_cnn"]])[:, 1]), agg(o["p_for"]), agg(o["p_cnn"])]
    Xv = np.nan_to_num(np.array([mf(O[k]) for k in vk])); yv = np.array([lab_of(k) for k in vk]); gv = np.array([str(k[0]) for k in vk])
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    meta_ = mk().fit(Xv, yv)
    sc = lambda ks: dict(forensique=np.array([top3(O[k]["p_for"]) for k in ks]), cnn=np.array([top3(O[k]["p_cnn"]) for k in ks]),
                         fusion=meta_.predict_proba(np.nan_to_num(np.array([mf(O[k]) for k in ks])))[:, 1])
    St, Se = sc(tk), sc(_ek); Sv = sc(vk); Sv["fusion"] = oof_scores(Xv, yv, gv, mk)
    yt_ = np.array([lab_of(k) for k in tk]); gt_ = np.array([str(k[0]) for k in tk]); at_ = np.array([DATA[k]["attack"] for k in tk])
    R = dict(graine=seed, test=dict(n=int(len(tk)), falsifiees=int(yt_.sum()), patients=int(len(set(gt_)))),
             interne={b: auc_ci(yt_, v, gt_, n=1000) for b, v in St.items()}, externe={b: auc_ci(_ey, v, _eg, n=1000) for b, v in Se.items()},
             interne_fusion_moins_forensique=diff_ci(yt_, St["fusion"], St["forensique"], gt_, n=1000),
             externe_fusion_moins_forensique=diff_ci(_ey, Se["fusion"], Se["forensique"], _eg, n=1000),
             tpr1_courbe_test={b: round(float(tpr_at(yt_, v)), 3) for b, v in St.items()},
             seuil_validation={b: thr_eval(yt_, St[b], Sv[b][yv == 0]) for b in St})
    mi = (at_ == "gan_injection") | (at_ == "authentique")
    if mi.sum() and yt_[mi].min() != yt_[mi].max():
        R["interne_injGAN_fusion_moins_forensique"] = diff_ci(yt_[mi], St["fusion"][mi], St["forensique"][mi], gt_[mi], n=1000)
    me = np.isin(_et, ["FM", "TM"])
    R["ctgan_FM_vs_TM"] = {b: round(float(roc_auc_score(_ey[me], v[me])), 3) for b, v in Se.items()}
    R["ctgan_FM_vs_TM_fusion_moins_forensique"] = diff_ci(_ey[me], Se["fusion"][me], Se["forensique"][me], _eg[me], n=1000)
    mx = ~np.isin(_eg, ID_TRV)
    R["externe_sans_patients_trv"] = {b: round(float(roc_auc_score(_ey[mx], v[mx])), 3) for b, v in Se.items()}
    R["externe_sans_patients_trv_fusion_moins_forensique"] = diff_ci(_ey[mx], Se["fusion"][mx], Se["forensique"][mx], _eg[mx], n=1000)
    if with_nod and v5h() < V5_LIMIT_H + 1.0:
        nm = [nod_mm(k) for k in tk]; nod_ = np.array([bool(v) for v in nm]); auth_ = at_ == "authentique"
        mm = (at_ == "gan_injection") | (auth_ & nod_)
        if mm.sum() and yt_[mm].min() != yt_[mm].max():
            R["interne_injGAN_vs_auth_nodule"] = dict(pos=int(yt_[mm].sum()), neg=int((1 - yt_[mm]).sum()),
                                                      auc={b: round(float(roc_auc_score(yt_[mm], v[mm])), 3) for b, v in St.items()},
                                                      fusion_moins_forensique=diff_ci(yt_[mm], St["fusion"][mm], St["forensique"][mm], gt_[mm], n=1000))
    return R


SEEDRES = {}
try:
    _JOB = ex_imgs
    EXF = dict(zip(_ek, pmap(_cpu_job, _ek)))
    _JOB = {}
    SEEDRES[CFG["SEED"]] = eval_partition({k[0]: v["split"] for k, v in DATA.items()}, forensic_clf, cnn, CFG["SEED"])
    SEEDRES[CFG["SEED"]]["note"] = "partition principale : branches du carnet, fusion à deux branches réapprise sur la validation"
    v5out(f"PARTITION_{CFG['SEED']}", SEEDRES[CFG["SEED"]])
except Exception:
    v5err("F0")

_pts = sorted({k[0] for k in DATA})
for _seed in V5_SEEDS:
    if v5h() > V5_LIMIT_H:
        print("V5 : limite de durée atteinte, partitions restantes non exécutées"); break
    try:
        _t = time.time()
        _perm = np.random.default_rng(_seed).permutation(_pts); _n = len(_perm); _ntr = int(0.70 * _n); _nva = int(0.15 * _n)
        SPL = {p: "train" for p in _perm[:_ntr]}; SPL.update({p: "val" for p in _perm[_ntr:_ntr + _nva]}); SPL.update({p: "test" for p in _perm[_ntr + _nva:]})
        for k in DATA:
            DATA[k]["split"] = SPL[k[0]]
        _ktr = [k for k in PF if SPL[k[0]] == "train" and len(PF[k]["coords"])]
        _X = np.vstack([PF[k]["f"] for k in _ktr]); _y = np.concatenate([PF[k]["y"] for k in _ktr])
        _pos = np.where(_y == 1)[0]; _neg = np.where(_y == 0)[0]
        _sel = np.concatenate([_pos, np.random.default_rng(_seed).choice(_neg, min(len(_neg), 3 * len(_pos)), replace=False)])
        _fclf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight="balanced", random_state=0).fit(_X[_sel], _y[_sel])
        del _X, _y
        _cnn, _best, _ne = train_cnn_seed(_seed)
        SEEDRES[_seed] = eval_partition(SPL, _fclf, _cnn, _seed)
        SEEDRES[_seed].update(cnn_auc_val_patchs=round(_best, 4), cnn_epoques=_ne, duree_min=round((time.time() - _t) / 60, 1),
                              recouvrement_test_principal=int(len({p for p, v in SPL.items() if v == "test"} & set(_g))))
        v5out(f"PARTITION_{_seed}", SEEDRES[_seed])
        del _cnn; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        v5err(f"F{_seed}")
for k in DATA:
    DATA[k]["split"] = SPLIT[k[0]]
try:
    _keys = [s for s in SEEDRES if "interne" in SEEDRES[s]]
    assert _keys, "aucune partition évaluée"
    RES5 = {}
    for part in ("interne", "externe"):
        RES5[part] = {b: dict(moyenne=round(float(np.mean([SEEDRES[s][part][b][0] for s in _keys])), 3),
                              ecart_type=round(float(np.std([SEEDRES[s][part][b][0] for s in _keys], ddof=1)), 3) if len(_keys) > 1 else None,
                              valeurs=[SEEDRES[s][part][b][0] for s in _keys]) for b in ("forensique", "cnn", "fusion")}
    RES5["ecart_interne"] = [SEEDRES[s]["interne_fusion_moins_forensique"] for s in _keys]
    RES5["ecart_externe"] = [SEEDRES[s]["externe_fusion_moins_forensique"] for s in _keys]
    RES5["graines"] = _keys
    v5out("PARTITIONS_RESUME", RES5)
except Exception:
    v5err("F_resume")

# ---------- G. Scores par coupe (partition principale) pour les analyses hors carnet ----------
try:
    v5out("SCORES", dict(interne=dict(nodule_mm=[None if v is None else round(float(v), 1) for v in NODMM],
                                       zhang=None if ZS_te is None else [round(float(v), 4) for v in ZS_te],
                                       trufor=None if TF_te is None else [round(float(v), 4) for v in TF_te]),
                          externe=dict(uuid=[str(k[1]) for k in _ek], fichier=[Path(k[0]).name for k in _ek], type=_et.tolist(),
                                       zhang=None if ZS_ex is None else [round(float(v), 4) for v in ZS_ex],
                                       trufor=None if TF_ex is None else [round(float(v), 4) for v in TF_ex],
                                       scores={b: [round(float(x), 4) for x in v] for b, v in _es.items()})))
except Exception:
    v5err("G")
print(f"V5 terminé en {(time.time() - V5_T0) / 60:.1f} min (durée totale {v5h():.2f} h)")

# ---------- H. Modèles enregistrés (réutilisables pour un test externe ultérieur) ----------
try:
    import joblib
    joblib.dump(dict(forensic_clf=forensic_clf, local_fuser=local_fuser, meta=meta, loc2=_loc2, meta2=_meta2), OUT / "modeles_sklearn.joblib")
    print("V5 : modèles enregistrés dans", OUT)
except Exception:
    v5err("H")

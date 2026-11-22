"""Expériences de la preuve de concept : détection, typage, généralisation, robustesse, localisation."""
import json, time
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
import features as F

CLASSES = ["authentique", "injection", "suppression", "copier-déplacer", "bruit", "génératif"]
d = np.load("simct_tamper.npz")
X = d["X"]; Y = d["Y"]; P = d["P"]
M = np.unpackbits(d["M"], axis=-1)[:, :, :256].astype(bool)
L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)
yb = (Y > 0).astype(int)


def feat_job(i):
    t = time.perf_counter()
    f, (blocks, bf, dup) = F.image_features(X[i], L[i])
    dt = time.perf_counter() - t
    return f, F.intensity_features(X[i], L[i]), blocks, bf, dup, dt


def feat_img(img, lungs):
    return F.image_features(img, lungs)[0]


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(feat_job, range(len(X)), chunksize=10)
    FX = np.nan_to_num(np.array([r[0] for r in res]))
    IX = np.nan_to_num(np.array([r[1] for r in res]))
    times = np.array([r[5] for r in res])
    out = {"n": int(len(X)), "class_counts": np.bincount(Y).tolist(), "n_feat": int(FX.shape[1]),
           "feat_time_ms_mean": float(times.mean() * 1000), "feat_time_ms_std": float(times.std() * 1000)}

    models = {
        "Régression logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "Random Forest": lambda: RandomForestClassifier(400, n_jobs=2, random_state=0),
        "Gradient Boosting (HGB)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        "Perceptron multicouche": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=2000, alpha=1e-3, random_state=0)),
    }
    gkf = GroupKFold(5)

    def cv_eval(Xf, mk, y=yb):
        rows, prob = [], np.zeros(len(y))
        for tr, te in gkf.split(Xf, y, P):
            m = mk(); m.fit(Xf[tr], y[tr])
            p = m.predict_proba(Xf[te])[:, 1]; prob[te] = p
            pr = (p >= 0.5).astype(int)
            rows.append([accuracy_score(y[te], pr), precision_score(y[te], pr, zero_division=0),
                         recall_score(y[te], pr), f1_score(y[te], pr), roc_auc_score(y[te], p)])
        r = np.array(rows)
        return {"mean": r.mean(0).tolist(), "std": r.std(0).tolist()}, prob

    E1 = {}
    probs = {}
    E1["Ligne de base intensité (RF)"], probs["base"] = cv_eval(IX, models["Random Forest"])
    for name, mk in models.items():
        E1[name], probs[name] = cv_eval(FX, mk)
    E1["Forensique + intensité (HGB)"], _ = cv_eval(np.hstack([FX, IX]), models["Gradient Boosting (HGB)"])
    out["E1"] = E1
    best = max([k for k in models], key=lambda k: E1[k]["mean"][4])
    out["best"] = best
    pb = probs[best]
    out["per_class_detection"] = {CLASSES[c]: float(((pb >= 0.5) == (c > 0))[Y == c].mean()) for c in range(6)}
    fpr, tpr, _ = roc_curve(yb, pb); fb, tb, _ = roc_curve(yb, probs["base"])
    np.savez("roc.npz", fpr=fpr, tpr=tpr, fb=fb, tb=tb)
    out["tpr_at_fpr1"] = float(np.interp(0.01, fpr, tpr)); out["tpr_at_fpr5"] = float(np.interp(0.05, fpr, tpr))

    # Ablation par groupes de caractéristiques
    groups = {"Statistiques globales des résidus": list(range(0, 12)),
              "Autocorrélation + spectre du bruit": list(range(12, 20)),
              "Incohérence locale (blocs)": list(range(20, 38)),
              "Duplication de blocs": list(range(38, 40))}
    ab = {}
    for g, idx in groups.items():
        ab[g], _ = cv_eval(FX[:, idx], models["Gradient Boosting (HGB)"])
    ab["Toutes"] = E1["Gradient Boosting (HGB)"]
    out["ablation"] = ab

    # E2 multiclasse
    pred = np.zeros(len(Y), int)
    for tr, te in gkf.split(FX, Y, P):
        m = models["Gradient Boosting (HGB)"](); m.fit(FX[tr], Y[tr]); pred[te] = m.predict(FX[te])
    out["E2"] = {"acc": float(accuracy_score(Y, pred)), "macroF1": float(f1_score(Y, pred, average="macro")),
                 "cm": confusion_matrix(Y, pred).tolist(),
                 "f1_per_class": f1_score(Y, pred, average=None).tolist()}

    # E3 leave-one-attack-out
    E3 = {}
    for c in range(1, 6):
        aucs = []
        for tr, te in gkf.split(FX, Y, P):
            trm = tr[Y[tr] != c]
            tem = te[(Y[te] == 0) | (Y[te] == c)]
            m = models["Gradient Boosting (HGB)"](); m.fit(FX[trm], yb[trm])
            aucs.append(roc_auc_score(yb[tem], m.predict_proba(FX[tem])[:, 1]))
        E3[CLASSES[c]] = [float(np.mean(aucs)), float(np.std(aucs))]
    out["E3"] = E3

    # E4 robustesse aux post-traitements (split 70/30 par patient)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=1).split(FX, yb, P))
    m = models["Gradient Boosting (HGB)"](); m.fit(FX[tr], yb[tr])
    rng = np.random.default_rng(5)

    def pp(img, kind):
        f = img.astype(np.float32)
        if kind == "Flou gaussien σ=0,5":
            f = ndi.gaussian_filter(f, 0.5)
        elif kind == "Redimensionnement ×0,75":
            f = ndi.zoom(ndi.zoom(f, 0.75, order=1), 256 / 192, order=1)[:256, :256]
        elif kind == "Bruit additif σ=2 HU":
            f = f + rng.normal(0, 2, f.shape)
        elif kind == "Export 8 bits (fenêtre pulmonaire)":
            lo, hi = -1350, 150
            f = np.round(np.clip((f - lo) / (hi - lo), 0, 1) * 255) / 255 * (hi - lo) + lo
        return np.clip(np.round(f), -1024, 3071).astype(np.int16)

    E4 = {"Aucun": float(roc_auc_score(yb[te], m.predict_proba(FX[te])[:, 1]))}
    kinds = ["Flou gaussien σ=0,5", "Redimensionnement ×0,75", "Bruit additif σ=2 HU", "Export 8 bits (fenêtre pulmonaire)"]
    with Pool(2) as pool:
        for k in kinds:
            Fk = np.nan_to_num(np.array(pool.starmap(feat_img, [(pp(X[i], k), L[i]) for i in te])))
            E4[k] = float(roc_auc_score(yb[te], m.predict_proba(Fk)[:, 1]))
            # ré-entraînement avec augmentation (mêmes post-traitements appliqués à 50 % du train)
    out["E4"] = E4

    # E5 localisation par blocs (attaques locales)
    loc_cls = [1, 2, 3, 5]
    Bk = []
    for i, r in enumerate(res):
        blocks, bf, dup = r[2], r[3], r[4]
        if len(blocks) == 0:
            continue
        z = F.robust_z(bf)
        for (y, x), f, zz in zip(blocks, bf, z):
            lab = M[i, y:y + 16, x:x + 16].mean() > 0.3 if Y[i] in loc_cls else False
            Bk.append([i, y, x, *f, *np.abs(zz), dup[y:y + 16, x:x + 16].mean(), lab])
    Bk = np.nan_to_num(np.array(Bk, np.float32))
    bi = Bk[:, 0].astype(int)
    keep = np.isin(Y[bi], [0] + loc_cls)
    Bk, bi = Bk[keep], bi[keep]
    bX, by = Bk[:, 3:-1], Bk[:, -1].astype(int)
    bprob = np.zeros(len(by)); uns = np.abs(Bk[:, 9:15]).max(1) + 5 * Bk[:, 15]
    for trp, tep in gkf.split(bX, by, P[bi]):
        mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0, class_weight="balanced")
        mm.fit(bX[trp], by[trp]); bprob[tep] = mm.predict_proba(bX[tep])[:, 1]
    out["E5_block_auc_supervised"] = float(roc_auc_score(by, bprob))
    out["E5_block_auc_unsupervised"] = float(roc_auc_score(by, uns))

    def pixmap(i, score):
        sel = bi == i
        acc = np.zeros((256, 256)); cnt = np.zeros((256, 256))
        for (y, x), s in zip(Bk[sel, 1:3].astype(int), score[sel]):
            acc[y:y + 16, x:x + 16] += s; cnt[y:y + 16, x:x + 16] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)

    dice = {c: [] for c in loc_cls}; iou = {c: [] for c in loc_cls}
    maps = {}
    for i in np.where(np.isin(Y, loc_cls))[0]:
        pm = pixmap(i, bprob)
        pr = pm >= 0.5
        gt = M[i] & L[i]
        if gt.sum() == 0:
            gt = M[i]
        inter = (pr & gt).sum(); uni = (pr | gt).sum()
        dice[Y[i]].append(2 * inter / (pr.sum() + gt.sum() + 1e-9)); iou[Y[i]].append(inter / (uni + 1e-9))
        if Y[i] not in maps:
            maps[int(Y[i])] = (int(i), pm)
    out["E5_dice"] = {CLASSES[c]: [float(np.mean(v)), float(np.std(v))] for c, v in dice.items()}
    out["E5_iou"] = {CLASSES[c]: [float(np.mean(v)), float(np.std(v))] for c, v in iou.items()}
    np.savez("maps.npz", **{f"m{c}": v[1] for c, v in maps.items()}, idx=np.array([maps[c][0] for c in sorted(maps)]),
             cls=np.array(sorted(maps)))
    # importance des caractéristiques (permutation simplifiée via HGB n'expose pas) -> RF
    rf = RandomForestClassifier(400, n_jobs=2, random_state=0).fit(FX, yb)
    out["rf_importance"] = rf.feature_importances_.tolist()
    np.save("probs_best.npy", pb)
    json.dump(out, open("results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rf_importance",)}, ensure_ascii=False, indent=1))

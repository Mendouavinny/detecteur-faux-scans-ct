"""Approche proposée (PoC) : détection hiérarchique local -> global.
Étape 1 : classifieur de blocs (probabilité de falsification locale).
Étape 2 : agrégation (max, moyenne top-k) + descripteurs globaux -> décision image.
Validation croisée imbriquée groupée par patient (aucune fuite entre patients)."""
import json, time
import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
import features as F

CLASSES = ["authentique", "injection", "suppression", "copier-déplacer", "bruit", "génératif"]
d = np.load("simct_tamper.npz")
X, Y, P = d["X"], d["Y"], d["P"]
M = np.unpackbits(d["M"], axis=-1)[:, :, :256].astype(bool)
L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)
yb = (Y > 0).astype(int)


def job(i):
    f, (blocks, bf, dup) = F.image_features(X[i], L[i])
    rows = []
    if len(blocks):
        z = np.abs(F.robust_z(bf))
        img = X[i].astype(np.float32)
        for (y, x), b, zz in zip(blocks, bf, z):
            lab = (M[i, y:y + 16, x:x + 16].mean() > 0.3) and Y[i] in (1, 2, 3, 5)
            blk = img[y:y + 16, x:x + 16]
            rows.append([i, y, x, *b, *zz, dup[y:y + 16, x:x + 16].mean(), blk.mean(), blk.std(), lab])
    return f, np.array(rows, np.float32)


def hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)


def agg(bi, prob, n):
    out = np.zeros((n, 4))
    for i in range(n):
        p = np.sort(prob[bi == i])[::-1]
        if len(p) == 0:
            continue
        out[i] = [p[0], p[:3].mean(), p[:10].mean(), (p > 0.5).sum()]
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(job, range(len(X)), chunksize=10)
    FX = np.nan_to_num(np.array([r[0] for r in res]))
    BK = np.nan_to_num(np.vstack([r[1] for r in res if len(r[1])]))
    np.savez("cache.npz", FX=FX, BK=BK)
    bi, bX, by = BK[:, 0].astype(int), BK[:, 3:-1], BK[:, -1].astype(int)
    pid = np.unique(P)
    fold_of = {p: k % 5 for k, p in enumerate(np.random.default_rng(0).permutation(pid))}
    fold = np.array([fold_of[p] for p in P])
    n = len(X)
    prob_img = np.zeros(n); rows = []; block_prob_test = np.zeros(len(by))
    for k in range(5):
        te_i = np.where(fold == k)[0]; tr_i = np.where(fold != k)[0]
        te_b = np.isin(bi, te_i); tr_b = np.isin(bi, tr_i)
        # méta-caractéristiques OOF internes pour le train
        inner = np.zeros(len(by))
        for kk in [j for j in range(5) if j != k]:
            itr = tr_b & ~np.isin(bi, np.where(fold == kk)[0]); ite = tr_b & np.isin(bi, np.where(fold == kk)[0])
            m = hgb().fit(bX[itr], by[itr]); inner[ite] = m.predict_proba(bX[ite])[:, 1]
        mb = hgb().fit(bX[tr_b], by[tr_b])
        block_prob_test[te_b] = mb.predict_proba(bX[te_b])[:, 1]
        A_tr = agg(bi, inner, n)[tr_i]; A_te = agg(bi, block_prob_test, n)[te_i]
        mi = hgb().fit(np.hstack([FX[tr_i], A_tr]), yb[tr_i])
        p = mi.predict_proba(np.hstack([FX[te_i], A_te]))[:, 1]; prob_img[te_i] = p
        pr = (p >= 0.5).astype(int)
        rows.append([accuracy_score(yb[te_i], pr), precision_score(yb[te_i], pr), recall_score(yb[te_i], pr),
                     f1_score(yb[te_i], pr), roc_auc_score(yb[te_i], p)])
    r = np.array(rows)
    out = {"hybrid": {"mean": r.mean(0).tolist(), "std": r.std(0).tolist()}}
    out["hybrid_per_class"] = {CLASSES[c]: float(((prob_img >= 0.5) == (c > 0))[Y == c].mean()) for c in range(6)}
    fpr, tpr, _ = roc_curve(yb, prob_img)
    out["tpr_at_fpr1"] = float(np.interp(0.01, fpr, tpr)); out["tpr_at_fpr5"] = float(np.interp(0.05, fpr, tpr))
    out["block_auc_nested"] = float(roc_auc_score(by, block_prob_test))
    # localisation à partir des probabilités de blocs (test externe)
    dice, iou, maps = {}, {}, {}
    for c in (1, 2, 3, 5):
        dl, il = [], []
        for i in np.where(Y == c)[0]:
            sel = bi == i
            acc = np.zeros((256, 256)); cnt = np.zeros((256, 256))
            for (y, x), s in zip(BK[sel, 1:3].astype(int), block_prob_test[sel]):
                acc[y:y + 16, x:x + 16] += s; cnt[y:y + 16, x:x + 16] += 1
            pm = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
            pr = pm >= 0.5; gt = M[i] & L[i]
            inter = (pr & gt).sum()
            dl.append(2 * inter / (pr.sum() + gt.sum() + 1e-9)); il.append(inter / ((pr | gt).sum() + 1e-9))
            if c not in maps or dl[-1] > 0.5 and maps[c][2] < 0.5:
                maps[c] = (i, pm, dl[-1])
        dice[CLASSES[c]] = [float(np.mean(dl)), float(np.std(dl)), float(np.median(dl))]
        iou[CLASSES[c]] = [float(np.mean(il)), float(np.std(il))]
    out["dice"] = dice; out["iou"] = iou
    # comptage détection locale : au moins un bloc correct > 0.5
    out["time_s"] = time.time() - t0
    fprs, tprs, _ = roc_curve(yb, prob_img)
    np.savez("roc_hybrid.npz", fpr=fprs, tpr=tprs)
    np.savez("maps_hybrid.npz", idx=np.array([maps[c][0] for c in (1, 2, 3, 5)]),
             **{f"m{c}": maps[c][1] for c in (1, 2, 3, 5)})
    np.save("prob_hybrid.npy", prob_img)
    json.dump(out, open("results_hybrid.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

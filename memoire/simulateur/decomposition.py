"""Décomposition des hypothèses H1 et H2 sur la preuve de concept (SimCT-Tamper).
H1 : information portée par le bruit vs par la duplication exacte.
H2 : effet du changement de classifieur, de l'agrégation locale et des descripteurs globaux.
Mêmes plis par patient que hybrid.py (permutation des 36 patients, graine 0) ; probabilités locales imbriquées.
Toutes les AUC sont calculées sur les prédictions hors pli regroupées ; IC à 95 % par rééchantillonnage apparié des patients."""
import json
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import features as F

d = np.load("simct_tamper.npz")
X, Y, P = d["X"], d["Y"], d["P"]
L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)
yb = (Y > 0).astype(int)
c = np.load("cache.npz")
FX, BK = c["FX"], c["BK"]
CLASSES = ["authentique", "injection", "suppression", "copier-déplacer", "bruit", "génératif"]
pid = np.unique(P)
fold_of = {p: k % 5 for k, p in enumerate(np.random.default_rng(0).permutation(pid))}
fold = np.array([fold_of[p] for p in P])
n = len(X)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def hgb():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=0)


def rf():
    return RandomForestClassifier(400, n_jobs=2, random_state=0)


def oof(Xf, mk):
    p = np.zeros(n); fa = []
    for k in range(5):
        te = fold == k; tr = ~te
        m = mk().fit(Xf[tr], yb[tr]); p[te] = m.predict_proba(Xf[te])[:, 1]
        fa.append(roc_auc_score(yb[te], p[te]))
    return p, float(np.mean(fa))


def agg(bi, prob):
    out = np.zeros((n, 4))
    for i in range(n):
        q = np.sort(prob[bi == i])[::-1]
        if len(q):
            out[i] = [q[0], q[:3].mean(), q[:10].mean(), (q > 0.5).sum()]
    return out


def local_probs(cols):
    """Probabilités de blocs : imbriquées pour l'entraînement de chaque pli, externes pour son test."""
    bi, by = BK[:, 0].astype(int), BK[:, -1].astype(int)
    bX = BK[:, 3:-1][:, cols]
    A_tr_all, A_te = {}, np.zeros((n, 4))
    btest = np.zeros(len(by))
    for k in range(5):
        te_i = np.where(fold == k)[0]; tr_i = np.where(fold != k)[0]
        te_b = np.isin(bi, te_i); tr_b = np.isin(bi, tr_i)
        inner = np.zeros(len(by))
        for kk in [j for j in range(5) if j != k]:
            ite = tr_b & np.isin(bi, np.where(fold == kk)[0]); itr = tr_b & ~ite
            inner[ite] = hgb().fit(bX[itr], by[itr]).predict_proba(bX[ite])[:, 1]
        btest[te_b] = hgb().fit(bX[tr_b], by[tr_b]).predict_proba(bX[te_b])[:, 1]
        A_tr_all[k] = agg(bi, inner)
    A_te = agg(bi, btest)
    return A_tr_all, A_te


def stacked(Ffeat, A_tr_all, A_te, mk):
    p = np.zeros(n); fa = []
    for k in range(5):
        te = fold == k; tr = ~te
        Xtr = A_tr_all[k][tr] if Ffeat is None else np.hstack([Ffeat[tr], A_tr_all[k][tr]])
        Xte = A_te[te] if Ffeat is None else np.hstack([Ffeat[te], A_te[te]])
        m = mk().fit(Xtr, yb[tr]); p[te] = m.predict_proba(Xte)[:, 1]
        fa.append(roc_auc_score(yb[te], p[te]))
    return p, float(np.mean(fa))


def boot(a, b=None, mask=None, nb=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = {u: np.where(P == u)[0] for u in pid}
    yy = yb if mask is None else yb
    D = []
    for _ in range(nb):
        ii = np.concatenate([idx[u] for u in rng.choice(pid, len(pid))])
        if mask is not None:
            ii = ii[mask[ii]]
        if yy[ii].min() == yy[ii].max():
            continue
        D.append(roc_auc_score(yy[ii], a[ii]) - (roc_auc_score(yy[ii], b[ii]) if b is not None else 0))
    m = np.ones(n, bool) if mask is None else mask
    base = roc_auc_score(yb[m], a[m]) - (roc_auc_score(yb[m], b[m]) if b is not None else 0)
    return [round(float(base), 3), round(float(np.percentile(D, 2.5)), 3), round(float(np.percentile(D, 97.5)), 3)]


def per_family(p):
    return {CLASSES[k]: round(float(roc_auc_score(yb[(Y == 0) | (Y == k)], p[(Y == 0) | (Y == k)])), 3) for k in range(1, 6)}


if __name__ == "__main__":
    with Pool(2) as pool:
        IX = np.nan_to_num(np.array(pool.starmap(F.intensity_features, [(X[i], L[i]) for i in range(n)])))
    out = {"plis": "5 plis par patient, permutation graine 0 (identiques à hybrid.py)"}
    S = {}
    # ---- H1 : bruit vs duplication ----
    S["intensite_RF"] = oof(IX, rf)
    S["forensique_LR"] = oof(FX, lr)
    S["bruit_seul_LR"] = oof(FX[:, :38], lr)
    S["bruit_seul_HGB"] = oof(FX[:, :38], hgb)
    S["duplication_seule_LR"] = oof(FX[:, 38:40], lr)
    # ---- H2 : décomposition ----
    S["global_HGB"] = oof(FX, hgb)
    A_tr, A_te = local_probs(list(range(15)))           # blocs : 6 bruit + 6 z + duplication + moyenne + écart-type
    S["local_seul_HGB"] = stacked(None, A_tr, A_te, hgb)
    S["local_seul_LR"] = stacked(None, A_tr, A_te, lr)
    S["global_plus_local_HGB"] = stacked(FX, A_tr, A_te, hgb)   # approche hiérarchique de la preuve de concept
    S["global_plus_local_LR"] = stacked(FX, A_tr, A_te, lr)
    An_tr, An_te = local_probs(list(range(12)))         # blocs : descripteurs de bruit seuls
    S["local_bruit_seul_HGB"] = stacked(None, An_tr, An_te, hgb)
    S["global_bruit_plus_local_bruit_HGB"] = stacked(FX[:, :38], An_tr, An_te, hgb)
    out["auc"] = {k: dict(auc_regroupee=round(float(roc_auc_score(yb, v[0])), 3), moyenne_plis=round(v[1], 3)) for k, v in S.items()}
    out["ic_auc"] = {k: boot(v[0]) for k, v in S.items()}
    out["par_famille"] = {k: per_family(S[k][0]) for k in ("intensite_RF", "forensique_LR", "bruit_seul_LR", "duplication_seule_LR",
                                                            "global_HGB", "local_seul_HGB", "global_plus_local_HGB", "local_bruit_seul_HGB")}
    p = lambda k: S[k][0]
    out["differences"] = {
        "H1 bruit seul (LR) - intensite (RF)": boot(p("bruit_seul_LR"), p("intensite_RF")),
        "H1 forensique complet (LR) - bruit seul (LR)": boot(p("forensique_LR"), p("bruit_seul_LR")),
        "H1 bruit seul (LR) - intensite (RF), sans copier-deplacer": boot(p("bruit_seul_LR"), p("intensite_RF"), mask=(Y != 3)),
        "H2 changement de classifieur : global HGB - global LR": boot(p("global_HGB"), p("forensique_LR")),
        "H2 ajout de l'agregation locale : global+local HGB - global HGB": boot(p("global_plus_local_HGB"), p("global_HGB")),
        "H2 ajout des descripteurs globaux : global+local HGB - local seul HGB": boot(p("global_plus_local_HGB"), p("local_seul_HGB")),
        "H2 local seul HGB - global LR": boot(p("local_seul_HGB"), p("forensique_LR")),
        "H2 total : global+local HGB - global LR": boot(p("global_plus_local_HGB"), p("forensique_LR")),
        "H2 bruit seul : local bruit HGB - global bruit LR": boot(p("local_bruit_seul_HGB"), p("bruit_seul_LR")),
    }
    json.dump(out, open("results_decomposition.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

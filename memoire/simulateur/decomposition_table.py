"""Tableau 3.8 : décomposition H1/H2 (preuve de concept), AUC hors pli regroupées et IC appariés par patient."""
import json, numpy as np
from multiprocessing import Pool
import decomposition as D
import features as F
from sklearn.metrics import roc_auc_score
with Pool(2) as pool:
    IX = np.nan_to_num(np.array(pool.starmap(F.intensity_features, [(D.X[i], D.L[i]) for i in range(D.n)])))
S = {}
S["Intensité seule (RF)"] = D.oof(IX, D.rf)
S["Bruit seul, sans duplication (RL)"] = D.oof(D.FX[:, :38], D.lr)
S["Duplication seule (RL)"] = D.oof(D.FX[:, 38:40], D.lr)
S["Forensique globale (RL)"] = D.oof(D.FX, D.lr)
S["Forensique globale (HGB)"] = D.oof(D.FX, D.hgb)
blocs = {"bruit": list(range(12)), "bruit + intensité": list(range(12)) + [13, 14], "complets": list(range(15))}
for name, cols in blocs.items():
    A_tr, A_te = D.local_probs(cols)
    if name == "complets":
        S["Agrégation locale seule, blocs complets (HGB)"] = D.stacked(None, A_tr, A_te, D.hgb)
    S[f"Globale + agrégation locale, blocs {name} (HGB)"] = D.stacked(D.FX, A_tr, A_te, D.hgb)
ref = S["Forensique globale (RL)"][0]
rows = []
for k, (p, fm) in S.items():
    rows.append(dict(config=k, auc=D.boot(p), moyenne_plis=round(fm, 3), ecart_vs_RL=None if k == "Forensique globale (RL)" else D.boot(p, ref),
                     par_famille=D.per_family(p)))
json.dump(rows, open("results_tableau_3_8.json", "w"), ensure_ascii=False, indent=1)
for r in rows: print(r["config"], r["auc"], r["moyenne_plis"], r["ecart_vs_RL"])

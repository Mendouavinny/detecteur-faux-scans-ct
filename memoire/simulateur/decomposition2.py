"""Compléments : quelles informations de bloc portent le gain de l'agrégation locale ?"""
import json, numpy as np
import decomposition as D
from sklearn.metrics import roc_auc_score
S = {}
S["forensique_LR"] = D.oof(D.FX, D.lr)
variants = {"bruit": list(range(12)), "bruit+duplication": list(range(13)),
            "bruit+intensite": list(range(12)) + [13, 14], "complet": list(range(15)), "intensite_seule": [13, 14]}
out = {}
for name, cols in variants.items():
    A_tr, A_te = D.local_probs(cols)
    S["local_" + name] = D.stacked(None, A_tr, A_te, D.hgb)
    S["global+local_" + name] = D.stacked(D.FX, A_tr, A_te, D.hgb)
out["auc"] = {k: round(float(roc_auc_score(D.yb, v[0])), 3) for k, v in S.items()}
out["diff_vs_global_LR"] = {k: D.boot(v[0], S["forensique_LR"][0]) for k, v in S.items() if k != "forensique_LR"}
out["par_famille"] = {k: D.per_family(v[0]) for k, v in S.items()}
json.dump(out, open("results_decomposition2.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

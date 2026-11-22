"""Exclusion des scanners CT-GAN identifiés pixel à pixel (notebook99d3b07677 v2) : recalcul sur les scores v4."""
import json, numpy as np
exec(open('tableau_3_9.py').read().split("rows = []")[0])
from sklearn.metrics import roc_auc_score
ID = {  # uuid CT-GAN -> (patient LIDC, partition, fraction de pixels identiques sur la coupe médiane / étiquetée)
 "1009": ("0182", "train", .9852), "1201": ("0401", "train", .9908), "1632": ("0065", "test", 1.0), "1871": ("0068", "val", 1.0),
 "1937": ("0964", "train", .9898), "2024": ("0517", "val", 1.0), "2031": ("0531", "train", .9922), "2104": ("0062", "test", 1.0),
 "2220": ("0844", "train", 1.0), "2366": ("0201", "train", 1.0), "2484": ("0545", "train", 1.0), "2494": ("0144", "test", .9704),
 "2508": ("0048", "val", 1.0), "2590": ("0502", "train", .9922), "2703": ("0244", "train", 1.0), "2819": ("0967", "test", .9878),
 "2997": ("0641", "test", .9414), "3246": ("0446", "train", .9909), "3304": ("0019", "test", .9888), "3378": ("0228", "val", 1.0),
 "4632": ("0060", "train", .9854), "5614": ("0725", "train", .9916), "7668": ("0754", "train", 1.0), "7794": ("0051", "train", .9705),
 "8972": ("0833", "train", 1.0)}
trv = {u for u, v in ID.items() if v[1] in ("train", "val")}; tst = {u for u, v in ID.items() if v[1] == "test"}
present = set(ge)
out = dict(identifies=len(ID), entrainement=sum(v[1] == "train" for v in ID.values()), validation=sum(v[1] == "val" for v in ID.values()),
           test=len(tst), identifies_presents_dans_les_147=len(set(ID) & present),
           trv_presents=len(trv & present), test_presents=len(tst & present),
           coupes_trv=int(np.isin(ge, list(trv)).sum()), coupes_test=int(np.isin(ge, list(tst)).sum()))
rows = []
auth = np.isin(T, ['m', 'b'])
for name, m in {"CT-GAN sans les scanners de patients d'entraînement ou de validation": ~np.isin(ge, list(trv)),
                "CT-GAN, injections, sans ces scanners": ~np.isin(ge, list(trv)) & (auth | (T == 'I')),
                "CT-GAN, uniquement les scanners de patients d'entraînement ou de validation": np.isin(ge, list(trv))}.items():
    y = ye[m]
    if y.min() == y.max():
        continue
    rows.append(dict(test=name, pos=int(y.sum()), neg=int((1 - y).sum()), groupes=int(len(set(ge[m]))),
                     auc_for=round(float(roc_auc_score(y, Re['forensique'][m])), 3), auc_fus=round(float(roc_auc_score(y, Re['hybride'][m])), 3),
                     auc_cnn=round(float(roc_auc_score(y, Re['cnn'][m])), 3), ecart=boot(y, Re['hybride'][m], Re['forensique'][m], ge[m])))
out["lignes"] = rows
json.dump(out, open('identite_ctgan.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

"""Contrôle du détecteur Python v7 sur les 10 coupes d'exemple, contre les sorties du carnet Kaggle.

Usage : python comparer_moteurs.py [--diffusion]
Compare, coupe par coupe, les probabilités des patchs (bruit, CNN), les 40 descripteurs de la coupe et les scores
des méta-classifieurs à site/modele/v7/reference.json, calculé dans le carnet par outils_site/export_kaggle_v7.py.
Avec --diffusion, la branche diffusion est calculée aussi (même graine que le carnet) : le score « trois branches »
est alors comparé.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pydicom

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import detecteur as D  # noqa: E402

avec_diffusion = "--diffusion" in sys.argv
det = D.DetecteurV7()
ref = json.load(open(D.fichier_v7("reference.json"), encoding="utf-8"))
pire = dict(p_for=0.0, p_cnn=0.0, g=0.0, bruit_cnn=0.0, trois=0.0, s=0.0)
t0 = time.time()
for nom, r in sorted(ref.items()):
    res = det.analyser(D.dicom_to_hu(pydicom.dcmread(ICI / "exemples" / nom)), avec_diffusion=avec_diffusion)
    if res["n_patchs"] != r["n"]:
        print(f"{nom} : {res['n_patchs']} patchs au lieu de {r['n']}")
        pire["p_for"] = np.inf
        continue
    e = dict(p_for=np.abs(res["p_for"] - r["p_for"]).max(), p_cnn=np.abs(res["p_cnn"] - r["p_cnn"]).max(),
             g=(np.abs(res["gfeat"] - r["g"]) / np.maximum(1e-6, np.abs(r["g"]))).max(),
             bruit_cnn=abs(res["score_bruit_cnn"] - r["meta2"]))
    if avec_diffusion:
        e["trois"] = abs(res["score_trois_branches"] - r["meta"])
        e["s"] = np.abs(res["s_diff"] - r["sz"]).max()
    for k, v in e.items():
        pire[k] = max(pire[k], float(v))
    print(f"{nom:30s} patchs {r['n']:3d}  bruit+CNN {res['score_bruit_cnn']:.4f} (carnet {r['meta2']:.4f})"
          + (f"  trois branches {res['score_trois_branches']:.4f} (carnet {r['meta']:.4f})" if avec_diffusion else ""))
print("Écarts maximaux : " + ", ".join(f"{k} {v:.1e}" for k, v in pire.items() if avec_diffusion or k not in ("trois", "s"))
      + f" ; {(time.time() - t0) / len(ref):.1f} s par coupe")

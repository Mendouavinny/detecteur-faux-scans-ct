"""Compare les scores de l'application sur les coupes d'exemple : moteur courant contre index.json (et entre moteurs).

Usage : python comparer_moteurs.py [dossier_modeles]   (DETECTEUR_MOTEUR=torch|onnx pour choisir le moteur)
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

det = D.Detecteur(Path(sys.argv[1])) if len(sys.argv) > 1 else D.Detecteur()
idx = json.load(open(ICI / "exemples" / "index.json", encoding="utf-8"))
ecarts, res = [], {}
t0 = time.time()
for e in idx:
    r = det.analyser(D.dicom_to_hu(pydicom.dcmread(ICI / "exemples" / e["fichier"])))
    res[e["fichier"]] = dict(score=r["score"], hgb=r["score_hgb"])
    ecarts.append(abs(r["score"] - e["score_authentique"]))
    print(f"{e['fichier']:30s} index {e['score_authentique']:.3f}  {det.moteur} {r['score']:.3f}  HGB {r['score_hgb']:.3f}")
print(f"moteur {det.moteur} : écart moyen {np.mean(ecarts):.4f}, max {np.max(ecarts):.4f} ; "
      f"{(time.time() - t0) / len(idx):.1f} s par coupe")
json.dump(res, open(ICI / f"scores_{det.moteur}.json", "w"), indent=1)

"""Vérifie que l'application reproduit le carnet : scores des coupes AUTHENTIQUES du test interne.

Les coupes authentiques n'ont pas été modifiées par le carnet : on peut les relire depuis LIDC-IDRI et comparer
les scores de l'application à ceux enregistrés par le carnet (scores_test_lidc.npz, coupes authentiques).
Seule différence attendue : les tirages aléatoires de la branche diffusion (graine fixe ici, libre dans le carnet).
Usage : python verifier_fidelite.py
"""
import configparser
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser
import pylidc as pl  # noqa: E402
import detecteur as D  # noqa: E402

G = Path.home() / "Documents" / "GPU_ROCm"
man = pd.read_csv(G / "out" / "manifest.csv")
test = man[(man.split == "test") & (man.attack == "authentique")]
S = np.load(G / "out" / "scores_test_lidc.npz")
ref = np.sort(S["hybride"][S["y"] == 0])

det = D.Detecteur()
scores, temps = [], []
for pid, grp in test.groupby("pid"):
    vol = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first().to_volume(verbose=False).astype(np.int16)
    for z in grp.z:
        t = time.time()
        r = det.analyser(vol[:, :, int(z)])
        temps.append(time.time() - t)
        if r["ok"]:
            scores.append(r["score"])
app = np.sort(scores)
print(f"coupes analysées : {len(app)} (carnet : {len(ref)}) ; temps moyen {np.mean(temps):.2f} s par coupe")
if len(app) == len(ref):
    print(f"écart absolu des scores triés : moyen {np.abs(app - ref).mean():.4f}, max {np.abs(app - ref).max():.4f}")
print("quantiles application :", np.quantile(app, [0.1, 0.5, 0.9]).round(3))
print("quantiles carnet      :", np.quantile(ref, [0.1, 0.5, 0.9]).round(3))
json.dump(dict(n_app=len(app), n_carnet=len(ref), app=app.tolist(), carnet=ref.tolist(),
               temps_moyen_s=float(np.mean(temps))),
          open(Path(__file__).with_name("verification_fidelite.json"), "w"), indent=1)

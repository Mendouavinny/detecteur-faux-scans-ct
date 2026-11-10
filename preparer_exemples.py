"""Copie quelques coupes DICOM authentiques de patients du TEST (jamais vus à l'entraînement) dans exemples/.

Les fichiers sont copiés tels quels depuis LIDC-IDRI (aucune modification) ; exemples/index.json donne, pour
chaque coupe, le patient, l'indice de coupe et les nodules annotés par au moins trois radiologues (ligne, colonne,
rayon en pixels), utiles aux attaques de suppression de la démonstration.
Usage : python preparer_exemples.py
"""
import configparser
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser
import pylidc as pl  # noqa: E402

ICI = Path(__file__).resolve().parent
G = Path.home() / "Documents" / "GPU_ROCm"
EX = ICI / "exemples"
EX.mkdir(exist_ok=True)
man = pd.read_csv(G / "out" / "manifest.csv")
test = man[(man.split == "test") & (man.attack == "authentique")]
index = []
for pid in sorted(test.pid.unique())[:8]:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    imgs = scan.load_all_dicom_images(verbose=False)            # triées comme to_volume (axe z)
    nods = []
    for cl in scan.cluster_annotations():
        if len(cl) >= 3:
            c = np.mean([a.centroid for a in cl], axis=0)
            d = float(np.mean([a.diameter for a in cl]))
            nods.append((int(round(c[0])), int(round(c[1])), int(round(c[2])), d / scan.pixel_spacing / 2))
    zs = test[test.pid == pid].z.tolist()
    avec = [z for z in zs if any(abs(n[2] - z) <= 1 for n in nods)]
    choix = (avec[:1] + [z for z in zs if z not in avec][:1]) or zs[:1]
    for z in choix:
        src = Path(imgs[z].filename)
        nom = f"{pid}_coupe{z:03d}.dcm"
        shutil.copy2(src, EX / nom)
        nz = [[r, c, rad] for (r, c, zz, rad) in nods if abs(zz - z) <= 1 and 40 < r < 472 and 40 < c < 472]
        index.append(dict(fichier=nom, patient=pid, coupe=int(z), nodules=nz))
        print(nom, "nodules :", nz)
# score du détecteur sur chaque coupe authentique (affiché dans l'application, par transparence) ;
# les coupes sans patch pulmonaire exploitable sont retirées
import pydicom  # noqa: E402
import detecteur as D  # noqa: E402
det = D.Detecteur()
garde = []
for e in index:
    r = det.analyser(D.dicom_to_hu(pydicom.dcmread(EX / e["fichier"])))
    if r["ok"]:
        e["score_authentique"] = round(r["score"], 3)
        garde.append(e)
    else:
        (EX / e["fichier"]).unlink()
        print("retirée (non analysable) :", e["fichier"])
json.dump(garde, open(EX / "index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(garde), "coupes ->", EX)

"""Télécharge les modèles de l'application depuis le dépôt de modèles Hugging Face (hébergement en ligne).

Usage : python telecharger_modeles.py [dépôt] ; dépôt par défaut : MENDOUA/detecteur-faux-scans-ct-modeles
Les fichiers déjà présents et complets sont conservés.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

DEPOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DETECTEUR_DEPOT_HF", "MENDOUA/detecteur-faux-scans-ct-modeles")
FICHIERS = ["cnn.onnx", "ddpm.onnx", "gan.onnx", "modeles_fusion.joblib", "selection_fusion.json", "resultats_lidc.json"]
DST = Path(__file__).resolve().parent / "modeles"
DST.mkdir(exist_ok=True)

for nom in FICHIERS:
    url = f"https://huggingface.co/{DEPOT}/resolve/main/{nom}"
    cible = DST / nom
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        taille = int(r.headers.get("Content-Length") or r.headers.get("X-Linked-Size") or 0)
    if cible.exists() and taille and cible.stat().st_size == taille:
        print("déjà présent :", nom)
        continue
    for essai in range(5):
        try:
            urllib.request.urlretrieve(url, cible)
            break
        except Exception as e:
            print(f"  nouvel essai {nom} : {e}")
            time.sleep(5)
    else:
        raise SystemExit(f"échec du téléchargement : {nom}")
    print(f"téléchargé : {nom} ({cible.stat().st_size / 1e6:.1f} Mo)")

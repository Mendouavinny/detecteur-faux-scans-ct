"""Télécharge les modèles de l'application depuis le dépôt de modèles Hugging Face (hébergement en ligne).

Usage : python telecharger_modeles.py [dépôt] ; dépôt par défaut : MENDOUA/detecteur-faux-scans-ct-modeles
Les réseaux ONNX y sont publiés en morceaux de 4 Mo (morceaux.json) : ils sont rassemblés ici et leur empreinte
SHA-256 est vérifiée. Les fichiers déjà présents et intègres sont conservés.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DEPOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DETECTEUR_DEPOT_HF", "MENDOUA/detecteur-faux-scans-ct-modeles")
SIMPLES = ["modeles_fusion.joblib", "selection_fusion.json", "resultats_lidc.json"]
DST = Path(__file__).resolve().parent / "modeles"
DST.mkdir(exist_ok=True)


def lire(chemin):
    url = f"https://huggingface.co/{DEPOT}/resolve/main/{chemin}"
    for essai in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print(f"  nouvel essai {chemin} : {e}", flush=True)
            time.sleep(5)
    raise SystemExit(f"échec du téléchargement : {chemin}")


for nom in SIMPLES:
    if not (DST / nom).exists():
        (DST / nom).write_bytes(lire(nom))
    print("prêt :", nom, flush=True)

manifeste = json.loads(lire("morceaux.json"))
for nom, m in manifeste.items():
    cible = DST / nom
    if cible.exists() and hashlib.sha256(cible.read_bytes()).hexdigest() == m["sha256"]:
        print("déjà présent :", nom)
        continue
    data = b"".join(lire(p) for p in m["morceaux"])
    if hashlib.sha256(data).hexdigest() != m["sha256"]:
        raise SystemExit(f"empreinte SHA-256 incorrecte : {nom}")
    cible.write_bytes(data)
    print(f"rassemblé et vérifié : {nom} ({len(data) / 1e6:.1f} Mo)", flush=True)

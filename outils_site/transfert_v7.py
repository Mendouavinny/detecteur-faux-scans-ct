"""Transfert des fichiers export_web_v7 depuis une session Kaggle vers l'onglet parent.

À lancer dans le carnet Kaggle (session interactive) avec :
    %run /tmp/d/outils_site/transfert_v7.py

Le serveur Jupyter de Kaggle n'expose pas /files ni /api/contents : les fichiers
passent donc par les sorties de la cellule, en morceaux base64. Chaque morceau est
une petite sortie HTML dont le script le transmet à la page Kaggle qui héberge le
carnet (window.top.postMessage, origine www.kaggle.com). Une seule sortie est
gardée à la fois (clear_output), le carnet ne grossit pas. Le débit reste sous la
limite IOPub de Jupyter (1 Mo/s). Rien n'est envoyé ailleurs.
"""
import base64
import hashlib
import json
import os
import time

from IPython.display import HTML, clear_output, display

D = "/kaggle/working/export_web_v7"
MORCEAU = 600_000  # caractères base64 par sortie
PAUSE = 0.85  # secondes entre deux sorties

noms = sorted(n for n in os.listdir(D) if os.path.isfile(os.path.join(D, n)))
plan = []
for n in noms:
    with open(os.path.join(D, n), "rb") as f:
        b = f.read()
    s = base64.b64encode(b).decode("ascii")
    k = (len(s) + MORCEAU - 1) // MORCEAU
    plan.append((n, s, k, hashlib.sha256(b).hexdigest(), len(b)))

total = sum(p[2] for p in plan)
j = 0
for n, s, k, sha, taille in plan:
    for i in range(k):
        j += 1
        meta = json.dumps({"type": "morceau-v7", "nom": n, "i": i, "k": k, "sha": sha, "taille": taille})
        script = (
            "<script>(function(){var m=" + meta + ";m.data='" + s[i * MORCEAU:(i + 1) * MORCEAU] + "';"
            "window.top.postMessage(m,'https://www.kaggle.com');})();</script>"
        )
        clear_output(wait=True)
        display(HTML("<pre>transfert " + str(j) + "/" + str(total) + " : " + n + " " + str(i + 1) + "/" + str(k) + "</pre>" + script))
        time.sleep(PAUSE)

clear_output(wait=True)
display(HTML("<pre>transfert termine : " + str(total) + " morceaux, " + str(len(plan)) + " fichiers</pre>"))

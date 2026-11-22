# Modèles de la version 7 (exécution finale, 300 patients)

Fichiers exportés par `outils_site/export_kaggle_v7.py` depuis le carnet Kaggle de l'exécution finale :

| Fichier | Rôle |
|---|---|
| `hgb_forensique.json` | classifieur forensique par patch (27 descripteurs de bruit) |
| `morceaux/`, `morceaux.json` | réseaux ONNX en morceaux de 4 Mo, avec taille et SHA-256 : CNN par patch (logit et segmentation), DDPM défenseur (erreur de reconstruction), générateur GAN de l'attaque. Fichiers entiers : `modeles/v7/` à la racine du dépôt |
| `fusion.json` | régressions logistiques locales (`loc2`, `local_fuser`) et seuils |
| `hgb_meta2.json` | méta-classifieur forensique + CNN (52 entrées) |
| `hgb_meta.json` | méta-classifieur des trois branches (60 entrées) |
| `reference.json` | sorties Python de référence pour la page de vérification |

Les arbres sont évalués en JavaScript (`site/js/hybride.js`), les réseaux avec ONNX Runtime Web. Contrôle sans navigateur : `node outils_site/verifier_v7.mjs`.

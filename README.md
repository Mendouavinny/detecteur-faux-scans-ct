# Détection de faux scans CT par l'analyse du bruit

Mémoire de Master 2 en science des données et intelligence artificielle de **Vinny Jordan Mendoua** (École nationale supérieure polytechnique de Douala) : *Détection et localisation de falsifications dans les coupes de scanner thoracique par analyse du bruit et apprentissage profond.*

On peut ajouter ou effacer une tumeur dans une coupe de scanner sans qu'un radiologue le voie. Il reste pourtant une trace : le bruit de l'image, que la falsification ne sait pas imiter. Ce dépôt contient le code du mémoire, le détecteur hybride et une application web.

## Essayer en ligne

- **Application web** : https://mendouavinny.github.io/detecteur-faux-scans-ct/
  Elle fonctionne sur téléphone, tablette et ordinateur, s'installe comme une application et s'ouvre ensuite sans connexion. Elle analyse une coupe DICOM avec la branche du bruit (instantanée) ou avec le détecteur complet (bruit, CNN et fusion, diffusion en option), entièrement dans le navigateur : aucune image n'est envoyée. On peut falsifier une coupe d'exemple, notamment avec le GAN entraîné du mémoire (injection ou suppression de nodule), et chaque analyse est inscrite dans un journal d'audit chaîné.
- **Détecteur complet** (CNN, diffusion et fusion) : https://detecteur-faux-scans-ct.onrender.com
  Il est hébergé sur l'offre gratuite de Render. Après 15 minutes d'inactivité, le serveur se met en veille : le premier accès prend alors environ une minute, et l'analyse d'une coupe dure de l'ordre d'une minute.

**N'y déposez aucune donnée de patient réel : ce sont des démonstrateurs de recherche.**

## Contenu du dépôt

| Élément | Rôle |
|---|---|
| [`memoire/`](memoire/) | Code du mémoire : carnet Kaggle, cellules complémentaires, simulateur SimCT-Tamper, résultats bruts, schémas d'architecture. Voir son [README](memoire/README.md). |
| [`site/`](site/) | Application web statique (HTML, CSS, JavaScript), publiée sur GitHub Pages par `.github/workflows/pages.yml` |
| `app.py` | Interface du détecteur complet (Gradio) |
| `detecteur.py` | Détecteur hybride : prétraitement, trois branches, fusion, attaques de démonstration, module de sécurité |
| `features.py` | Descripteurs forensiques du bruit |
| `modeles/` | Classifieurs et fusion entraînés, résultats de l'exécution ; `modeles/v7/` : réseaux ONNX de l'exécution finale (CNN, DDPM, GAN) |
| `exemples/` | 10 coupes DICOM authentiques de patients du test, jamais vus à l'entraînement |
| `outils_site/` | Export des modèles pour le site (`export_kaggle_v7.py`, à lancer dans le carnet Kaggle) et contrôles de parité JavaScript / Python |
| `preparer_exemples.py`, `verifier_fidelite.py` | Préparation et contrôle des exemples (exigent la base LIDC-IDRI en local) |

## Le détecteur

- **Prétraitement** : unités Hounsfield, écrêtage identique de toutes les coupes à [−1024 ; 3071] HU, segmentation des poumons, patchs de 64 × 64 pixels au pas de 32.
- **Branche forensique** : quatre résidus du bruit (médian, SRM 3 × 3, KV 5 × 5, différence), 27 descripteurs par patch (niveau, autocorrélation, spectre, kurtosis, écart robuste aux autres patchs de la coupe, duplication exacte), classifieur à gradient boosting.
- **Branche CNN** : couche contrainte de Bayar et Stamm, filtres SRM, encodeur EfficientNet-B0 et décodeur U-Net.
- **Branche diffusion** : erreur de reconstruction d'un DDPM appris sur des patchs authentiques.
- **Fusion du local au global** : scores des patchs et descripteurs de la coupe, seuil fixé sur la seule validation.
- **Module de sécurité** (prototype) : empreinte SHA-256, verdict à trois niveaux (conforme, à vérifier, alerte) et journal d'audit chaîné.

## Résultats du mémoire

Exécution finale (version 7 du carnet), 300 patients LIDC-IDRI séparés par patient : 210 pour l'entraînement, 45 pour la validation, 45 pour le test (413 coupes, dont 213 falsifiées).

| Méthode | AUC (test interne) |
|---|---|
| **Fusion des trois branches** | **0,870** |
| Branche forensique (bruit) | 0,735 |
| Branche CNN | 0,704 |
| Branche diffusion | 0,510 |
| Cascade de Zhang et al. (réimplémentée) | 0,631 |
| TruFor (poids publiés) | 0,491 |

- Le bruit sépare là où l'intensité échoue : 0,790 contre 0,482 sur les coupes réelles (+0,309, IC 95 % : 0,228 à 0,383).
- Au seuil fixé sur la validation, la fusion détecte 35 % des coupes falsifiées sans aucune fausse alerte. L'avance de la fusion bruit + CNN se retrouve dans cinq partitions des patients (0,881 ± 0,026).
- Face à des générateurs inconnus, l'avantage disparaît : 0,76 à 0,79 sur CT-GAN (scanners copiés de nos patients exclus), 0,55 à 0,79 sur M3Dsynth. La branche diffusion n'apporte aucun gain mesurable (−0,009) et la localisation reste faible (Dice 0,089).
- **Artefact d'écrêtage corrigé.** Dans les premières exécutions, seules les coupes falsifiées étaient écrêtées à −1024 HU. Cet indice trivial donnait à lui seul une AUC de 0,829 et gonflait celle de la fusion de 0,125. Toutes les coupes reçoivent désormais le même écrêtage.

**Modèles des applications.** L'application web utilise les modèles de l'exécution finale (version 7, 300 patients), exportés du carnet Kaggle dans `site/modele/v7/`. Le détecteur complet en Python (dossier `modeles/`, Render) utilise encore ceux d'une exécution antérieure à 120 patients, artefact d'écrêtage déjà corrigé : sur leur propre test interne, la branche forensique y atteint une AUC de 0,785 et la fusion robuste 0,772.

## L'application web

Le dossier `site/` ne contient que des fichiers statiques.

- `site/js/forensique.js` reprend en JavaScript la branche forensique de `detecteur.py` et `features.py`. Il reproduit l'arithmétique en simple précision de NumPy et évalue les arbres du classifieur exportés dans `site/modele/v7/hgb_forensique.json`.
- `site/js/globaux.js` calcule les 40 descripteurs de la coupe (bruit global, anneaux du spectre, blocs, duplication) utilisés par le méta-classifieur.
- `site/js/hybride.js` exécute le CNN, le DDPM et le générateur GAN avec ONNX Runtime Web, puis la fusion locale et le méta-classifieur de la version 7 : « bruit + CNN » (52 entrées, AUC 0,879) par défaut, « trois branches » (60 entrées, AUC 0,870) avec la diffusion.
- Les réseaux sont servis en morceaux de 4 Mo (`site/modele/v7/morceaux/`, vérifiés par SHA-256) : sur une connexion qui coupe, seul le morceau interrompu est retéléchargé. Ils sont ensuite gardés dans le cache du navigateur. Les fichiers ONNX entiers sont dans `modeles/v7/`.
- Parité avec Python sur les 10 coupes d'exemple (`site/modele/v7/reference.json`, calculé dans le carnet) : probabilités forensiques identiques à 10⁻¹⁶ près, descripteurs de la coupe à 4 × 10⁻⁵ près en relatif, scores des deux méta-classifieurs identiques à partir des mêmes sorties de réseaux. `node outils_site/verifier_v7.mjs` refait ce contrôle ; la page `site/verification.html` le refait dans le navigateur avec les réseaux ONNX.
- `site/sw.js` et `site/manifest.webmanifest` rendent l'application installable et utilisable hors connexion.

Pour contrôler les modèles v7 :

```bash
node outils_site/verifier_v7.mjs        # branche forensique, descripteurs et méta-classifieurs contre reference.json
```

Pour l'ouvrir en local : `python -m http.server --directory site`, puis http://localhost:8000.

## Installer le détecteur complet

Il faut Python 3.11 ou plus récent. Le calcul se fait sur processeur, environ 3 s par coupe ; aucune carte graphique n'est nécessaire.

```bash
git clone https://github.com/Mendouavinny/detecteur-faux-scans-ct
cd detecteur-faux-scans-ct
python -m venv .venv
.venv\Scripts\activate          # Windows ; sous Linux/macOS : source .venv/bin/activate
pip install -r requirements-render.txt
python telecharger_modeles.py
python app.py
```

Ouvrir ensuite http://127.0.0.1:7860. L'application n'écoute que sur ce poste et n'envoie aucune image. Sous Windows, on peut aussi double-cliquer sur `lancer_application.bat`.

Les réseaux du détecteur Python pèsent 17 Mo chacun et ne sont pas dans ce dépôt. Ils sont publiés sur Hugging Face : [MENDOUA/detecteur-faux-scans-ct-modeles](https://huggingface.co/MENDOUA/detecteur-faux-scans-ct-modeles).

- `python telecharger_modeles.py` télécharge dans `modeles/` les versions ONNX (`cnn.onnx`, `ddpm.onnx`, `gan.onnx`). Avec elles, PyTorch n'est pas nécessaire.
- Avec PyTorch, on peut aussi placer dans `modeles/` les poids d'origine `cnn_best.pt`, `ddpm_attaquant.pt` et `gan_G.pt`. Le moteur se choisit avec `DETECTEUR_MOTEUR=torch` ou `DETECTEUR_MOTEUR=onnx`. Un autre dossier de modèles peut être indiqué avec `DETECTEUR_MODELES`.

En ligne, les réseaux tournent en ONNX Runtime, sans PyTorch, pour tenir dans 512 Mo de mémoire. `comparer_moteurs.py` vérifie que les deux moteurs donnent les mêmes scores (écart maximal 3 × 10⁻⁷ sur les 10 coupes d'exemple).

## Limites

- **Outil de recherche, pas de diagnostic.** C'est une aide à la vigilance, sans aucune validation clinique.
- **Seuils non calibrés cliniquement.** Le verdict compare le score à 0,5 et 0,85. Sur les coupes d'exemple, la branche forensique seule produit aussi des fausses alertes.
- **DICOM d'origine indispensable.** Une compression ou un rééchantillonnage efface l'indice de bruit.
- **Générateurs inconnus.** La détection ne se transpose pas aux générateurs absents de l'entraînement.
- **Localisation grossière.** Les cartes sont construites sur des patchs de 64 × 64 pixels.

## Données

Les coupes d'exemple et les données d'entraînement proviennent de la base publique **LIDC-IDRI** (The Cancer Imaging Archive), diffusée sous licence [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

> Armato SG III et al. *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.* Medical Physics, 38(2):915–931, 2011.

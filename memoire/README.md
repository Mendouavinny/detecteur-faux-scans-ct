# Code du mémoire

*Détection et localisation de falsifications dans les coupes de scanner thoracique par analyse du bruit et apprentissage profond*, mémoire de Master 2 en science des données et intelligence artificielle, Vinny Jordan Mendoua, École nationale supérieure polytechnique de Douala.

Ce dossier rassemble le code qui produit les résultats du mémoire. Le détecteur prêt à l'emploi est à la racine du dépôt (`app.py`, `detecteur.py`), et l'application web est dans `site/`.

## Contenu

| Dossier | Contenu | Chapitre |
|---|---|---|
| `carnet/` | Carnet Kaggle principal (`detection_faux_scans_LIDC.ipynb`) et sa version allégée pour Kaggle. Il est généré par `build_nb.py` à partir des cellules `cells_a.py` (configuration, données, attaques), `cells_b.py` et `cells_c.py` (branches, fusion, évaluation). | 2 et 3 |
| `complements/` | Cellules ajoutées au carnet après l'audit et scripts d'évaluation externe (voir le tableau ci-dessous). | 3 |
| `simulateur/` | Preuve de concept SimCT-Tamper : simulateur physique du scanner (projection de Radon, bruit de Poisson, rétroprojection filtrée avec quatre noyaux), cinq attaques camouflées, expériences et décomposition H1/H2. | 3.1.1 à 3.1.6 |
| `resultats/` | Résultats bruts relevés dans les journaux Kaggle (JSON) et scripts de ré-analyse des tableaux. | 3 |
| `figures_architecture/` | Scripts des schémas d'architecture (figures 2.1 et 2.7 à 2.11). | 2 |

### Compléments

| Script | Rôle | Résultats |
|---|---|---|
| `complements/v5_cell.py` | Nodules réels, test contrôlé, seuils fixés sur la validation, décomposition H1/H2 sur LIDC-IDRI, cascade de Zhang et al. réimplémentée, partitions supplémentaires (tableaux 3.8, 3.9, 3.12, 3.13) | `resultats/v7_resultats.json` |
| `complements/ablation_cell.py` | Ablation de la branche diffusion, mêmes patients et même protocole (tableau 3.7) | `resultats/v7_principal.json` |
| `complements/verif_ecretage.py` | Mesure de l'artefact d'écrêtage à −1024 HU (tableau 3.11) | `resultats/verif_ecretage.json` |
| `complements/m3d_extract.py` | Extraction des coupes de M3Dsynth par lecture en flux des archives (sans GPU) | — |
| `complements/m3d_v7.py` | Test externe M3Dsynth avec les modèles de l'exécution corrigée (tableau 3.10) | `resultats/m3d_resultats_v7.json` |
| `complements/wp1_identite.py`, `provenance_ctgan.py` | Identification des scanners de CT-GAN copiés de patients LIDC-IDRI (section 3.2.1) | `resultats/identite_ctgan.json`, `provenance_ctgan_resume.json` |
| `complements/externe_eval.py`, `m3d_eval.py` | Premières évaluations M3Dsynth, affectées par l'artefact d'écrêtage, conservées pour mémoire | `resultats/m3d_resultats.json` |
| `complements/test_v5_mock.py` | Banc d'essai local de `v5_cell.py`, sans GPU | — |

Les fichiers `v6_resultats.json`, `ablation_kaggle_v4.json`, `tableau_3_9.json` et `resultats_kaggle_v3.json` documentent les exécutions antérieures à la correction de l'écrêtage.

## L'exécution corrigée (version 7)

Les chiffres du mémoire viennent de la version 7 du carnet, exécutée sur Kaggle (GPU, 3 h 10 min) avec 300 patients LIDC-IDRI séparés par patient : 210 pour l'entraînement, 45 pour la validation, 45 pour le test. Par rapport au carnet de ce dossier, elle ajoute deux lignes à la cellule d'amorçage. Elles appliquent à **toutes** les coupes lues, authentiques, falsifiées et externes, le même écrêtage à [−1024 ; 3071] HU. Sans cette correction, seules les coupes falsifiées étaient écrêtées. Le modèle apprenait alors cet indice, qui gonflait son AUC de 0,125 (section 3.1.11 du mémoire).

| Résultat (test interne, 45 patients) | AUC |
|---|---|
| Fusion des trois branches | 0,870 |
| Branche forensique (bruit) | 0,735 |
| Branche CNN | 0,704 |
| Branche diffusion | 0,510 |
| Cascade de Zhang et al. (réimplémentée) | 0,631 |
| TruFor (poids publiés) | 0,491 |

## Reproduire

### Le carnet principal

1. Créer un carnet Kaggle avec GPU et accès à internet, puis y importer `carnet/detection_faux_scans_LIDC.ipynb`.
2. Les données LIDC-IDRI se téléchargent automatiquement avec le client NBIA (accès invité public). Le jeu externe CT-GAN est disponible sur l'*UCI Machine Learning Repository* (identifiant 520).
3. Exécuter les cellules dans l'ordre, puis les cellules de `complements/` comme indiqué dans leur en-tête.

### La preuve de concept (sans GPU)

```bash
cd simulateur
python generate.py              # simct_tamper.npz : 1 080 coupes, environ 4 min sur 2 cœurs
python experiments.py           # expériences E1, E3 à E7 -> results.json
python hybrid.py                # approche hiérarchique (E2) -> results_hybrid.json
python decomposition_table.py   # tableau 3.12 (décomposition H1/H2)
mkdir -p figures && python figures.py
```

Dépendances : Python ≥ 3.10, NumPy, SciPy, scikit-image, scikit-learn, OpenCV et Matplotlib. Les graines aléatoires sont fixées ; de légères variations numériques peuvent apparaître selon les versions des bibliothèques.

## Données

Base **LIDC-IDRI** (The Cancer Imaging Archive), licence CC BY 3.0 : Armato SG III et al., *Medical Physics*, 38(2):915–931, 2011. Jeux externes : CT-GAN (Mirsky et al., 2019) et M3Dsynth (Zingarini et al., 2024).

# Détection automatique de faux scans CT — application de démonstration

Application web locale (Gradio) du mémoire de Master Recherche SDIA (ENSPD) de **MENDOUA Vinny Jordan** :
*Détection automatique de faux scans CT : approche hybride forensique et apprentissage profond.*

Elle applique à une coupe CT au format DICOM le détecteur du mémoire. Les modèles sont ceux de la 4ᵉ exécution du carnet (120 patients LIDC-IDRI, artefact d'écrêtage corrigé) et l'application ne les réentraîne pas.

- **Prétraitement** : unités Hounsfield, segmentation des poumons, patchs de 64 × 64 pixels.
- **Branche forensique** : descripteurs du bruit tomographique (résidus, autocorrélation, spectre, incohérence locale, duplication exacte).
- **Branche CNN** : couche contrainte de Bayar et Stamm, filtres SRM, encodeur EfficientNet-B0 et décodeur U-Net.
- **Branche diffusion** : erreur de reconstruction d'un DDPM appris sur des patchs authentiques.
- **Fusion robuste** : régression logistique à poids positifs, choisie sur la seule validation.
- **Module de sécurité** : empreinte SHA-256, verdict à trois niveaux (conforme / à vérifier / alerte) et journal d'audit chaîné.

## Onglets

1. **Analyser une coupe DICOM.** Affiche le verdict, le score de chaque branche, la carte de falsification et l'empreinte SHA-256. Des exemples authentiques sont fournis.
2. **Démonstration.** Falsifie une coupe authentique (copier-déplacer, bruit, injection et suppression par GAN, inpainting), puis analyse l'originale et la version falsifiée.
3. **Journal d'audit.** Liste les analyses et vérifie l'intégrité du journal chaîné.
4. **À propos et limites.**

## Installation

Il faut Python 3.11 ou plus récent. Le calcul se fait sur processeur, environ 3 s par coupe ; aucune carte graphique n'est nécessaire.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows ; sous Linux/macOS : source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Ouvrir ensuite http://127.0.0.1:7860. L'application n'écoute que sur ce poste et n'envoie aucune image. Sous Windows, on peut aussi double-cliquer sur `lancer_application.bat`.

## Contenu

| Élément | Rôle |
|---|---|
| `app.py` | Interface web (Gradio) |
| `detecteur.py` | Détecteur repris du carnet : chargement des modèles, branches, fusion, attaques de démonstration, module de sécurité |
| `features.py` | Descripteurs forensiques (copie du fichier écrit par le carnet) |
| `modeles/` | Modèles entraînés (CNN, DDPM, GAN de démonstration, fusions) et résultats de l'exécution |
| `exemples/` | 10 coupes DICOM authentiques de patients du test, jamais vus à l'entraînement |
| `preparer_exemples.py`, `verifier_fidelite.py` | Outils de préparation et de contrôle ; ils exigent la base LIDC-IDRI en local |
| `verification_fidelite.json` | Résultat du contrôle de fidélité |

Un autre dossier de modèles peut être indiqué avec la variable d'environnement `DETECTEUR_MODELES`.

## Résultats (mémoire, tableau 3.6)

| Branche | AUC test interne (153 coupes) | AUC CT-GAN externe (37 coupes) |
|---|---|---|
| Forensique | 0,785 | 0,791 |
| CNN contraint | 0,731 | 0,724 |
| Diffusion | 0,478 | 0,439 |
| **Fusion robuste (retenue)** | 0,772 | 0,788 |
| Méta-classifieur HGB (comparaison) | 0,827 | 0,758 |

Aucune fusion ne dépasse significativement la branche forensique seule. Les détails, intervalles de confiance compris, sont dans le mémoire (§3.1.7).

**Fidélité au carnet.** Sur les 76 coupes authentiques du test interne, l'application reproduit les scores du carnet à 0,0002 près en moyenne.

## Limites

- **Outil de recherche, pas de diagnostic.** C'est une aide à la vigilance, sans aucune validation clinique.
- **Seuils non calibrés.** Au seuil de 0,5, environ 78 % des falsifications sont signalées, mais aussi plus de deux coupes authentiques sur cinq.
- **DICOM d'origine indispensable.** Un rééchantillonnage efface l'indice de bruit.
- **Localisation grossière.** La carte est construite sur des patchs de 64 × 64 pixels.

## Données

Les coupes d'exemple et les données d'entraînement proviennent de la base publique **LIDC-IDRI** (The Cancer Imaging Archive), diffusée sous licence [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

> Armato SG III et al. *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.* Medical Physics, 38(2):915–931, 2011.

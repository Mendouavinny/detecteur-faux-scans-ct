"""Application web de démonstration — Détection automatique de faux scans CT (MENDOUA Vinny Jordan).

Lancement : lancer_application.bat (ou python app.py), puis ouvrir http://127.0.0.1:7860
L'application n'écoute que sur le poste local (127.0.0.1) : aucune image ne quitte la machine.
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import gradio as gr

import detecteur as D

ICI = Path(__file__).resolve().parent
EX = ICI / "exemples"
INDEX = json.load(open(EX / "index.json", encoding="utf-8")) if (EX / "index.json").exists() else []
AUDIT = D.AuditLog(ICI / "journal" / "audit_log.jsonl")
print("Chargement des modèles de l'exécution finale (v7, 300 patients)…", flush=True)
DET = D.DetecteurV7()
TAU1_DEF, TAU2_DEF = float(DET.seuils["tau1"]), float(DET.seuils["tau2"])   # 0,5 et 0,85 (fusion.json)
print("Modèles chargés :", DET.dossier, flush=True)
NOM_SCORE = {False: "fusion bruit + CNN", True: "fusion des trois branches"}

COULEURS = {"CONFORME": ("#1b7f4b", "#e6f4ec"), "A VERIFIER": ("#9a6700", "#fff4d6"), "ALERTE": ("#b42318", "#fde8e6")}
LIBELLE = {"CONFORME": "Conforme", "A VERIFIER": "À vérifier", "ALERTE": "Alerte"}


# ------------------------------------------------------------------ rendu des images
def fenetre(img, lo=-1350, hi=150):
    g = (np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def superposer(img, carte, seuil=0.05, alpha=0.6):
    base = fenetre(img)
    c = (np.clip(carte, 0, 1) * 255).astype(np.uint8)
    col = cv2.cvtColor(cv2.applyColorMap(c, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    m = (carte > seuil)[..., None]
    a = alpha * np.clip(carte, 0, 1)[..., None] ** 0.5
    return np.where(m, (base * (1 - a) + col * a), base).astype(np.uint8)


def contour(rgb, masque, couleur=(0, 230, 120)):
    out = rgb.copy()
    cs, _ = cv2.findContours(masque.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, couleur, 2)
    return out


def croix_max(rgb, carte):
    out = rgb.copy()
    y, x = np.unravel_index(np.argmax(carte), carte.shape)
    cv2.drawMarker(out, (int(x), int(y)), (60, 200, 255), cv2.MARKER_CROSS, 28, 2)
    return out


# ------------------------------------------------------------------ fragments HTML
def html_verdict(v, score, tau1, tau2, hash_ok, n_patchs, duree, avec_diffusion=False):
    fg, bg = COULEURS[v]
    raison = ("empreinte SHA-256 différente de l’empreinte enregistrée" if not hash_ok else
              f"score {score:.3f} {'>' if v != 'CONFORME' else '≤'} τ{'2' if v == 'ALERTE' else '1'} = "
              f"{tau2 if v == 'ALERTE' else tau1:.2f}")
    return (f"<div style='border:2px solid {fg};background:{bg};border-radius:12px;padding:16px 20px'>"
            f"<div style='font-size:13px;color:{fg};letter-spacing:.08em;text-transform:uppercase'>Verdict</div>"
            f"<div style='font-size:34px;font-weight:700;color:{fg};line-height:1.2'>{LIBELLE[v]}</div>"
            f"<div style='font-size:15px;color:#333;margin-top:6px'>Score de falsification ({NOM_SCORE[avec_diffusion]}, "
            f"0 = authentique, 1 = falsifiée) : "
            f"<b>{score:.3f}</b> — {raison}</div>"
            f"<div style='font-size:13px;color:#555;margin-top:4px'>{n_patchs} patchs pulmonaires analysés en "
            f"{duree:.1f} s</div></div>")


def html_branches(r):
    b = r["branches"]
    nom = NOM_SCORE[r["avec_diffusion"]]
    lignes = [(nom[0].upper() + nom[1:] + " (retenue)", r["score"], True, "méta-classifieur, probabilité de falsification")]
    if r["avec_diffusion"]:
        lignes.append(("Fusion bruit + CNN (comparaison)", r["score_bruit_cnn"], False, "même coupe, sans la branche diffusion"))
    lignes += [("Branche du bruit", b["forensique"], False, "moyenne des 3 patchs les plus suspects"),
               ("Branche CNN contrainte", b["cnn"], False, "moyenne des 3 patchs les plus suspects"),
               ("Fusion locale des patchs", b["fusion_locale"], False, "moyenne des 3 patchs les plus suspects")]
    h = "<div style='font-family:inherit'>"
    for nom, v, fort, aide in lignes:
        coul = "#eb6834" if fort else "#2a78d6"
        h += (f"<div style='margin:8px 0'><div style='display:flex;justify-content:space-between;font-size:14px'>"
              f"<span style='font-weight:{700 if fort else 500}'>{nom}</span><span><b>{v:.3f}</b></span></div>"
              f"<div style='background:#e9e9e4;border-radius:6px;height:10px;margin-top:3px'>"
              f"<div style='width:{max(0, min(1, v)) * 100:.1f}%;background:{coul};height:10px;border-radius:6px'></div>"
              f"</div><div style='font-size:12px;color:#777'>{aide}</div></div>")
    diff = (f"Branche diffusion (|écart robuste| de l'erreur de reconstruction, 3 patchs les plus suspects) : "
            f"{b['diffusion']:.2f}" if r["avec_diffusion"] else "Branche diffusion non calculée (case décochée)")
    h += (f"<div style='margin-top:10px;font-size:13px;color:#555'>{diff} · Duplication exacte maximale par patch : "
          f"{r['duplication']:.3f}</div>")
    return h + "</div>"


# ------------------------------------------------------------------ onglet 1 : analyser une coupe DICOM
def lire_dicom(fichier, exemple):
    if fichier is not None:
        chemin = fichier if isinstance(fichier, str) else fichier.name
    elif exemple:
        chemin = str(EX / exemple)
    else:
        raise gr.Error("Déposez un fichier DICOM ou choisissez un exemple.")
    try:
        ds = pydicom.dcmread(chemin)
        img = D.dicom_to_hu(ds)
    except Exception as e:
        raise gr.Error(f"Fichier DICOM illisible : {e}")
    if img.ndim != 2:
        raise gr.Error("Le fichier doit contenir une seule coupe CT (image 2D).")
    return ds, img, Path(chemin).name


def analyser_dicom(fichier, exemple, empreinte_ref, tau1, tau2, utilisateur, avec_diffusion=False):
    ds, img, nom = lire_dicom(fichier, exemple)
    t = time.time()
    r = DET.analyser(img, avec_diffusion=bool(avec_diffusion))
    duree = time.time() - t
    fp = D.dicom_fingerprint(ds)
    ref = (empreinte_ref or "").strip().lower()
    hash_ok = (not ref) or ref == fp
    infos = (f"**Fichier :** {nom}  \n**SOPInstanceUID :** {getattr(ds, 'SOPInstanceUID', '—')}  \n"
             f"**Empreinte SHA-256 (pixels + attributs critiques) :** `{fp}`  \n"
             + ("**Empreinte de référence :** non fournie (contrôle actif non effectué)" if not ref else
                ("**Empreinte de référence :** identique ✅" if hash_ok else
                 "**Empreinte de référence :** DIFFÉRENTE ❌ — le fichier a été modifié après l’enregistrement")))
    if not r["ok"]:
        return ("<div style='padding:14px;border:2px solid #777;border-radius:12px'>" + r["message"] + "</div>",
                "", fenetre(img), None, None, infos, None)
    v = D.verdict(r["score"], tau1, tau2, hash_ok)
    rec = AUDIT.append(dict(fichier=nom, sop=str(getattr(ds, "SOPInstanceUID", "")), sha256=fp, hash_ok=hash_ok,
                            score=round(r["score"], 4), verdict=v, user=utilisateur or "service-verif",
                            modele="v7 " + NOM_SCORE[r["avec_diffusion"]],
                            tau1=round(tau1, 3), tau2=round(tau2, 3)))
    carte = croix_max(superposer(img, r["carte"]), r["carte"])
    return (html_verdict(v, r["score"], tau1, tau2, hash_ok, r["n_patchs"], duree, r["avec_diffusion"]), html_branches(r),
            contour(fenetre(img), r["lung"], (90, 160, 255)), carte, superposer(img, r["carte_cnn"]), infos,
            json.dumps(rec, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ onglet 2 : démonstration d'attaque
def demo(exemple, attaque_lib, graine, tau1, tau2, avec_diffusion=False):
    if not exemple:
        raise gr.Error("Choisissez une coupe authentique.")
    nom = next(k for k, v in D.ATTAQUES.items() if v == attaque_lib)
    meta = next(e for e in INDEX if e["fichier"] == exemple)
    img = D.dicom_to_hu(pydicom.dcmread(EX / exemple))
    try:
        fausse, masque = D.appliquer_attaque(DET, nom, img, meta["nodules"], int(graine))
    except ValueError as e:
        raise gr.Error(str(e))
    r0, r1 = DET.analyser(img, bool(avec_diffusion)), DET.analyser(fausse, bool(avec_diffusion))
    if not (r0["ok"] and r1["ok"]):
        raise gr.Error("Coupe non analysable (poumon trop petit).")
    v0, v1 = D.verdict(r0["score"], tau1, tau2), D.verdict(r1["score"], tau1, tau2)
    carte = r1["carte"]
    y, x = np.unravel_index(np.argmax(carte), carte.shape)
    globale = nom in D.GLOBAL_ATTACKS
    touche = bool(masque[y, x]) if not globale else None
    diff = np.abs(fausse.astype(np.int32) - img.astype(np.int32))
    def ligne(titre, rr, vv):
        fg, bg = COULEURS[vv]
        return (f"<tr><td style='padding:6px 10px'>{titre}</td><td style='padding:6px 10px;text-align:center'>"
                f"<b>{rr['score']:.3f}</b></td><td style='padding:6px 10px;text-align:center'>{rr['branches']['forensique']:.3f}</td>"
                f"<td style='padding:6px 10px;text-align:center'><span style='background:{bg};color:{fg};"
                f"padding:3px 10px;border-radius:10px;font-weight:600'>{LIBELLE[vv]}</span></td></tr>")
    th = "style='padding:8px 10px;background:#14305a;color:#ffffff !important;font-weight:600;text-align:{}'"
    res = ("<table style='width:100%;border-collapse:collapse;font-size:15px'><tr>"
           f"<th {th.format('left')}>Coupe</th><th {th.format('center')}>Score de falsification ({NOM_SCORE[bool(avec_diffusion)]})</th>"
           f"<th {th.format('center')}>Branche du bruit seule</th><th {th.format('center')}>Verdict</th></tr>"
           + ligne("Authentique (originale)", r0, v0) + ligne("Falsifiée", r1, v1) + "</table>")
    res += (f"<p style='font-size:14px;margin-top:10px'>Attaque : <b>{attaque_lib}</b> — {int(masque.sum())} pixels "
            f"modifiés ({100 * masque.mean():.2f} % de la coupe), écart maximal {int(diff.max())} HU. ")
    if touche is not None:
        res += ("Le point le plus suspect de la carte (croix bleue) <b>tombe dans la région falsifiée</b>."
                if touche else "Le point le plus suspect de la carte (croix bleue) <b>est hors de la région "
                               "falsifiée</b> : la localisation est grossière (patchs de 64 × 64 pixels).")
    else:
        res += "Attaque globale : toute la coupe est concernée, la carte n’a pas de région cible."
    res += "</p>"
    vue = contour(fenetre(fausse), masque) if not globale else fenetre(fausse)
    loupe = (np.clip(diff / max(1, np.percentile(diff[diff > 0], 99) if (diff > 0).any() else 1), 0, 1) * 255).astype(np.uint8)
    return (fenetre(img), vue, croix_max(contour(superposer(fausse, carte), masque) if not globale
                                         else superposer(fausse, carte), carte),
            cv2.cvtColor(cv2.applyColorMap(loupe, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB), res)


# ------------------------------------------------------------------ onglet 3 : journal d'audit
def journal():
    e = AUDIT.entrees()
    ok, i = AUDIT.verify()
    etat = ("✅ Journal intègre : la chaîne des empreintes est cohérente." if ok else
            f"❌ Journal altéré à l’entrée {i + 1} : une entrée a été modifiée ou supprimée.")
    rows = [[x.get("time", ""), x.get("fichier", ""), x.get("score", ""), x.get("verdict", ""),
             "oui" if x.get("hash_ok", True) else "NON", x.get("user", ""), x.get("entry_hash", "")[:16] + "…"]
            for x in reversed(e)]
    return f"**{len(e)} entrée(s).** {etat}", rows


# ------------------------------------------------------------------ interface
CSS = """
.gradio-container {max-width: 1320px !important}
#entete {background: linear-gradient(90deg,#10294f,#1d4f8c); color: #fff; border-radius: 14px; padding: 18px 26px}
#entete h1 {color:#fff; margin:0; font-size:28px} #entete p {color:#cfe0f7; margin:4px 0 0 0}
"""
def etiquette(e):
    s = e.get("score_authentique")
    return (f"{e['patient']} · coupe {e['coupe']}" + (" · avec nodule" if e["nodules"] else "")
            + (f" · score {s:.3f}".replace(".", ",") + (" (fausse alerte à 0,5)" if s > TAU1_DEF else "") if s is not None else ""))


EX_CHOIX = [(etiquette(e), e["fichier"]) for e in INDEX]
EX_NOD = [e for e in INDEX if e["nodules"]]
# par défaut : coupes représentatives (bien classées) ; les fausses alertes restent visibles dans la liste
DEF_ANALYSE = min(INDEX, key=lambda e: e.get("score_authentique", 1))["fichier"] if INDEX else None
DEF_DEMO = min(EX_NOD, key=lambda e: e.get("score_authentique", 1))["fichier"] if EX_NOD else None

def fr(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


EN_LIGNE = bool(os.environ.get("EN_LIGNE") or os.environ.get("SPACE_ID"))   # version hébergée en ligne
CONFIDENTIALITE = (
    "- **Version en ligne de démonstration.** Les coupes déposées sont traitées sur un serveur distant et le "
    "journal d’audit est commun à tous les visiteurs : n’y déposez **aucune donnée de patient réel**. Les exemples "
    "fournis proviennent de la base publique LIDC-IDRI." if EN_LIGNE else
    "- Tout s’exécute sur ce poste : l’application n’écoute que l’adresse locale 127.0.0.1 et n’envoie aucune image.")


def a_propos():
    return f"""
### Ce que fait l’application
Elle applique à une coupe CT le **détecteur hybride** du mémoire, avec les modèles de l’**exécution finale**
(version 7 du carnet : 300 patients LIDC-IDRI séparés par patient, aucun réentraînement ici) :

1. **Prétraitement** : unités Hounsfield, écrêtage identique de toutes les coupes à [−1024 ; 3071] HU, segmentation des poumons, patchs de 64 × 64 pixels au pas de 32.
2. **Branche du bruit** : 27 descripteurs par patch (résidus médian et KV, autocorrélation, spectre, écart robuste aux autres patchs, duplication exacte), classés par gradient boosting (300 arbres).
3. **Branche CNN** : couche contrainte de Bayar et Stamm + filtres SRM, encodeur EfficientNet-B0, décodeur U-Net (carte de segmentation).
4. **Branche diffusion** (facultative, plus lente) : erreur de reconstruction d’un DDPM appris sur des patchs authentiques.
5. **Fusion du local au global** : fusion locale des patchs, puis méta-classifieur sur les 40 descripteurs de la coupe et les agrégats des patchs. Sans diffusion : « bruit + CNN » ; avec diffusion : « trois branches ».
6. **Module de sécurité** : empreinte SHA-256, politique à trois niveaux (conforme / à vérifier / alerte) et journal d’audit chaîné.

### Performances mesurées (mémoire, exécution finale)
Test interne : 45 patients, 413 coupes dont 213 falsifiées.

| Méthode | AUC (test interne) |
|---|---|
| **Fusion bruit + CNN** (score par défaut) | **0,879** |
| **Fusion des trois branches** (avec diffusion) | **0,870** |
| Branche du bruit | 0,735 |
| Branche CNN | 0,704 |
| Branche diffusion | 0,510 |
| Cascade de Zhang et al. (réimplémentée) | 0,631 |
| TruFor (poids publiés) | 0,491 |

Face à des générateurs absents de l’entraînement, l’avantage disparaît : AUC de 0,76 à 0,79 sur CT-GAN, de 0,55 à
0,79 sur M3Dsynth. La branche diffusion n’apporte aucun gain mesurable (−0,009).

### À savoir avant toute utilisation
- **Outil de recherche, pas de diagnostic.** Il s’agit d’une aide à la vigilance ; aucune validation clinique n’a été faite.
- **Seuils non calibrés cliniquement.** Par défaut, τ1 = {fr(TAU1_DEF)} et τ2 = {fr(TAU2_DEF)}. Dans le mémoire, au
  seuil fixé sur la validation, la fusion détecte 35 % des coupes falsifiées du test interne sans aucune fausse alerte.
- **Générateurs inconnus.** La détection ne se transpose pas aux générateurs absents de l’entraînement.
- **Opérer sur le DICOM d’origine.** Une compression ou un rééchantillonnage efface l’indice de bruit.
- **Localisation grossière** : la carte est construite sur des patchs de 64 × 64 pixels (Dice 0,089 dans le mémoire).
- **Artefact d’écrêtage corrigé.** Dans les premières exécutions, seules les coupes falsifiées étaient écrêtées à
  −1024 HU : cet indice trivial donnait à lui seul une AUC de 0,829. Toutes les coupes reçoivent désormais le même
  écrêtage, et l’indice tombe à 0,500.
- **Fidélité vérifiée** : sur les 10 coupes d’exemple, l’application reproduit les sorties du carnet
  (`python comparer_moteurs.py --diffusion` : mêmes scores, à moins de 10⁻⁶ près pour les réseaux).
{CONFIDENTIALITE}

*Mémoire de Master Recherche SDIA (ENSPD) — MENDOUA Vinny Jordan.*
"""


with gr.Blocks(title="Détection de faux scans CT") as app:
    gr.HTML("<div id='entete'><h1>Détection automatique de faux scans CT</h1>"
            "<p>Approche hybride forensique + apprentissage profond — démonstrateur du mémoire de Master "
            "Recherche SDIA (ENSPD), MENDOUA Vinny Jordan — modèles de l’exécution finale (300 patients)</p>"
            + ("<p><b>Version en ligne de démonstration : ne déposez aucune donnée de patient réel.</b></p>"
               if EN_LIGNE else "") + "</div>")
    with gr.Accordion("Réglages de la politique d’alerte", open=False):
        with gr.Row():
            tau1 = gr.Slider(0.05, 0.99, TAU1_DEF, step=0.01, label="τ1 : au-dessus → « à vérifier »")
            tau2 = gr.Slider(0.05, 0.999, TAU2_DEF, step=0.01, label="τ2 : au-dessus → « alerte »")
            user = gr.Textbox("service-verif", label="Utilisateur inscrit au journal")

    with gr.Tab("1 · Analyser une coupe DICOM"):
        with gr.Row():
            with gr.Column(scale=1):
                f_in = gr.File(label="Déposer une coupe CT au format DICOM (.dcm)", file_types=[".dcm", ".dicom", ""])
                ex_in = gr.Dropdown(EX_CHOIX, label="… ou choisir un exemple authentique (patients du test, jamais vus)",
                                    value=DEF_ANALYSE,
                                    info="Le score affiché est celui du détecteur (fusion bruit + CNN) sur la coupe "
                                         "authentique ; une seule des dix dépasse 0,5.")
                diff_in = gr.Checkbox(False, label="Ajouter la branche diffusion (fusion des trois branches, analyse plus lente)")
                ref_in = gr.Textbox(label="Empreinte SHA-256 enregistrée à l’acquisition (facultatif)",
                                    placeholder="Collez ici l’empreinte de référence pour le contrôle actif")
                go = gr.Button("Analyser", variant="primary", size="lg")
                v_out = gr.HTML()
                b_out = gr.HTML()
            with gr.Column(scale=2):
                with gr.Row():
                    i0 = gr.Image(label="Coupe (fenêtre pulmonaire) et masque pulmonaire", type="numpy", height=400)
                    i1 = gr.Image(label="Carte de falsification fusionnée (croix : point le plus suspect)",
                                  type="numpy", height=400)
                with gr.Row():
                    i2 = gr.Image(label="Carte de segmentation de la branche CNN", type="numpy", height=400)
                    with gr.Column():
                        inf = gr.Markdown()
                        rec = gr.Code(label="Entrée ajoutée au journal d’audit", language="json")
        go.click(analyser_dicom, [f_in, ex_in, ref_in, tau1, tau2, user, diff_in], [v_out, b_out, i0, i1, i2, inf, rec])

    with gr.Tab("2 · Démonstration : fabriquer puis détecter un faux"):
        gr.Markdown("Une coupe **authentique** d’un patient du test est falsifiée ici, avec les attaques du carnet "
                    "(dont le GAN entraîné). L’originale et la version falsifiée sont ensuite analysées. "
                    "Le contour vert indique la vérité terrain de la région modifiée. Les images falsifiées restent "
                    "dans l’application et ne sont pas enregistrées.")
        with gr.Row():
            d_ex = gr.Dropdown(EX_CHOIX, label="Coupe authentique", value=DEF_DEMO, scale=2,
                               info="Les attaques de suppression exigent une coupe « avec nodule ».")
            d_atk = gr.Dropdown(list(D.ATTAQUES.values()), value=D.ATTAQUES["gan_injection"], label="Attaque")
            d_seed = gr.Number(1, label="Graine (position et tirages)", precision=0)
            d_diff = gr.Checkbox(False, label="Avec la branche diffusion (plus lent)")
            d_go = gr.Button("Falsifier puis analyser", variant="primary")
        d_res = gr.HTML()
        with gr.Row():
            d0 = gr.Image(label="Originale", type="numpy", height=360)
            d1 = gr.Image(label="Falsifiée (contour vert : région modifiée)", type="numpy", height=360)
            d2 = gr.Image(label="Carte de détection sur la coupe falsifiée", type="numpy", height=360)
            d3 = gr.Image(label="Différence |falsifiée − originale|", type="numpy", height=360)
        d_go.click(demo, [d_ex, d_atk, d_seed, tau1, tau2, d_diff], [d0, d1, d2, d3, d_res])

    with gr.Tab("3 · Journal d’audit"):
        gr.Markdown("Chaque analyse est inscrite dans un journal chaîné : chaque entrée contient l’empreinte de la "
                    "précédente, si bien que toute modification ou suppression ultérieure est détectée.")
        j_btn = gr.Button("Actualiser et vérifier l’intégrité")
        j_etat = gr.Markdown()
        j_tab = gr.Dataframe(headers=["Date", "Fichier", "Score", "Verdict", "Empreinte conforme", "Utilisateur",
                                      "Empreinte de l’entrée"], interactive=False, wrap=True)
        j_btn.click(journal, None, [j_etat, j_tab])
        app.load(journal, None, [j_etat, j_tab])

    with gr.Tab("4 · À propos et limites"):
        gr.Markdown(a_propos())

if __name__ == "__main__":
    # en local : 127.0.0.1 seulement ; sur Hugging Face Spaces, GRADIO_SERVER_NAME vaut 0.0.0.0
    app.queue(default_concurrency_limit=1).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                                                  server_port=int(os.environ.get("PORT", 7860)),
                                                  css=CSS, theme=gr.themes.Soft(primary_hue="blue"),
                                                  inbrowser=os.environ.get("OUVRIR", "0") == "1", show_error=True)

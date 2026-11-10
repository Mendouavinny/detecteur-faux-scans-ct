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
TAU1_DEF, TAU2_DEF = 0.50, 0.85     # seuil de décision du mémoire ; 95e centile des coupes authentiques de validation

print("Chargement des modèles…", flush=True)
DET = D.Detecteur()
print("Modèles chargés :", DET.dossier, "| fusion :", DET.fusion_choix, flush=True)

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
def html_verdict(v, score, tau1, tau2, hash_ok, n_patchs, duree):
    fg, bg = COULEURS[v]
    raison = ("empreinte SHA-256 différente de l’empreinte enregistrée" if not hash_ok else
              f"score {score:.3f} {'>' if v != 'CONFORME' else '≤'} τ{'2' if v == 'ALERTE' else '1'} = "
              f"{tau2 if v == 'ALERTE' else tau1:.2f}")
    return (f"<div style='border:2px solid {fg};background:{bg};border-radius:12px;padding:16px 20px'>"
            f"<div style='font-size:13px;color:{fg};letter-spacing:.08em;text-transform:uppercase'>Verdict</div>"
            f"<div style='font-size:34px;font-weight:700;color:{fg};line-height:1.2'>{LIBELLE[v]}</div>"
            f"<div style='font-size:15px;color:#333;margin-top:6px'>Score de falsification (fusion robuste, "
            f"0 = authentique, 1 = falsifiée) : "
            f"<b>{score:.3f}</b> — {raison}</div>"
            f"<div style='font-size:13px;color:#555;margin-top:4px'>{n_patchs} patchs pulmonaires analysés en "
            f"{duree:.1f} s</div></div>")


def html_branches(r):
    b = r["branches"]
    lignes = [("Fusion robuste (retenue)", r["score"], True, "probabilité de falsification"),
              ("Branche forensique", b["forensique"], False, "moyenne des 3 patchs les plus suspects"),
              ("Branche CNN contrainte", b["cnn"], False, "moyenne des 3 patchs les plus suspects"),
              ("Fusion locale des cartes", b["fusion_locale"], False, "moyenne des 3 patchs les plus suspects"),
              ("Méta-classifieur HGB (comparaison)", r["score_hgb"], False, "version initiale, non retenue")]
    h = "<div style='font-family:inherit'>"
    for nom, v, fort, aide in lignes:
        coul = "#eb6834" if fort else "#2a78d6"
        h += (f"<div style='margin:8px 0'><div style='display:flex;justify-content:space-between;font-size:14px'>"
              f"<span style='font-weight:{700 if fort else 500}'>{nom}</span><span><b>{v:.3f}</b></span></div>"
              f"<div style='background:#e9e9e4;border-radius:6px;height:10px;margin-top:3px'>"
              f"<div style='width:{max(0, min(1, v)) * 100:.1f}%;background:{coul};height:10px;border-radius:6px'></div>"
              f"</div><div style='font-size:12px;color:#777'>{aide}</div></div>")
    h += (f"<div style='margin-top:10px;font-size:13px;color:#555'>Branche diffusion (écart normalisé, poids nul dans "
          f"la fusion) : {b['diffusion_z']:+.2f} · Duplication exacte maximale par patch : {r['duplication']:.3f}</div>")
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


def analyser_dicom(fichier, exemple, empreinte_ref, tau1, tau2, utilisateur):
    ds, img, nom = lire_dicom(fichier, exemple)
    t = time.time()
    r = DET.analyser(img)
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
                            tau1=round(tau1, 3), tau2=round(tau2, 3)))
    carte = croix_max(superposer(img, r["carte"]), r["carte"])
    return (html_verdict(v, r["score"], tau1, tau2, hash_ok, r["n_patchs"], duree), html_branches(r),
            contour(fenetre(img), r["lung"], (90, 160, 255)), carte, superposer(img, r["carte_cnn"]), infos,
            json.dumps(rec, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ onglet 2 : démonstration d'attaque
def demo(exemple, attaque_lib, graine, tau1, tau2):
    if not exemple:
        raise gr.Error("Choisissez une coupe authentique.")
    nom = next(k for k, v in D.ATTAQUES.items() if v == attaque_lib)
    meta = next(e for e in INDEX if e["fichier"] == exemple)
    img = D.dicom_to_hu(pydicom.dcmread(EX / exemple))
    try:
        fausse, masque = D.appliquer_attaque(DET, nom, img, meta["nodules"], int(graine))
    except ValueError as e:
        raise gr.Error(str(e))
    r0, r1 = DET.analyser(img), DET.analyser(fausse)
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
                f"<b>{rr['score']:.3f}</b></td><td style='padding:6px 10px;text-align:center'>{rr['score_hgb']:.3f}</td>"
                f"<td style='padding:6px 10px;text-align:center'><span style='background:{bg};color:{fg};"
                f"padding:3px 10px;border-radius:10px;font-weight:600'>{LIBELLE[vv]}</span></td></tr>")
    th = "style='padding:8px 10px;background:#14305a;color:#ffffff !important;font-weight:600;text-align:{}'"
    res = ("<table style='width:100%;border-collapse:collapse;font-size:15px'><tr>"
           f"<th {th.format('left')}>Coupe</th><th {th.format('center')}>Score de falsification (fusion robuste)</th>"
           f"<th {th.format('center')}>HGB (comparaison)</th><th {th.format('center')}>Verdict</th></tr>"
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
            + (f" · score {s:.3f}".replace(".", ",") + (" (fausse alerte à 0,5)" if s > 0.5 else "") if s is not None else ""))


EX_CHOIX = [(etiquette(e), e["fichier"]) for e in INDEX]
EX_NOD = [e for e in INDEX if e["nodules"]]
# par défaut : coupes représentatives (bien classées) ; les fausses alertes restent visibles dans la liste
DEF_ANALYSE = min(INDEX, key=lambda e: e.get("score_authentique", 1))["fichier"] if INDEX else None
DEF_DEMO = min(EX_NOD, key=lambda e: e.get("score_authentique", 1))["fichier"] if EX_NOD else None
_R = [p / "resultats_lidc.json" for p in (D.MODELES, Path.home() / "Documents" / "GPU_ROCm" / "out",
                                          ICI.parent / "3_Resultats" / "scores_et_tableaux")]
RES = next((json.load(open(p, encoding="utf-8")) for p in _R if p.exists()), None)


def fr(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


def a_propos():
    t = """
### Ce que fait l’application
Elle applique à une coupe CT le **détecteur hybride** du mémoire, avec les modèles entraînés lors de la quatrième
exécution du carnet, celle qui corrige l’artefact d’écrêtage (aucun réentraînement) :

1. **Prétraitement** : conversion en unités Hounsfield, segmentation des poumons, découpage en patchs de 64 × 64 pixels (pas de 32).
2. **Branche forensique** : 27 descripteurs du bruit par patch (résidus médian et KV, autocorrélation, spectre, incohérence locale, duplication exacte), classés par gradient boosting.
3. **Branche CNN** : couche contrainte de Bayar et Stamm + filtres SRM, encodeur EfficientNet-B0, décodeur U-Net (carte de segmentation).
4. **Branche diffusion** : erreur de reconstruction d’un DDPM appris sur des patchs authentiques (poids nul dans la fusion retenue).
5. **Fusion robuste** : régression logistique à poids positifs sur les scores des branches, choisie sur la seule validation.
6. **Module de sécurité** : empreinte SHA-256, politique à trois niveaux (conforme / à vérifier / alerte) et journal d’audit chaîné.

### Performances mesurées (mémoire, tableau 3.6)
"""
    if RES:
        n = {"forensique": "Forensique", "cnn": "CNN contraint", "diffusion": "Diffusion",
             "hybride": "**Fusion robuste (retenue)**", "hybride_hgb": "Méta-classifieur HGB (comparaison)"}
        t += "| Branche | AUC test interne (153 coupes) | AUC CT-GAN externe (37 coupes) |\n|---|---|---|\n"
        for k, lab in n.items():
            if k in RES["RES"]:
                ext = RES.get("EXT", {}).get(k, {}).get("auc")
                t += f"| {lab} | {RES['RES'][k]['auc']:.3f} | {ext:.3f} |\n".replace(".", ",") if ext else ""
    t += f"""
### À savoir avant toute utilisation
- **Outil de recherche, pas de diagnostic.** Il s’agit d’une aide à la vigilance ; aucune validation clinique n’a été faite.
- **Seuils non calibrés cliniquement.** Par défaut, τ1 = {fr(TAU1_DEF)} (seuil de décision du mémoire) et
  τ2 = {fr(TAU2_DEF)} (95ᵉ centile des coupes authentiques de validation). Au seuil de 0,5, le détecteur signale environ
  78 % des falsifications du test interne, mais aussi plus de deux coupes authentiques sur cinq.
- **Pas de gain de la fusion sur la branche forensique.** Sur le test interne comme sur CT-GAN, aucun écart entre la
  fusion robuste et la branche forensique seule n’est significatif.
- **Opérer sur le DICOM d’origine.** Un rééchantillonnage efface l’indice de bruit (AUC ramenée à 0,494).
- **Localisation grossière** : la carte est construite sur des patchs de 64 × 64 pixels.
- **Artefact d’écrêtage corrigé.** Dans les premières exécutions du carnet, les attaques écrêtaient la coupe à
  [−1024, 3071] HU et modifiaient ainsi le remplissage hors du champ de vue sur les seules coupes falsifiées ; le
  méta-classifieur HGB exploitait cet indice. Le carnet restitue désormais ces pixels, comme la démonstration de
  l’onglet 2 : avec les modèles actuels, un écrêtage seul ne change plus le score de la fusion robuste et ne déplace
  celui du HGB que de 0,03 en moyenne.
- Fidélité vérifiée : sur les 76 coupes authentiques du test interne, l’application reproduit les scores du carnet
  (écart moyen 0,0002).
- Tout s’exécute sur ce poste : l’application n’écoute que l’adresse locale 127.0.0.1 et n’envoie aucune image.

*Mémoire de Master Recherche SDIA (ENSPD) — MENDOUA Vinny Jordan.*
"""
    return t


with gr.Blocks(title="Détection de faux scans CT") as app:
    gr.HTML("<div id='entete'><h1>Détection automatique de faux scans CT</h1>"
            "<p>Approche hybride forensique + apprentissage profond — démonstrateur du mémoire de Master "
            "Recherche SDIA (ENSPD), MENDOUA Vinny Jordan</p></div>")
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
                                    info="Le score affiché est celui du détecteur sur la coupe authentique : au seuil "
                                         "de 0,5, plus de deux coupes authentiques sur cinq sont signalées (mémoire, §3.1.7).")
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
        go.click(analyser_dicom, [f_in, ex_in, ref_in, tau1, tau2, user], [v_out, b_out, i0, i1, i2, inf, rec])

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
            d_go = gr.Button("Falsifier puis analyser", variant="primary")
        d_res = gr.HTML()
        with gr.Row():
            d0 = gr.Image(label="Originale", type="numpy", height=360)
            d1 = gr.Image(label="Falsifiée (contour vert : région modifiée)", type="numpy", height=360)
            d2 = gr.Image(label="Carte de détection sur la coupe falsifiée", type="numpy", height=360)
            d3 = gr.Image(label="Différence |falsifiée − originale|", type="numpy", height=360)
        d_go.click(demo, [d_ex, d_atk, d_seed, tau1, tau2], [d0, d1, d2, d3, d_res])

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
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                                                  css=CSS, theme=gr.themes.Soft(primary_hue="blue"),
                                                  inbrowser=os.environ.get("OUVRIR", "0") == "1", show_error=True)

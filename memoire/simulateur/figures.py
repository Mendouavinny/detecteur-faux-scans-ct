import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import ndimage as ndi
import features as F

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
OUT = "figures/"
import os; os.makedirs(OUT, exist_ok=True)
CL = ["Authentique", "Injection", "Suppression", "Copier-déplacer", "Bruit", "Génératif"]
d = np.load("simct_tamper.npz")
X, Y, P = d["X"], d["Y"], d["P"]
M = np.unpackbits(d["M"], axis=-1)[:, :, :256].astype(bool)
L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)
R = json.load(open("results.json")); H = json.load(open("results_hybrid.json"))


def win(img, lo=-1350, hi=150):
    return np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)


def grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


# Fig 1 : exemples de falsifications
fig, axs = plt.subplots(2, 6, figsize=(13, 4.6))
for c in range(6):
    i = np.where(Y == c)[0][3]
    axs[0, c].imshow(win(X[i]), cmap="gray"); axs[0, c].set_title(CL[c], fontsize=10)
    axs[1, c].imshow(win(X[i]), cmap="gray")
    if M[i].any():
        mm = ndi.binary_fill_holes(ndi.binary_closing(M[i], iterations=3)) if c == 4 else M[i]
        axs[1, c].contour(mm, levels=[0.5], colors=[C2], linewidths=1.2)
    for a in axs[:, c]:
        a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel("Coupe"); axs[1, 0].set_ylabel("Vérité terrain")
plt.tight_layout(); plt.savefig(OUT + "fig_exemples.png", dpi=200); plt.close()

# Fig 2 : texture du bruit (résidu KV) authentique vs injection
i = np.where(Y == 1)[0][3]
ys, xs = np.where(M[i]); cy, cx = int(ys.mean()), int(xs.mean())
res = F.residuals(X[i])["kv"]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
axs[0].imshow(win(X[i][cy - 40:cy + 40, cx - 40:cx + 40]), cmap="gray"); axs[0].set_title("Zone injectée (fenêtre poumon)")
axs[1].imshow(np.clip(res[cy - 40:cy + 40, cx - 40:cx + 40], -60, 60), cmap="gray"); axs[1].set_title("Résidu haute fréquence (filtre KV)")
axs[1].contour(M[i][cy - 40:cy + 40, cx - 40:cx + 40], levels=[0.5], colors=[C2], linewidths=1)
# spectre radial résidu : bloc authentique vs bloc falsifié
def radial(r):
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(r - r.mean()))) ** 2
    h = r.shape[0]; yy, xx = np.mgrid[:h, :h]; rad = np.sqrt((yy - h / 2) ** 2 + (xx - h / 2) ** 2).astype(int)
    return np.bincount(rad.ravel(), Fm.ravel()) / np.bincount(rad.ravel())
s_t = []; s_a = []
for j in np.where(Y == 1)[0][:60]:
    ys, xs = np.where(M[j]); y0, x0 = int(ys.mean()), int(xs.mean())
    r = F.residuals(X[j])["med"]
    s_t.append(radial(r[y0 - 8:y0 + 8, x0 - 8:x0 + 8])[:9])
    bl = F.lung_blocks(L[j])
    yb_, xb_ = bl[len(bl) // 3]
    s_a.append(radial(r[yb_:yb_ + 16, xb_:xb_ + 16])[:9])
f = np.arange(9) / 8
axs[2].plot(f, np.mean(s_a, 0) / np.mean(s_a, 0).sum(), color=C1, lw=2, marker="o", ms=5, label="Bloc authentique")
axs[2].plot(f, np.mean(s_t, 0) / np.mean(s_t, 0).sum(), color=C2, lw=2, marker="s", ms=5, label="Bloc falsifié")
axs[2].set_xlabel("Fréquence radiale normalisée"); axs[2].set_ylabel("Énergie relative")
axs[2].set_title("Spectre moyen du résidu (médian)"); axs[2].legend(frameon=False, fontsize=8); grid(axs[2])
for a in axs[:2]:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT + "fig_bruit.png", dpi=200); plt.close()

# Fig 3 : ROC
r1 = np.load("roc.npz"); r2 = np.load("roc_hybrid.npz")
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(r1["fb"], r1["tb"], color=C3, lw=2, label=f"Intensité seule (AUC = {R['E1']['Ligne de base intensité (RF)']['mean'][4]:.3f})")
ax.plot(r1["fpr"], r1["tpr"], color=C1, lw=2, label=f"Forensique globale (AUC = {R['E1'][R['best']]['mean'][4]:.3f})")
ax.plot(r2["fpr"], r2["tpr"], color=C2, lw=2, label=f"Hybride local→global (AUC = {H['hybrid']['mean'][4]:.3f})")
ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT + "fig_roc.png", dpi=200); plt.close()

# Fig 4 : détection par type d'attaque
pc_g = [R["per_class_detection"][k] for k in ["authentique", "injection", "suppression", "copier-déplacer", "bruit", "génératif"]]
pc_h = [H["hybrid_per_class"][k] for k in ["authentique", "injection", "suppression", "copier-déplacer", "bruit", "génératif"]]
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(6); w = 0.38
b1 = ax.bar(x - w / 2 - 0.01, pc_g, w, color=C1, label="Forensique globale")
b2 = ax.bar(x + w / 2 + 0.01, pc_h, w, color=C2, label="Hybride local→global")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.2f}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(CL); ax.set_ylim(0, 1.1); ax.set_ylabel("Taux de décision correcte")
ax.legend(frameon=False, ncol=2, loc="upper left"); grid(ax)
plt.tight_layout(); plt.savefig(OUT + "fig_par_classe.png", dpi=200); plt.close()

# Fig 5 : matrice de confusion
cm = np.array(R["E2"]["cm"]); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for a in range(6):
    for b in range(6):
        ax.text(b, a, f"{cm[a, b]}", ha="center", va="center", fontsize=8, color="white" if cmn[a, b] > 0.55 else INK)
ax.set_xticks(range(6)); ax.set_xticklabels(CL, rotation=35, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(CL)
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle"); ax.spines[:].set_visible(False)
plt.colorbar(im, fraction=0.046, label="Proportion par ligne")
plt.tight_layout(); plt.savefig(OUT + "fig_confusion.png", dpi=200); plt.close()

# Fig 6 : ablation
ab = R["ablation"]; names = list(ab.keys()); vals = [ab[k]["mean"][4] for k in names]; sd = [ab[k]["std"][4] for k in names]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
yy = np.arange(len(names))
ax.barh(yy, vals, xerr=sd, color=C1, height=0.55, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for k, v in enumerate(vals):
    ax.text(v + 0.03, k, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(0.4, 0.85); ax.set_xlabel("AUC (validation croisée groupée, 5 plis)")
ax.axvline(0.5, color=INK2, lw=1, ls="--"); ax.invert_yaxis(); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT + "fig_ablation.png", dpi=200); plt.close()

# Fig 7 : robustesse + généralisation
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
e4 = R["E4"]; k4 = list(e4.keys())
axs[0].bar(range(len(k4)), [e4[k] for k in k4], color=C1, width=0.6)
for j, k in enumerate(k4):
    axs[0].text(j, e4[k] + 0.01, f"{e4[k]:.3f}", ha="center", fontsize=8, color=INK2)
axs[0].set_xticks(range(len(k4))); axs[0].set_xticklabels(["Aucun", "Flou\ngaussien\nσ=0,5", "Redim.\n×0,75", "Bruit\nadditif\nσ=2 HU", "Export\n8 bits"], fontsize=8)
axs[0].set_ylim(0.4, 0.85); axs[0].axhline(0.5, color=INK2, ls="--", lw=1); axs[0].set_ylabel("AUC"); axs[0].set_title("(a) Robustesse aux post-traitements"); grid(axs[0])
e3 = R["E3"]; k3 = list(e3.keys())
axs[1].bar(range(len(k3)), [e3[k][0] for k in k3], yerr=[e3[k][1] for k in k3], color=C3, width=0.6, error_kw=dict(ecolor=INK2, capsize=3))
for j, k in enumerate(k3):
    axs[1].text(j + 0.18, e3[k][0] + 0.01, f"{e3[k][0]:.2f}", fontsize=8, color=INK2)
axs[1].set_xticks(range(len(k3))); axs[1].set_xticklabels([k.capitalize() for k in k3], fontsize=8)
axs[1].set_ylim(0.3, 0.85); axs[1].axhline(0.5, color=INK2, ls="--", lw=1); axs[1].set_title("(b) Attaque non vue à l'entraînement"); grid(axs[1])
plt.tight_layout(); plt.savefig(OUT + "fig_robustesse.png", dpi=200); plt.close()

# Fig 8 : cartes de localisation
mp = np.load("maps_hybrid.npz")
fig, axs = plt.subplots(3, 4, figsize=(11, 8.4))
for k, c in enumerate((1, 2, 3, 5)):
    i = int(mp["idx"][k]); pm = mp[f"m{c}"]
    axs[0, k].imshow(win(X[i]), cmap="gray"); axs[0, k].set_title(CL[c])
    axs[1, k].imshow(win(X[i]), cmap="gray"); axs[1, k].contour(M[i], levels=[0.5], colors=[C2], linewidths=1.2)
    axs[2, k].imshow(win(X[i]), cmap="gray"); h = axs[2, k].imshow(np.ma.masked_less(pm, 0.05), cmap="magma", vmin=0, vmax=1, alpha=0.7)
    axs[2, k].contour(M[i], levels=[0.5], colors=["#1baf7a"], linewidths=1)
    for a in axs[:, k]:
        a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel("Coupe falsifiée"); axs[1, 0].set_ylabel("Vérité terrain"); axs[2, 0].set_ylabel("Carte de probabilité")
cax = fig.add_axes([0.92, 0.12, 0.012, 0.22]); fig.colorbar(h, cax=cax, label="P(falsifié)")
plt.savefig(OUT + "fig_localisation.png", dpi=200, bbox_inches="tight"); plt.close()


# Fig 9 : architecture du système proposé
def box(ax, x, y, w, h, txt, fc="#eef4fc", ec=C1, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))


fig, ax = plt.subplots(figsize=(12, 5.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
box(ax, 0.1, 2.5, 1.6, 1.0, "PACS / modalité\nDICOM entrant", "#f4f3f0", INK2)
box(ax, 2.0, 2.4, 1.9, 1.2, "Pré-traitement\nHU, fenêtrage,\nsegmentation\npulmonaire")
box(ax, 4.2, 4.2, 2.5, 1.1, "Branche forensique\nrésidus SRM/KV, spectre,\nincohérence locale,\nduplication de blocs", "#fdf0ea", C2)
box(ax, 4.3, 2.5, 2.2, 1.1, "Branche CNN\ncouche contrainte (Bayar)\n+ EfficientNet / U-Net", "#eef4fc", C1)
box(ax, 4.3, 0.8, 2.2, 1.1, "Branche diffusion\nerreur de reconstruction\n(type DIRE / BTD)", "#eaf7f1", C3)
box(ax, 7.0, 2.5, 1.7, 1.0, "Fusion\nlocal → global\n(agrégation top-k)")
box(ax, 9.2, 3.6, 2.6, 1.0, "Décision : authentique /\nfalsifiée + type d'attaque")
box(ax, 9.2, 2.2, 2.6, 1.0, "Carte de localisation\n(probabilité par région)")
box(ax, 9.2, 0.6, 2.6, 1.1, "Module sécurité : empreinte\nSHA-256, journal d'audit,\nalerte radiologue / RSSI", "#f4f3f0", INK2)
arr(ax, (1.7, 3.0), (2.1, 3.0))
for yy in (4.75, 3.05, 1.35):
    arr(ax, (3.8, 3.0), (4.3, yy)); arr(ax, (6.5, yy), (7.0, 3.0))
arr(ax, (8.7, 3.1), (9.2, 4.1)); arr(ax, (8.7, 3.0), (9.2, 2.7)); arr(ax, (10.5, 2.2), (10.5, 1.7))
plt.savefig(OUT + "fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 10 : chaîne de simulation d'acquisition + menace
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 2.4); ax.axis("off")
steps = ["Fantôme thoracique\n(carte HU)", "Atténuation μ\n+ projection Radon", "Bruit de Poisson\n(dose I₀ variable)",
         "Reconstruction FBP\n(4 noyaux)", "Falsification\n(5 attaques)", "Jeu SimCT-Tamper\n1 080 coupes"]
for k, s in enumerate(steps):
    box(ax, 0.1 + k * 2.0, 0.7, 1.7, 1.0, s, "#fdf0ea" if k == 4 else "#eef4fc", C2 if k == 4 else C1)
    if k:
        arr(ax, (k * 2.0 - 0.2, 1.2), (0.1 + k * 2.0, 1.2))
plt.savefig(OUT + "fig_simulation.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 11 : modèle de menace
fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
box(ax, 0.1, 1.2, 1.8, 1.0, "Scanner CT\n(modalité)", "#f4f3f0", INK2)
box(ax, 2.6, 1.2, 1.9, 1.0, "Réseau hospitalier\n(DICOM C-STORE)", "#f4f3f0", INK2)
box(ax, 5.2, 1.2, 1.8, 1.0, "Serveur PACS", "#f4f3f0", INK2)
box(ax, 7.7, 1.2, 1.9, 1.0, "Station de lecture\nradiologue", "#f4f3f0", INK2)
box(ax, 10.2, 1.2, 1.7, 1.0, "Diagnostic /\nassurance / justice", "#f4f3f0", INK2)
for a_, b_ in [((1.9, 1.7), (2.6, 1.7)), ((4.5, 1.7), (5.2, 1.7)), ((7.0, 1.7), (7.7, 1.7)), ((9.6, 1.7), (10.2, 1.7))]:
    arr(ax, a_, b_)
box(ax, 2.7, 2.55, 4.2, 0.75, "Attaquant : interception (MitM) ou accès au PACS\n→ injection / suppression / bruit / contenu génératif", "#fdf0ea", C2)
arr(ax, (3.6, 2.55), (3.55, 2.2)); arr(ax, (6.0, 2.55), (6.1, 2.2))
box(ax, 5.0, 0.05, 4.8, 0.75, "Point de contrôle proposé : détecteur forensique + empreinte d'intégrité", "#eaf7f1", C3)
arr(ax, (6.1, 0.8), (6.1, 1.2)); arr(ax, (8.6, 0.8), (8.65, 1.2))
plt.savefig(OUT + "fig_menace.png", dpi=200, bbox_inches="tight"); plt.close()
print("ok")

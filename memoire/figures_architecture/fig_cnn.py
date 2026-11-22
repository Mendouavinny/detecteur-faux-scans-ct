from common import *

W, H = 6.3, 3.98
fig, ax = canvas(W, H)
img, msk, lung = ct_images()
ys, xs = np.where(msk)
cy, cx = int(ys.mean()), int(xs.mean())
patch = img[cy - 16:cy + 16, cx - 16:cx + 16]
mpatch = msk[cy - 16:cy + 16, cx - 16:cx + 16].astype(float)

S = {64: 0.92, 32: 0.70, 16: 0.52, 8: 0.39, 4: 0.30, 2: 0.23}
T = lambda c: 0.032 * np.log2(c) + 0.02
ye, yd = 2.86, 1.45

# --- entrée et banque de filtres ---
thumb(ax, patch, 0.05, ye - 0.27, 0.54, 0.54)
text(ax, 0.32, ye - 0.33, "Patch 1×64×64\nHU / 1000", fs=5.7, va="top", color=SOFT)
b_id = box(ax, 0.74, ye + 0.46, 0.94, 0.22, "Coupe (identité)", kind="data", fs=5.8)
b_by = box(ax, 0.74, ye - 0.10, 0.94, 0.42, "Bayar et Stamm", "3 noyaux 5×5 appris\n$w_0=-1$, $\\Sigma w_k=1$", kind="reuse", fs=5.6)
b_sr = box(ax, 0.74, ye - 0.66, 0.94, 0.42, "SRM fixes", "KV 5×5, SRM 3×3,\ndifférence $\\Delta_x$", kind="reuse", fs=5.6)
for b in (b_id, b_by, b_sr):
    arrow(ax, (0.59, ye), (b[0], b[1] + b[3] / 2), ms=5)
s7 = slab(ax, 2.08, ye, S[64], T(7), kind="own", label="7×64×64")
text(ax, 2.16, ye + 0.70, "Concaténation", fs=5.6, color="#C45A1E", weight="bold")
for b, n in ((b_id, "1"), (b_by, "3"), (b_sr, "3")):
    p = (b[0] + b[2], b[1] + b[3] / 2)
    arrow(ax, p, (s7["l"][0], p[1] * 0.35 + ye * 0.65), ms=5)
    text(ax, p[0] + 0.05, p[1] - 0.06, n, fs=5.4, color=SOFT)

# --- encodeur EfficientNet-B0 ---
enc = [(16, 32, 2.72), (24, 16, 3.25), (40, 8, 3.70), (112, 4, 4.11), (320, 2, 4.50)]
E = []
prev = s7
for c, s, x in enc:
    e = slab(ax, x, ye, S[s], T(c), kind="reuse", label=f"{c}×{s}×{s}")
    arrow(ax, prev["r"], (e["l"][0], ye), ms=5)
    E.append(e); prev = e
bracket(ax, 2.35, 4.70, 3.70, "Encodeur EfficientNet-B0 (Tan et Le), préentraîné sur ImageNet, 7 canaux d'entrée", fs=5.7)
text(ax, 3.52, 3.58, "sorties des étages 1, 2, 3, 5 et 7 (pas 2 à 32)", fs=5.2, color=SOFT)

# --- tête de classification ---
hb = box(ax, 4.95, ye + 0.08, 1.33, 0.52, "Tête de classification", "moyenne globale (320)\nDropout 0,3 · Linéaire 320→1", kind="own", fs=5.6)
arrow(ax, E[-1]["r"], (hb[0], E[-1]["r"][1] + 0.05), ms=5)
ob = box(ax, 5.17, ye - 0.42, 0.83, 0.28, "$p_{cnn}=\\sigma(\\ell)$", "score du patch", kind="data", fs=5.6)
arrow(ax, (hb[0] + hb[2] / 2, hb[1]), (ob[0] + ob[2] / 2, ob[1] + ob[3]), ms=5)

# --- décodeur de type U-Net ---
dec = [(112, 4, 4.11), (40, 8, 3.70), (32, 16, 3.25), (32, 32, 2.72)]
D = []
for c, s, x in dec:
    D.append(slab(ax, x, yd, S[s], T(c), kind="reuse", label=f"{c}×{s}×{s}"))
poly_arrow(ax, [E[-1]["b"], (E[-1]["b"][0], yd), (D[0]["r"][0], yd)], ms=5)
for a, b in zip(D[:-1], D[1:]):
    arrow(ax, (a["l"][0], yd), (b["r"][0], yd), ms=5)
for e, d in zip(E[:4][::-1], D):
    arrow(ax, (e["b"][0], e["b"][1] - 0.16), (d["t"][0] - 0.03, d["t"][1] + 0.02), ls=(0, (2.2, 1.6)), color="#3A6EA5", ms=4.5, lw=0.65)
text(ax, 3.86, 2.08, "sauts", fs=5.3, color="#3A6EA5", ha="left")
bracket(ax, 2.35, 4.70, 0.80, "Décodeur de type U-Net (Ronneberger et al.)", fs=5.7, up=False)

sg = box(ax, 1.25, yd - 0.24, 0.92, 0.48, "Sortie", "Conv 1×1 → 1×32×32\n×2 bilinéaire, $\\sigma$", kind="own", fs=5.5)
arrow(ax, D[-1]["l"], (sg[0] + sg[2], yd), ms=5)
thumb(ax, mpatch, 0.18, yd - 0.25, 0.50, 0.50, vmin=0, vmax=1)
arrow(ax, (sg[0], yd), (0.68, yd), ms=5)
text(ax, 0.43, yd - 0.31, "Carte $\\hat m$ 1×64×64", fs=5.6, va="top", color=SOFT)

# --- définition du bloc Up, perte, apprentissage ---
box(ax, 4.85, 1.12, 1.40, 0.62, "Bloc Up (flèches du décodeur)", "×2 bilinéaire, concaténation\ndu saut, 2 × (Conv 3×3,\nBN, ReLU)", kind="reuse", fs=5.4)
box(ax, 4.85, 1.86, 1.40, 0.42, "Perte multitâche", "$\\mathrm{BCE}(p_{cnn},y)+1-\\mathrm{Dice}(\\hat m,m)$", kind="own", fs=5.5)
for k, line in enumerate(["Apprentissage : Adam $3\\cdot10^{-4}$, lots de 128, 30 époques au plus, arrêt après 6 époques sans gain d'AUC de validation ; négatifs 2 : 1",
                         "Augmentation : symétries D4, puis flou, rééchantillonnage, bruit ou export 8 bits avec une probabilité de 0,4",
                         "Paramètres appris : encodeur 3 596 540 · décodeur 669 489 · tête 321 · Bayar 75 · total 4 266 425 (filtres SRM fixes)"]):
    text(ax, 0.05, 0.50 - k * 0.12, line, fs=5.3, ha="left", color=SOFT)
legend(ax, 0.05, 0.03)
save(fig, "figures/arch_cnn.png")

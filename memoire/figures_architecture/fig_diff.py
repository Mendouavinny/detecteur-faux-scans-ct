from common import *

W, H = 6.3, 3.62
fig, ax = canvas(W, H)
img, msk, lung = ct_images()
ys, xs = np.where(msk)
cy, cx = int(ys.mean()), int(xs.mean())
patch = img[cy - 16:cy + 16, cx - 16:cx + 16]

S = {64: 0.66, 32: 0.50, 16: 0.38, 8: 0.29}
T = lambda c: 0.030 * np.log2(c) + 0.02
yr = 2.42

thumb(ax, patch, 0.12, yr - 0.25, 0.50, 0.50)
text(ax, 0.37, yr - 0.31, "Patch $x$ 1×64×64\n[−1100 ; 300] HU\n→ [−1 ; 1]", fs=5.2, va="top", color=SOFT)
nb = box(ax, 0.78, yr - 0.30, 1.12, 0.60, "Bruitage direct", "$x_{t^*}=\\sqrt{\\bar\\alpha_{t^*}}\\,x+\\sqrt{1-\\bar\\alpha_{t^*}}\\,\\epsilon$\n$t^*=150$, $\\epsilon\\sim\\mathcal{N}(0,I)$", kind="own", fs=5.5)
arrow(ax, (0.62, yr), (nb[0], yr), ms=5)

lv = [(64, 64, 2.25, "d1"), (128, 32, 2.72, "d2"), (256, 16, 3.14, "d3"), (256, 8, 3.52, "mid"),
      (128, 16, 3.90, "u3"), (64, 32, 4.32, "u2"), (64, 64, 4.80, "u1")]
L = []
prev_r = (nb[0] + nb[2], yr)
for c, s, x, nm in lv:
    e = slab(ax, x, yr, S[s], T(c), kind="reuse", label=f"{c}×{s}×{s}", fs=5.0, depth=0.26, lab_dy=0.05 + (0.13 if nm in ("d2", "u2") else 0))
    arrow(ax, prev_r, (e["l"][0], yr), ms=4.5)
    L.append(e); prev_r = e["r"]
for i, j in ((0, 6), (1, 5), (2, 4)):
    a, b = L[i], L[j]
    h = 0.20 + 0.10 * (2 - i)
    xa, xb = a["t"][0], b["t"][0]
    ya = max(a["t"][1], b["t"][1]) + 0.04
    ax.plot([xa, xa, xb], [a["t"][1] + 0.02, ya + h, ya + h], color="#3A6EA5", lw=0.65, ls=(0, (2.2, 1.6)), zorder=2)
    arrow(ax, (xb, ya + h), (xb, b["t"][1] + 0.02), color="#3A6EA5", ls=(0, (2.2, 1.6)), ms=4.5, lw=0.65)
text(ax, 4.86, 3.02, "sauts\n(concaténation)", fs=5.2, color="#3A6EA5", ha="left")
eo = box(ax, 5.25, yr - 0.20, 1.0, 0.40, "$\\hat\\epsilon=\\epsilon_\\theta(x_{t^*},t^*)$", "1×64×64", kind="data", fs=5.6)
arrow(ax, L[-1]["r"], (eo[0], yr), ms=4.5)
bracket(ax, 2.0, 5.05, 3.40, "Débruiteur U-Net du DDPM (Ho et al.), 4 352 385 paramètres", fs=5.7)

# plongement du temps et bloc résiduel
tb = box(ax, 0.78, 1.16, 1.12, 0.52, "Temps $t^*$", "sinusoïdal 128\n→ MLP (SiLU) → 512", kind="reuse", fs=5.4)
ax.plot([tb[0] + tb[2], 4.80], [1.42, 1.42], color=SOFT, lw=0.6, ls=(0, (1, 1.2)), zorder=1)
for e in L:
    ax.plot([e["cx"], e["cx"]], [1.42, e["b"][1] - 0.22], color=SOFT, lw=0.6, ls=(0, (1, 1.2)), zorder=1)
rb = box(ax, 2.05, 0.93, 2.25, 0.40, "ResB (chaque volume)", "GN, SiLU, Conv 3×3, + projection de $t$ ; GN, SiLU, Conv 3×3 ; saut 1×1\ndescente : moyenne 2×2 · montée : plus proche ×2", kind="reuse", fs=4.9)

# score
x0b = box(ax, 4.45, 1.20, 1.80, 0.44, "Estimation en un pas", "$\\hat x_0=(x_{t^*}-\\sqrt{1-\\bar\\alpha_{t^*}}\\,\\hat\\epsilon)/\\sqrt{\\bar\\alpha_{t^*}}$", kind="own", fs=5.4)
arrow(ax, (eo[0] + eo[2] / 2, eo[1]), (5.62, x0b[1] + x0b[3]), ms=4.5)
sb = box(ax, 4.45, 0.62, 1.80, 0.44, "Score d'anomalie (principe de DIRE)", "$s=\\overline{(\\hat x_0-x)^2}$, moyenne sur 3 tirages de $\\epsilon$", kind="reuse", fs=5.2)
arrow(ax, (5.35, x0b[1]), (5.35, sb[1] + sb[3]), ms=4.5)
zb = box(ax, 2.05, 0.40, 2.25, 0.44, "Normalisation dans la coupe", "$z=(s-\\mathrm{méd})/(1{,}4826\\cdot\\mathrm{MAD})$ → $z$, $|z|$", kind="own", fs=5.2)
arrow(ax, (sb[0], sb[1] + sb[3] / 2), (zb[0] + zb[2], zb[1] + zb[3] / 2), ms=4.5)
ob = box(ax, 0.78, 0.44, 1.12, 0.36, "$z$, $|z|$ par patch", "vers la fusion", kind="data", fs=5.4)
arrow(ax, (zb[0], zb[1] + zb[3] / 2), (ob[0] + ob[2], zb[1] + zb[3] / 2), ms=4.5)

for k, line in enumerate(["Apprentissage : DDPM à T = 1 000 pas, $\\beta_t$ linéaire de $10^{-4}$ à 0,02, perte $\\Vert\\epsilon-\\epsilon_\\theta(x_t,t)\\Vert^2$, AdamW $2\\cdot10^{-4}$, lots de 64, 40 époques, symétries D4",
                         "Données : patchs pulmonaires authentiques d'une moitié des patients d'entraînement ; l'autre moitié sert au modèle de l'attaquant"]):
    text(ax, 0.05, 0.30 - k * 0.11, line, fs=5.1, ha="left", color=SOFT)
legend(ax, 0.05, 0.01)
save(fig, "figures/arch_diffusion.png")

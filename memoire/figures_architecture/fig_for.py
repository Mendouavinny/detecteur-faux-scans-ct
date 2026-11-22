from common import *
import sys
sys.path.insert(0, "../simulateur")
import features as FZ

W, H = 6.3, 3.28
fig, ax = canvas(W, H)
img, msk, lung = ct_images()

# coupe et grille de patchs (illustration : patchs 32×32 sur une coupe 256×256, comme 64×64 sur 512×512)
x0, y0, sz = 0.05, 1.78, 1.0
thumb(ax, img, x0, y0, sz, sz)
P, S = 32, 16
for y in range(0, 256 - P + 1, S):
    for x in range(0, 256 - P + 1, S):
        if lung[y:y + P, x:x + P].mean() > 0.5 and (x // S + y // S) % 3 == 0:
            ax.add_patch(Rectangle((x0 + x / 256 * sz, y0 + (256 - y - P) / 256 * sz), P / 256 * sz, P / 256 * sz,
                                   fc="none", ec="#E8A23A", lw=0.45, zorder=5))
text(ax, x0 + sz / 2, y0 - 0.05, "Coupe $x$ (HU) ; patchs 64×64,\npas 32, ≥ 50 % de poumon", fs=5.2, va="top", color=SOFT)

# banque de résidus
bx = 1.30
text(ax, 2.20, 3.17, "Quatre résidus haute fréquence (Fridrich et Kodovský)", fs=5.6, weight="bold", color=SOFT)
box(ax, bx, 2.36, 0.86, 0.50, "Médian 3×3", "$r=x-\\mathrm{méd}_{3\\times3}(x)$", kind="reuse", fs=5.3)
matrix(ax, FZ.K_SRM3 * 4, bx + 1.05, 2.36, 0.13, fmt="{:g}", fs=5.0)
text(ax, bx + 1.05 + 0.44, 2.55, "× 1/4", fs=5.2, ha="left", color=SOFT)
text(ax, bx + 1.245, 2.92, "SRM 3×3", fs=5.4, weight="bold")
matrix(ax, FZ.K_KV * 12, bx, 1.30, 0.115, fmt="{:g}", fs=4.6)
text(ax, bx + 0.29, 1.95, "KV 5×5", fs=5.4, weight="bold")
text(ax, bx + 0.66, 1.36, "× 1/12", fs=5.2, ha="left", color=SOFT)
matrix(ax, np.array([[-1, 1]]), bx + 1.15, 1.55, 0.13, fs=5.0)
text(ax, bx + 1.28, 1.78, "Différence $\\Delta_x$", fs=5.4, weight="bold")
arrow(ax, (x0 + sz, y0 + sz / 2), (bx - 0.04, y0 + sz / 2), ms=5)

# descripteurs par patch
vx, unit = 3.45, 0.103
text(ax, vx, 3.05, "Descripteurs par patch (27)", fs=5.9, weight="bold", ha="left")
vy = 2.72
vector(ax, vx, vy, [(1, "own", None), (2, "own", None), (1, "own", None), (2, "own", None), (6, "own", None),
                    (2, "own", None), (12, "own", None), (1, "own", None)], unit, 0.17, fs=5.0)
labs = [(0.5, "$\\hat\\sigma$", 1), (2.0, "$\\rho_x,\\rho_y$", 0), (3.5, "$E_6/E_3$", 1), (5.0, "$\\sigma,\\kappa$", 0),
        (9.0, "$E_1\\ldots E_6$", 1), (13.0, "$\\mu,\\sigma$ HU", 0), (20.0, "$|z|$ des 12 premiers", 1), (26.5, "dup.", 0)]
for c, l, low in labs:
    text(ax, vx + c * unit, vy - 0.04 - 0.11 * low, l, fs=4.9, va="top", color=SOFT)
arrow(ax, (bx + 1.75, 2.62), (vx - 0.04, vy + 0.08), ms=5)
hb = box(ax, vx + 0.2, 1.72, 2.35, 0.50, "HGB local (Ke et al., scikit-learn)",
         "300 itérations, taux 0,06, classes équilibrées\nnégatifs 3 : 1 ; étiquette : masque > 30 % du patch", kind="reuse", fs=5.2)
arrow(ax, (vx + 1.4, vy - 0.30), (vx + 1.4, hb[1] + hb[3]), ms=5)
pb = box(ax, vx + 0.85, 1.25, 1.05, 0.30, "$p_{for}$ par patch", "", kind="data", fs=5.5)
arrow(ax, (vx + 1.37, hb[1]), (vx + 1.37, pb[1] + pb[3]), ms=5)

# descripteurs globaux
text(ax, vx, 0.98, "Descripteurs globaux $F_{glob}(x)$ (40), vers la fusion", fs=5.9, weight="bold", ha="left")
vector(ax, vx, 0.66, [(12, "own", "résidus"), (8, "own", "$\\rho$, spectre"),
                      (18, "own", "incohérence des blocs 16×16"), (2, "own", None)], 0.0695, 0.17, fs=5.0)
text(ax, vx + 39 * 0.0695, 0.62, "dup.", fs=4.9, va="top", color=SOFT)

# définitions
defs = ["$\\hat\\sigma=1{,}4826\\cdot\\mathrm{MAD}(r)$ ;  $\\rho_x=\\Sigma\\,\\tilde r_{i,j}\\tilde r_{i,j+1}/\\Sigma\\,\\tilde r_{i,j}^2$ (résidu KV)",
        "$E_b$ : énergie normalisée du résidu KV dans l'anneau de fréquence $b$",
        "$|z|$ : écart robuste $(f-\\mathrm{méd})/(1{,}4826\\cdot\\mathrm{MAD})$ entre patchs de la coupe",
        "dup. : part du patch couverte par des fenêtres 8×8 identiques",
        "(écart-type > 3 HU, distance > 8 pixels), repérées par hachage"]
for k, d in enumerate(defs):
    text(ax, 0.05, 1.05 - k * 0.125, d, fs=5.0, ha="left", color=SOFT)
legend(ax, 0.05, 0.36)
save(fig, "figures/arch_forensique.png")

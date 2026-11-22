from common import *
import sys
sys.path.insert(0, "../simulateur")
import features as FZ

W, H = 6.3, 2.75
fig, ax = canvas(W, H)
d = np.load("../simulateur/simct_tamper.npz")
mp = np.load("../simulateur/maps_hybrid.npz")
i = int(mp["idx"][0])
img = d["X"][i].astype(np.float32)
L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)[i]
pm = mp["m1"]

# coupe et blocs
x0, y0, sz = 0.05, 1.55, 1.0
thumb(ax, img, x0, y0, sz, sz)
er = __import__("scipy.ndimage", fromlist=["x"]).binary_erosion(L, iterations=2)
B = 16
for y in range(0, 256 - B + 1, B // 2):
    for x in range(0, 256 - B + 1, B // 2):
        if er[y:y + B, x:x + B].all() and (x // 8 + y // 8) % 5 == 0:
            ax.add_patch(Rectangle((x0 + x / 256 * sz, y0 + (256 - y - B) / 256 * sz), B / 256 * sz, B / 256 * sz,
                                   fc="none", ec="#E8A23A", lw=0.4, zorder=5))
text(ax, x0 + sz / 2, y0 - 0.05, "Coupe SimCT-Tamper\n256×256 ; blocs 16×16,\npas 8, poumon érodé", fs=5.2, va="top", color=SOFT)

# descripteurs par bloc
vx, unit = 1.32, 0.105
text(ax, vx, 2.50, "Descripteurs par bloc (15)", fs=5.9, weight="bold", ha="left")
vector(ax, vx, 2.22, [(6, "own", "$f_b$"), (6, "own", "$|z_b|$"), (1, "own", None), (2, "own", None)], unit, 0.17, fs=5.0)
text(ax, vx + 12.5 * unit, 2.18, "dup.", fs=4.9, va="top", color=SOFT)
text(ax, vx + 14 * unit, 2.07, "$\\mu,\\sigma$", fs=4.9, va="top", color=SOFT)
text(ax, vx, 1.86, "$f_b$ : $\\hat\\sigma$, $\\rho_x$, $\\rho_y$, $E_4/E_2$, $\\sigma$ et $\\kappa$ du résidu KV\n$z_b$ : normalisation robuste entre les blocs de la coupe (2.7)",
     fs=5.0, ha="left", va="top", color=SOFT)
arrow(ax, (x0 + sz, y0 + sz * 0.72), (vx - 0.04, 2.30), ms=5)
hl = box(ax, 3.12, 2.06, 1.22, 0.46, "HGB local", "250 itérations, taux 0,06\nétiquette : masque > 30 %", kind="reuse", fs=5.2)
arrow(ax, (vx + 15 * unit + 0.02, 2.30), (hl[0], 2.30), ms=5)
mx, my, ms_ = 4.62, 1.78, 0.62
thumb(ax, img, mx, my, ms_, ms_)
ov = np.ma.masked_less(pm, 0.08)
ax.imshow(ov, extent=(mx, mx + ms_, my, my + ms_), cmap="autumn_r", vmin=0, vmax=0.8, alpha=0.85, zorder=4)
arrow(ax, (hl[0] + hl[2], 2.30), (4.60, 2.30), ms=5)
text(ax, 5.30, 2.09, "carte : moyenne\ndes $p_b$ par pixel", fs=5.0, ha="left", color=SOFT)
text(ax, 4.93, 2.47, "$p_b$ par bloc", fs=5.2, va="bottom", color=SOFT)

# décision globale
y2 = 0.62
ag = box(ax, 4.38, y2, 1.20, 0.58, "Agrégation $A(x)$", "max, moyenne des 3\net des 10 plus grands,\nnombre > 0,5", kind="own", fs=5.0)
arrow(ax, (4.93, my - 0.02), (4.93, y2 + 0.58), ms=4.5)
u = 0.030
vx2 = 2.88
vector(ax, vx2, y2 + 0.21, [(40, "own", None), (4, "own", None)], u, 0.15, fs=4.9)
text(ax, vx2 + 20 * u, y2 + 0.41, "$F_{glob}(x)$ (40)", fs=5.0, color=SOFT, va="bottom")
text(ax, vx2 + 22 * u, y2 + 0.16, "44 entrées", fs=5.0, color=SOFT, va="top")
arrow(ax, (ag[0], y2 + 0.29), (vx2 + 44 * u + 0.02, y2 + 0.29), ms=4.5)
hg = box(ax, 1.40, y2 - 0.01, 1.30, 0.60, "HGB global", "250 itérations, taux 0,06\n→ $P(\\mathrm{falsifiée}\\mid x)$", kind="reuse", fs=5.1)
arrow(ax, (vx2 - 0.02, y2 + 0.29), (hg[0] + hg[2], y2 + 0.29), ms=4.5)
text(ax, 0.05, 0.36, "Apprentissage sans fuite : 5 plis externes par patient ; les $p_b$ qui servent à apprendre le HGB global sont prédits hors pli, "
     "dans 4 plis internes", fs=5.0, ha="left", color=SOFT)
legend(ax, 0.05, 0.05)
save(fig, "figures/arch_poc.png")

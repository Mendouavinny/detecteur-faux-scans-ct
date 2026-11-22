from common import *

W, H = 6.3, 3.42
fig, ax = canvas(W, H)
rng = np.random.default_rng(3)


def column(ax, x, y, n, h, w, kind, label, vals=None):
    fc, ec = COL[kind]
    for i in range(n):
        v = vals[i] if vals is not None else None
        f = fc if v is None else shade(fc, 1 - 0.28 * v)
        ax.add_patch(Rectangle((x, y + i * h), w, h, fc=f, ec=ec, lw=0.45, zorder=4))
    text(ax, x + w / 2, y + n * h + 0.05, label, fs=5.4, va="bottom")
    text(ax, x + w / 2, y - 0.03, "⋮", fs=6, va="top", color=SOFT)
    return (x, y, w, n * h)


# --- ligne 1 : sorties des branches, fusion locale, agrégation ---
y1, ch, cw, n = 2.30, 0.085, 0.15, 8
text(ax, 0.05, 3.30, "Sorties des branches pour les $N$ patchs", fs=5.6, weight="bold", color=SOFT, ha="left")
cols = []
for k, (lab, kind) in enumerate([("$p_{for}$", "data"), ("$p_{cnn}$", "data"), ("$z$", "data"), ("$|z|$", "data")]):
    cols.append(column(ax, 0.12 + k * 0.27, y1, n, ch, cw, kind, lab, rng.random(n)))
lr = box(ax, 1.27, y1 + 0.08, 1.27, 0.56, "Fusion locale", "régression logistique\n4 entrées, classes équilibrées", kind="reuse", fs=5.3)
arrow(ax, (1.12, y1 + n * ch / 2), (lr[0], y1 + n * ch / 2), ms=5)
pf = column(ax, 2.74, y1, n, ch, cw, "own", "$p_{fus}$", rng.random(n))
arrow(ax, (lr[0] + lr[2], y1 + n * ch / 2), (pf[0], y1 + n * ch / 2), ms=5)
ag = box(ax, 3.12, y1 + 0.02, 1.62, 0.68, "Agrégation $A(\\cdot)$",
         "max, moyenne des 3 et des 10 plus grands,\nnombre > 0,5, pour chacune des 5 séries\n$p_{fus}$, $p_{for}$, $p_{cnn}$, $z$, $|z|$",
         kind="own", fs=5.1)
arrow(ax, (pf[0] + pf[2], y1 + n * ch / 2), (ag[0], y1 + n * ch / 2), ms=5)
va = vector(ax, 4.97, y1 + 0.28, [(20, "own", None)], 0.062, 0.15, fs=5.0)
text(ax, 4.97 + 10 * 0.062, y1 + 0.47, "5 × 4 = 20 valeurs", fs=5.2, va="bottom", color=SOFT)
arrow(ax, (ag[0] + ag[2], y1 + 0.355), (4.95, y1 + 0.355), ms=5)

# --- ligne 2 (de droite à gauche) : concaténation, méta-classifieur, décision ---
y2 = 1.35
text(ax, 4.62 + 20 * 0.0265, y2 + 0.31, "$F_{glob}(x)$ (40)", fs=5.2, color=SOFT, va="bottom")
cc = vector(ax, 4.62, y2 + 0.12, [(40, "own", None), (20, "own", None)], 0.0265, 0.15, fs=4.9)
text(ax, 4.62 + 30 * 0.0265, y2 + 0.06, "concaténation : 60 entrées", fs=5.2, va="top", color=SOFT)
arrow(ax, (5.94, y1 + 0.28), (5.94, y2 + 0.28), ms=5)
hb = box(ax, 3.02, y2 - 0.02, 1.38, 0.58, "Méta-classifieur HGB", "300 itérations, taux 0,05\nappris sur les coupes\nde validation", kind="reuse", fs=5.2)
arrow(ax, (4.60, y2 + 0.195), (hb[0] + hb[2], y2 + 0.195), ms=5)
sb = box(ax, 1.60, y2 + 0.06, 0.82, 0.42, "$s(x)\\in[0,1]$", "score de la coupe", kind="data", fs=5.4)
arrow(ax, (hb[0], y2 + 0.27), (sb[0] + sb[2], y2 + 0.27), ms=5)
db = box(ax, 0.05, y2 - 0.02, 1.33, 0.58, "Décision", "$s(x)\\geq\\tau_1$ : coupe signalée\n$\\tau_1$ : 99$^e$ centile des scores\nhors pli des authentiques", kind="own", fs=5.1)
arrow(ax, (sb[0], y2 + 0.27), (db[0] + db[2], y2 + 0.27), ms=5)

# --- ligne 3 : cartes ---
y3 = 0.62
mf = box(ax, 1.85, y3, 2.15, 0.42, "Carte de fusion (renvoyée par le prototype)",
         "$C(u)$ = moyenne des $p_{fus}$ des patchs couvrant $u$", kind="own", fs=5.1)
import matplotlib.patheffects as pe
ax.plot([pf[0] + pf[2] / 2] * 2, [y1 - 0.12, y3 + 0.5], color="#404040", lw=0.75, zorder=3, path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
arrow(ax, (pf[0] + pf[2] / 2, y3 + 0.5), (pf[0] + pf[2] / 2, y3 + 0.42), ms=5)
mu = box(ax, 4.20, y3, 2.05, 0.42, "Carte U-Net (branche CNN)", "moyenne des cartes $\\hat m$ des patchs couvrant $u$",
         kind="own", fs=5.1)
text(ax, 0.05, y3 + 0.21, "Localisation : cartes\nbinarisées à 0,5\n(Dice, IoU)", fs=5.1, ha="left", color=SOFT)

for k, line in enumerate(["Variante forensique + CNN : régression locale sur $[p_{for}, p_{cnn}]$, agrégation de 3 séries, 40 + 12 = 52 entrées ; "
                          "branche seule : moyenne des 3 plus grands scores de patch",
                          "Patients : 210 en entraînement (branches) · 45 en validation (fusion locale, méta-classifieur, seuil hors pli en 5 plis) · 45 en test"]):
    text(ax, 0.05, 0.42 - k * 0.12, line, fs=5.0, ha="left", color=SOFT)
legend(ax, 0.05, 0.02)
save(fig, "figures/arch_fusion.png")

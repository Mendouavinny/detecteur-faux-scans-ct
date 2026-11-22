"""Outils de dessin des schémas d'architecture (style « réseau » : tenseurs en volumes, opérations en boîtes)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
from matplotlib.colors import to_rgb

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "mathtext.fontset": "dejavusans",
                     "axes.unicode_minus": True})

# remplissage, contour
COL = {
    "own": ("#FCE2CC", "#C45A1E"),     # apport de ce travail
    "reuse": ("#E6E6E6", "#666666"),   # composant repris d'un travail publié
    "data": ("#DDEAF6", "#3A6EA5"),    # tenseur, donnée, sortie
    "plain": ("#FFFFFF", "#404040"),
}
INK = "#262626"
SOFT = "#5A5A5A"


def shade(c, f):
    r, g, b = to_rgb(c)
    return (r * f, g * f, b * f)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def text(ax, x, y, s, fs=6.5, ha="center", va="center", color=INK, weight="normal", style="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color, fontweight=weight, fontstyle=style,
                   zorder=z, **kw)


def box(ax, x, y, w, h, title, body="", kind="own", fs=6.4, r=0.035, lw=0.75, z=3, title_fs=None, ls="-"):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, linestyle=ls))
    lines = [(title, True)] if title else []
    lines += [(b, False) for b in body.split("\n")] if body else []
    lh = fs * 1.32 / 72
    y0 = y + h / 2 + (len(lines) - 1) * lh / 2
    for i, (s, b) in enumerate(lines):
        text(ax, x + w / 2, y0 - i * lh, s, fs=(title_fs or fs) if b else fs, weight="bold" if b else "normal", z=z + 1)
    return (x, y, w, h)


def slab(ax, cx, cy, s, t, kind="data", label=None, fs=5.6, lab_dy=0.06, z=4, depth=0.32, label_above=False):
    """Tenseur en volume : face avant t (canaux) × s (taille spatiale), profondeur depth·s."""
    fc, ec = COL[kind]
    o = depth * s
    x, y = cx - t / 2 - o / 2, cy - s / 2 - o / 2
    ax.add_patch(Rectangle((x, y), t, s, fc=fc, ec=ec, lw=0.6, zorder=z))
    ax.add_patch(Polygon([(x, y + s), (x + o, y + s + o), (x + t + o, y + s + o), (x + t, y + s)],
                         closed=True, fc=shade(fc, 0.93), ec=ec, lw=0.6, zorder=z))
    ax.add_patch(Polygon([(x + t, y), (x + t + o, y + o), (x + t + o, y + s + o), (x + t, y + s)],
                         closed=True, fc=shade(fc, 0.84), ec=ec, lw=0.6, zorder=z))
    if label:
        if label_above:
            text(ax, cx, y + s + o + lab_dy, label, fs=fs, va="bottom", color=SOFT)
        else:
            text(ax, cx, y - lab_dy, label, fs=fs, va="top", color=SOFT)
    # points d'accroche : gauche, droite, haut, bas (milieux)
    return dict(l=(x, y + s / 2), r=(x + t + o, y + o + s / 2), t=(x + o / 2 + t / 2, y + s + o), b=(x + t / 2, y),
                cx=cx, cy=cy)


def arrow(ax, p, q, lw=0.75, color="#404040", ls="-", ms=6.5, rad=0.0, z=2, head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-", mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def poly_arrow(ax, pts, lw=0.75, color="#404040", ls="-", ms=6.5, z=2):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], lw=lw, color=color, ls=ls, ms=ms, z=z)


def chevron(ax, x, y, w, h, s, kind="plain", fs=6.4):
    fc, ec = COL[kind]
    n = h * 0.32
    ax.add_patch(Polygon([(x, y), (x + w - n, y), (x + w, y + h / 2), (x + w - n, y + h), (x, y + h)], closed=True,
                         fc=fc, ec=ec, lw=0.75, zorder=3))
    text(ax, x + (w - n) / 2 + n * 0.15, y + h / 2, s, fs=fs, z=4)


def thumb(ax, img, x, y, w, h, vmin=-1350, vmax=150, cmap="gray", z=3, ec="#404040"):
    ax.imshow(img, extent=(x, x + w, y, y + h), cmap=cmap, vmin=vmin, vmax=vmax, zorder=z, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=0.6, zorder=z + 1))


def matrix(ax, M, x, y, cell, fmt="{:g}", fs=4.6, kind="reuse", z=4):
    """Petite grille de coefficients (filtre)."""
    fc, ec = COL[kind]
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x + j * cell, y + (n - 1 - i) * cell), cell, cell, fc=fc, ec=ec, lw=0.35, zorder=z))
            text(ax, x + (j + 0.5) * cell, y + (n - 1 - i + 0.5) * cell, fmt.format(M[i, j]).replace("-", "−"),
                 fs=fs, z=z + 1)
    return (x, y, m * cell, n * cell)


def vector(ax, x, y, segs, unit, h, fs=5.0, z=4, labels_below=True):
    """Vecteur de descripteurs découpé en segments : segs = [(n, kind, label)]."""
    cx = x
    for n, kind, lab in segs:
        fc, ec = COL[kind]
        w = n * unit
        ax.add_patch(Rectangle((cx, y), w, h, fc=fc, ec=ec, lw=0.5, zorder=z))
        text(ax, cx + w / 2, y + h / 2, str(n), fs=fs, z=z + 1)
        if lab:
            text(ax, cx + w / 2, y - 0.04 if labels_below else y + h + 0.04, lab, fs=fs - 0.4,
                 va="top" if labels_below else "bottom", color=SOFT, z=z + 1)
        cx += w
    return cx


def bracket(ax, x1, x2, y, label, fs=5.8, up=True, color=SOFT, lw=0.6):
    d = 0.05 if up else -0.05
    ax.plot([x1, x1, x2, x2], [y - d, y, y, y - d], color=color, lw=lw, zorder=2)
    text(ax, (x1 + x2) / 2, y + (0.035 if up else -0.035), label, fs=fs, va="bottom" if up else "top", color=color)


def legend(ax, x, y, items=("reuse", "own", "data"), fs=5.9):
    names = {"reuse": "Composant repris d'un travail publié", "own": "Apport de ce travail",
             "data": "Tenseur ou sortie (C × H × W)", "plain": "Bloc détaillé dans une autre figure",
             "out": "Sortie"}
    cx = x
    for k in items:
        fc, ec = COL["data" if k == "out" else k]
        ax.add_patch(Rectangle((cx, y), 0.16, 0.1, fc=fc, ec=ec, lw=0.6, zorder=5))
        t = text(ax, cx + 0.21, y + 0.05, names[k], fs=fs, ha="left", z=6)
        cx += 0.21 + len(names[k]) * fs * 0.0072 + 0.28


def save(fig, path, dpi=400):
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def ct_images():
    d = np.load("../simulateur/simct_tamper.npz")
    X, Y = d["X"], d["Y"]
    M = np.unpackbits(d["M"], axis=-1)[:, :, :256].astype(bool)
    L = np.unpackbits(d["L"], axis=-1)[:, :, :256].astype(bool)
    i = int(np.where(Y == 1)[0][3])
    return X[i].astype(np.float32), M[i], L[i]

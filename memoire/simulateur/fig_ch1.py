"""Figures du chapitre 1 (revue de littérature), produites par calcul.

Si un fichier real_slice.npy (coupe CT réelle en HU, int16, par ex. extraite de LIDC-IDRI par le notebook)
est présent, il est utilisé ; sinon une coupe thoracique simulée (simct.py) sert d'illustration.
Usage : python fig_ch1.py [dossier_sortie]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import radon, iradon
import simct as S
import features as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_ch1/"
os.makedirs(OUT, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
rng = np.random.default_rng(7)
REAL = os.path.exists("real_slice.npy")
SRC = "coupe LIDC-IDRI" if REAL else "coupe simulée"


def win(img, L=-600, W=1500):
    return np.clip((img.astype(float) - (L - W / 2)) / W, 0, 1)


def noaxis(*axs):
    for a in axs:
        a.set_xticks([]); a.set_yticks([])


# ------------------------------------------------------------------ données de base
hu_true, lungs, nods = S.make_phantom(np.random.default_rng(11), with_nodule=True)
def lung_mask(hu):
    m = hu < -400
    lab, n = ndi.label(m)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = m & ~np.isin(lab, border)
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3))


if REAL:   # coupe réelle (LIDC-IDRI) : sert d'objet et d'image pour les figures 1.1 à 1.3, 1.7 et 1.8
    img = np.load("real_slice.npy").astype(np.int16)
    if img.shape != (S.N, S.N):
        img = np.clip(ndi.zoom(img.astype(float), S.N / img.shape[0], order=1), -1024, 3071).astype(np.int16)
    hu_true, lungs = img.astype(float), lung_mask(img)
else:
    img = S.acquire(hu_true, rng, 6e4, "ramp")

# ------------------------------------------------------------------ Fig 1 : principe CT
theta = S.THETA
mu = np.clip(S.MU_W * (1 + hu_true / 1000.0), 0, None) * S.PIX
circ = S.ellipse(hu_true.shape, S.N / 2 - 0.5, S.N / 2 - 0.5, S.N / 2 - 1, S.N / 2 - 1)
sino = radon(mu * circ, theta, circle=True)
bp = iradon(sino, theta, circle=True, filter_name=None)
fbp = iradon(sino, theta, circle=True, filter_name="ramp")
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
ax[0].imshow(win(hu_true), cmap="gray"); ax[0].set_title("(a) Objet : carte d'atténuation")
ax[1].imshow(sino, cmap="gray", aspect="auto", extent=[0, 180, sino.shape[0], 0]); ax[1].set_title("(b) Sinogramme $p_\\theta(t)$")
ax[1].set_xlabel("angle θ (°)"); ax[1].set_ylabel("position t")
ax[2].imshow(bp, cmap="gray"); ax[2].set_title("(c) Rétroprojection non filtrée")
ax[3].imshow(win((fbp / S.PIX / S.MU_W - 1) * 1000), cmap="gray"); ax[3].set_title("(d) Rétroprojection filtrée (FBP)")
noaxis(ax[0], ax[2], ax[3]); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig(OUT + "f1_principe_ct.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 2 : noyaux et dose -> bruit
kernels = [("rampe", "ramp"), ("Shepp-Logan", "shepp-logan"), ("cosinus", "cosine"), ("Hann", "hann")]
I0 = 4e4
counts = np.maximum(rng.poisson(I0 * np.exp(-sino)), 1)
sino_n = -np.log(counts / I0)
ys, xs = np.where(ndi.binary_erosion(lungs, iterations=6))
cy, cx = int(np.median(ys)), int(np.median(xs)) - 30
fig = plt.figure(figsize=(13, 6.2))
nps = {}
for k, (lab, ker) in enumerate(kernels):
    rec_n = (iradon(sino_n, theta, circle=True, filter_name=ker) / S.PIX / S.MU_W - 1) * 1000
    rec_0 = (iradon(sino, theta, circle=True, filter_name=ker) / S.PIX / S.MU_W - 1) * 1000
    noise = rec_n - rec_0
    a = fig.add_subplot(2, 4, k + 1)
    a.imshow(win(rec_n[cy - 40:cy + 40, cx - 40:cx + 40]), cmap="gray")
    a.set_title(f"Noyau {lab}\nσ = {noise[lungs].std():.0f} HU"); noaxis(a)
    # spectre de puissance du bruit (NPS) radial, sur des fenêtres 64x64 dans le corps
    acc = []
    for (yy, xx) in [(96, 96), (96, 160), (160, 96), (160, 160), (128, 128)]:
        p = noise[yy - 32:yy + 32, xx - 32:xx + 32]
        acc.append(np.abs(np.fft.fftshift(np.fft.fft2(p - p.mean()))) ** 2)
    P = np.mean(acc, 0); h = P.shape[0]
    yy, xx = np.mgrid[:h, :h]; r = np.sqrt((yy - h / 2) ** 2 + (xx - h / 2) ** 2).astype(int)
    nps[lab] = np.bincount(r.ravel(), P.ravel())[:33] / np.bincount(r.ravel())[:33]
a = fig.add_subplot(2, 2, 3)
for (lab, _), c in zip(kernels, (C1, C2, C3, C4)):
    f = np.arange(33) / 64 / (S.PIX / 10)  # cycles/cm
    a.plot(f, nps[lab] / nps[lab].max(), color=c, lw=2, label=lab)
a.set_xlabel("Fréquence spatiale (cycles/cm)"); a.set_ylabel("NPS normalisé"); a.set_title("(b) Spectre de puissance du bruit (NPS)")
a.legend(frameon=False, fontsize=8); a.grid(color=GRID); a.set_axisbelow(True)
a = fig.add_subplot(2, 2, 4)
doses = np.array([1e4, 2e4, 4e4, 8e4, 1.6e5, 3.2e5])
sig = []
for d0 in doses:
    cnt = np.maximum(rng.poisson(d0 * np.exp(-sino)), 1)
    rn = (iradon(-np.log(cnt / d0), theta, circle=True, filter_name="ramp") / S.PIX / S.MU_W - 1) * 1000
    r0 = (iradon(sino, theta, circle=True, filter_name="ramp") / S.PIX / S.MU_W - 1) * 1000
    sig.append((rn - r0)[lungs].std())
a.loglog(doses, sig, "o-", color=C1, lw=2, ms=6, label="mesuré (FBP rampe)")
a.loglog(doses, sig[0] * np.sqrt(doses[0] / doses), "--", color=INK2, lw=1, label="loi en $1/\\sqrt{I_0}$")
a.set_xlabel("Dose $I_0$ (photons par rayon)"); a.set_ylabel("σ du bruit (HU)"); a.set_title("(c) Bruit en fonction de la dose")
a.legend(frameon=False, fontsize=8); a.grid(color=GRID, which="both"); a.set_axisbelow(True)
fig.text(0.01, 0.97, "(a) Même acquisition reconstruite avec quatre noyaux (zoom pulmonaire)", fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(OUT + "f2_noyaux_bruit.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 3 : fenêtrage HU
fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
for a, (t, L, W) in zip(ax[:3], [("Fenêtre pulmonaire\n(L = −600, W = 1500)", -600, 1500),
                                  ("Fenêtre médiastinale\n(L = 40, W = 400)", 40, 400),
                                  ("Fenêtre osseuse\n(L = 400, W = 1800)", 400, 1800)]):
    a.imshow(win(img, L, W), cmap="gray"); a.set_title(t, fontsize=9); noaxis(a)
v = img[img > -1000].ravel()
ax[3].hist(v, bins=160, range=(-1100, 1100), color=C1)
for x0, lab in [(-1000, "air"), (-860, "poumon"), (-90, "graisse"), (45, "tissus"), (700, "os")]:
    ax[3].axvline(x0, color=INK2, lw=0.7, ls=":")
    ax[3].text(x0, 0.02, lab, rotation=90, fontsize=7, color=INK2, ha="right", va="bottom",
               transform=ax[3].get_xaxis_transform())
ax[3].set_yscale("log"); ax[3].set_xlabel("HU"); ax[3].set_title("Histogramme des HU", fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "f3_fenetrage_hu.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 4 : manipulations classiques
base_hu, base_l, base_n = S.make_phantom(np.random.default_rng(21), with_nodule=True)
base = S.acquire(base_hu, rng, 6e4, "ramp")
for _try in range(50):
    inj = S.tamper_injection(base, base_l, np.random.default_rng(300 + _try))
    yy_, xx_ = np.where(inj[1])
    if all(np.hypot(yy_.mean() - a_, xx_.mean() - b_) > 35 for a_, b_, _ in base_n):
        break
import cv2


def clean_inpaint(im, lung, nod, rng_):
    """Suppression par inpainting de Telea limitée au disque du nodule + bruit au niveau estimé."""
    cy_, cx_, r_ = nod
    m_ = np.zeros(im.shape, np.uint8); cv2.circle(m_, (int(cx_), int(cy_)), int(r_ * 1.6 + 3), 1, -1)
    lo, hi = -1024.0, 400.0
    u8 = np.clip((im.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(u8, m_, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
    out_ = im.astype(np.float32).copy()
    mb = m_.astype(bool)
    out_[mb] = inp[mb] + rng_.normal(0, S.mad_sigma(im, lung), mb.sum())
    return np.clip(np.round(out_), -1024, 3071).astype(np.int16), mb


att = [("Injection (collage)", *inj),
       ("Suppression (inpainting)", *clean_inpaint(base, base_l, base_n[0], rng)),
       ("Copier-déplacer", *S.tamper_copymove(base, base_l, rng))]
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for i, (name, out, m) in enumerate(att):
    ys, xs = np.where(m); cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = max(cy - 40, 0), max(cx - 40, 0)
    ax[i, 0].imshow(win(base), cmap="gray"); ax[i, 0].contour(m, [0.5], colors=[C2], linewidths=1)
    ax[i, 1].imshow(win(base[y0:y0 + 80, x0:x0 + 80]), cmap="gray")
    ax[i, 2].imshow(win(out[y0:y0 + 80, x0:x0 + 80]), cmap="gray")
    d = np.abs(out.astype(float) - base)[y0:y0 + 80, x0:x0 + 80]
    ax[i, 3].imshow(d, cmap="magma", vmin=0, vmax=np.percentile(d, 99.5) + 1)
    ax[i, 0].set_ylabel(name)
    noaxis(*ax[i])
for j, t in enumerate(["Coupe (région ciblée)", "Avant (zoom)", "Après (zoom)", "|Différence|"]):
    ax[0, j].set_title(t)
plt.tight_layout(); plt.savefig(OUT + "f4_manipulations.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 5 : FGSM sur un classifieur de nodules
from sklearn.linear_model import LogisticRegression
P_ = 24
pos, neg = [], []
for s in range(60):
    r_ = np.random.default_rng(100 + s)
    h_, l_, n_ = S.make_phantom(r_, with_nodule=True)
    im_ = S.acquire(h_, r_, 6e4, "ramp").astype(np.float32)
    for (cy_, cx_, rr) in n_:
        p = im_[cy_ - P_ // 2:cy_ + P_ // 2, cx_ - P_ // 2:cx_ + P_ // 2]
        if p.shape == (P_, P_): pos.append(p)
    idx = np.argwhere(ndi.binary_erosion(l_, iterations=16))
    for _ in range(3):
        cy_, cx_ = idx[r_.integers(len(idx))]
        if all(abs(cy_ - a_) > P_ or abs(cx_ - b_) > P_ for a_, b_, _ in n_):
            neg.append(im_[cy_ - P_ // 2:cy_ + P_ // 2, cx_ - P_ // 2:cx_ + P_ // 2])
X = np.array(pos + neg).reshape(len(pos) + len(neg), -1) / 1000.0
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
clf = LogisticRegression(C=0.5, max_iter=5000).fit(X, y)
pp_ = clf.predict_proba(X[:len(pos)])[:, 1]
x0 = X[int(np.argmin(np.abs(pp_ - 0.9)))]
# gradient de la perte logistique / entrée (cible : « pas de nodule ») = (p - 0) * w  -> FGSM
p0 = clf.predict_proba(x0[None])[0, 1]
eps_list = [0, 10, 25, 50]
res = []
for eps in eps_list:
    xa = x0 + (eps / 1000.0) * np.sign(clf.coef_[0] * (p0 - 0.0)) * -1  # descente vers la classe 0
    res.append((xa, clf.predict_proba(xa[None])[0, 1]))
fig, ax = plt.subplots(1, 5, figsize=(13, 3.5))
ax[0].imshow(win(x0.reshape(P_, P_) * 1000), cmap="gray"); ax[0].set_title(f"Patch original\nP(nodule) = {p0:.2f}")
pert = (res[-1][0] - x0).reshape(P_, P_) * 1000
ax[1].imshow(pert, cmap="coolwarm", vmin=-50, vmax=50); ax[1].set_title("Perturbation ε·sign(∇)\n(ε = 50 HU)")
for k, (eps, (xa, pa)) in enumerate(zip(eps_list[1:], res[1:])):
    ax[2 + k].imshow(win(xa.reshape(P_, P_) * 1000), cmap="gray"); ax[2 + k].set_title(f"ε = {eps} HU\nP(nodule) = {pa:.2f}")
noaxis(*ax)
plt.tight_layout(); plt.savefig(OUT + "f5_fgsm.png", dpi=200); plt.close()
FGSM_INFO = dict(n_pos=len(pos), n_neg=len(neg), p0=float(p0), p=[float(r[1]) for r in res])

# ------------------------------------------------------------------ Fig 6 : processus de diffusion direct
T = 1000
betas = np.linspace(1e-4, 0.02, T); abar = np.cumprod(1 - betas)
x0d = np.clip((img.astype(float) + 1100) / 1400, 0, 1) * 2 - 1
ts = [0, 50, 150, 300, 600, 999]
fig, ax = plt.subplots(1, len(ts), figsize=(13, 3.2))
eps_ = rng.normal(size=x0d.shape)
for a, t in zip(ax, ts):
    xt = x0d if t == 0 else np.sqrt(abar[t]) * x0d + np.sqrt(1 - abar[t]) * eps_
    a.imshow(np.clip((xt + 1) / 2, 0, 1), cmap="gray"); a.set_title(f"t = {t}\n$\\sqrt{{\\bar\\alpha_t}}$ = {np.sqrt(abar[t]) if t else 1:.2f}", fontsize=9)
    noaxis(a)
plt.tight_layout(); plt.savefig(OUT + "f6_diffusion_directe.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 7 : traces spectrales du suréchantillonnage
f0 = img.astype(float)
low = f0[::2, ::2]
bil = ndi.zoom(low, 2, order=1)
zi = np.zeros_like(f0); zi[::2, ::2] = low
k = rng.normal(0, 1, (4, 4)); k = np.abs(k); k /= k.sum() / 4
tconv = ndi.convolve(zi, k)                                    # convolution transposée (stride 2) aléatoire


def spec(x):
    r = x - ndi.gaussian_filter(x, 1.0)
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(r))))


fig, ax = plt.subplots(1, 3, figsize=(11, 4.2))
for a, (t, x) in zip(ax, [("Coupe d'origine", f0), ("Suréchantillonnage bilinéaire", bil), ("Convolution transposée (type GAN)", tconv)]):
    a.imshow(spec(x), cmap="viridis"); a.set_title(t, fontsize=10); noaxis(a)
plt.tight_layout(); plt.savefig(OUT + "f7_traces_spectrales.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 8 : résidus SRM
out_i, m_i = att[0][1], att[0][2]
ys, xs = np.where(m_i); cy, cx = int(ys.mean()), int(xs.mean())
R = F.residuals(out_i)
fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.4))
sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
ax[0].imshow(win(out_i[sl]), cmap="gray"); ax[0].set_title("Région injectée")
for a, (k_, t) in zip(ax[1:], [("med", "Résidu médian 3×3"), ("srm3", "Résidu SRM 3×3"), ("kv", "Résidu KV 5×5")]):
    rr = R[k_][sl]; v = np.percentile(np.abs(rr), 98)
    a.imshow(rr, cmap="gray", vmin=-v, vmax=v); a.contour(m_i[sl], [0.5], colors=[C2], linewidths=1); a.set_title(t)
noaxis(*ax)
plt.tight_layout(); plt.savefig(OUT + "f8_residus_srm.png", dpi=200); plt.close()

# ------------------------------------------------------------------ Fig 9 : copier-déplacer et appariement de blocs
out_c, m_c = att[2][1], att[2][2]
dup = F.duplicate_map(out_c, out_c > -900)
fig, ax = plt.subplots(1, 3, figsize=(11, 4.3))
ax[0].imshow(win(out_c), cmap="gray"); ax[0].set_title("Coupe falsifiée (copier-déplacer)")
ax[1].imshow(win(out_c), cmap="gray"); ax[1].contour(m_c, [0.5], colors=[C2], linewidths=1.2); ax[1].set_title("Vérité terrain (destination)")
ax[2].imshow(win(out_c), cmap="gray"); ax[2].imshow(np.ma.masked_where(~dup, dup), cmap="autumn", alpha=0.7)
ax[2].set_title("Blocs 8×8 identiques détectés\n(source et destination)")
noaxis(*ax)
plt.tight_layout(); plt.savefig(OUT + "f9_copier_deplacer.png", dpi=200); plt.close()

import json
json.dump(dict(source=SRC, fgsm=FGSM_INFO, sigma_dose=[float(s) for s in sig], doses=doses.tolist()),
          open(OUT + "fig_ch1_info.json", "w"), indent=1)
print("ok", SRC, FGSM_INFO)

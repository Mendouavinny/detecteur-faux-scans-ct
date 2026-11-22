"""Simulateur de coupes CT thoraciques + générateur de falsifications (preuve de concept).

Chaîne physique : carte HU -> coefficients d'atténuation -> projection de Radon ->
bruit de Poisson (dose I0) -> rétroprojection filtrée (FBP). Le bruit obtenu est donc
corrélé spatialement (texture « CT » réelle), ce qui est la base des indices forensiques.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.transform import radon, iradon
import warnings

warnings.filterwarnings("ignore")
N = 256
MU_W = 0.02  # mm^-1 approx (eau) ; pixel = 1.4 mm
PIX = 1.4
THETA = np.linspace(0, 180, 360, endpoint=False)


def ellipse(shape, cy, cx, ry, rx, ang=0.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    y, x = yy - cy, xx - cx
    c, s = np.cos(ang), np.sin(ang)
    xr, yr = c * x + s * y, -s * x + c * y
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def nodule_profile(r, rng, spic=True):
    """Nodule pulmonaire 2D : profil lisse + spiculations éventuelles (en HU relatifs)."""
    size = int(2 * r + 9)
    yy, xx = np.mgrid[:size, :size] - size // 2
    rho = np.sqrt(yy ** 2 + xx ** 2)
    ang = np.arctan2(yy, xx)
    rr = r * (1 + (0.15 * np.sin(rng.integers(3, 8) * ang + rng.uniform(0, 6)) if spic else 0))
    prof = 1.0 / (1 + np.exp((rho - rr) / 0.8))
    return prof  # 0..1


def make_phantom(rng, with_nodule=False):
    """Retourne une carte HU (N x N) et la liste des nodules (cy, cx, r)."""
    hu = np.full((N, N), -1000.0)
    by, bx = N / 2 + rng.normal(0, 3), N / 2 + rng.normal(0, 3)
    ry, rx = rng.uniform(78, 90), rng.uniform(105, 118)
    body = ellipse(hu.shape, by, bx, ry, rx)
    hu[body] = -90 + rng.normal(0, 5)  # graisse sous-cutanée
    muscle = ellipse(hu.shape, by, bx, ry - 8, rx - 8)
    hu[muscle] = 45 + rng.normal(0, 5)
    # poumons
    lungs = np.zeros_like(body)
    for side in (-1, 1):
        lungs |= ellipse(hu.shape, by - 4 + rng.normal(0, 2), bx + side * rng.uniform(44, 52),
                         rng.uniform(55, 66), rng.uniform(30, 38), ang=side * rng.uniform(0.05, 0.2))
    tex = ndi.gaussian_filter(rng.normal(0, 1, hu.shape), 2.0)
    hu[lungs] = -860 + 25 * tex[lungs]
    # vaisseaux : segments aléatoires lissés
    ves = np.zeros(hu.shape, np.float32)
    for _ in range(rng.integers(25, 45)):
        p = (int(rng.uniform(40, 216)), int(rng.uniform(40, 216)))
        q = (p[0] + int(rng.normal(0, 18)), p[1] + int(rng.normal(0, 18)))
        cv2.line(ves, p, q, 1.0, int(rng.integers(1, 3)))
    ves = ndi.gaussian_filter(ves, 0.8) * lungs
    hu += ves * rng.uniform(600, 850)
    # coeur, aorte, vertèbre, côtes
    hu[ellipse(hu.shape, by + 12, bx + 8, 30, 34)] = 40 + rng.normal(0, 4)
    hu[ellipse(hu.shape, by - 20, bx + 12, 9, 9)] = 45
    hu[ellipse(hu.shape, by + ry - 26, bx, 13, 12)] = 700 + rng.normal(0, 30)
    hu[ellipse(hu.shape, by + ry - 26, bx, 8, 7)] = 250
    for k in range(8):
        a = np.pi * (0.15 + 0.7 * k / 7)
        for side in (-1, 1):
            cy = by - (ry - 6) * np.cos(a) * 0.95
            cx = bx + side * (rx - 6) * np.sin(a)
            hu[ellipse(hu.shape, cy, cx, 3.5, 5)] = 650
    nods = []
    if with_nodule:
        lung_idx = np.argwhere(ndi.binary_erosion(lungs, iterations=12))
        for _ in range(rng.integers(1, 3)):
            cy, cx = lung_idx[rng.integers(len(lung_idx))]
            r = rng.uniform(3, 8)
            prof = nodule_profile(r, rng)
            h = prof.shape[0] // 2
            sl = (slice(cy - h, cy - h + prof.shape[0]), slice(cx - h, cx - h + prof.shape[1]))
            hu[sl] = hu[sl] * (1 - prof) + (rng.uniform(10, 60)) * prof
            nods.append((int(cy), int(cx), float(r)))
    return hu, lungs, nods


def acquire(hu, rng, I0, kernel):
    """Acquisition CT simulée : Radon + bruit de Poisson + FBP. Retourne une image en HU (int16)."""
    mu = np.clip(MU_W * (1 + hu / 1000.0), 0, None) * PIX
    circ = ellipse(hu.shape, N / 2 - 0.5, N / 2 - 0.5, N / 2 - 1, N / 2 - 1)
    mu = mu * circ
    sino = radon(mu, THETA, circle=True)
    counts = rng.poisson(I0 * np.exp(-sino)).astype(np.float64)
    counts = np.maximum(counts, 1)
    sino_n = -np.log(counts / I0)
    rec = iradon(sino_n, THETA, circle=True, filter_name=kernel) / PIX
    out = (rec / MU_W - 1) * 1000.0
    out[~circ] = -1024
    return np.clip(np.round(out), -1024, 3071).astype(np.int16)


# ----------------------------------------------------------------------------------
# Estimation de bruit et falsifications (côté attaquant)
# ----------------------------------------------------------------------------------

def mad_sigma(img, mask):
    r = img.astype(np.float32) - ndi.median_filter(img.astype(np.float32), 3)
    v = r[mask]
    return 1.4826 * np.median(np.abs(v - np.median(v))) * 1.2


def blend_patch(img, patch, cy, cx, rng, sigma_noise):
    """Colle un patch (HU float) avec fenêtre gaussienne + bruit blanc ajusté (camouflage)."""
    out = img.astype(np.float32).copy()
    h, w = patch.shape
    y0, x0 = cy - h // 2, cx - w // 2
    yy, xx = np.mgrid[:h, :w]
    win = np.exp(-(((yy - h / 2) / (h / 3.2)) ** 2 + ((xx - w / 2) / (w / 3.2)) ** 2))
    region = out[y0:y0 + h, x0:x0 + w]
    noisy = patch + rng.normal(0, sigma_noise, patch.shape)
    out[y0:y0 + h, x0:x0 + w] = region * (1 - win) + noisy * win
    mask = np.zeros(img.shape, bool)
    mask[y0:y0 + h, x0:x0 + w] = win > 0.25
    return np.clip(np.round(out), -1024, 3071).astype(np.int16), mask


def tamper_injection(img, lungs, rng):
    idx = np.argwhere(ndi.binary_erosion(lungs, iterations=16))
    cy, cx = idx[rng.integers(len(idx))]
    r = rng.uniform(5, 9)
    prof = nodule_profile(r, rng)
    pad = 6
    prof = np.pad(prof, pad)
    h = prof.shape[0]
    base = ndi.gaussian_filter(img.astype(np.float32), 1.5)[cy - h // 2: cy - h // 2 + h, cx - h // 2: cx - h // 2 + h]
    patch = base * (1 - prof) + rng.uniform(10, 50) * prof
    sig = mad_sigma(img, lungs)
    return blend_patch(img, patch, cy, cx, rng, sig)


def tamper_removal(img, lungs, nods, rng):
    cy, cx, r = nods[0]
    h = int(2 * r + 14)
    m = np.zeros(img.shape, np.uint8)
    cv2.circle(m, (cx, cy), int(r + 3), 1, -1)
    win = ((img.astype(np.float32) + 1024) / 4095 * 255).astype(np.uint8)
    # inpainting en HU via image 16 bits normalisée -> on inpaint sur float32 via cv2 (8 bits) puis reprojection
    f = img.astype(np.float32)
    lo, hi = -1024.0, 400.0
    u8 = np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(u8, m, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255 * (hi - lo) + lo
    patch = ndi.gaussian_filter(inp, 1.0)[cy - h // 2: cy - h // 2 + h, cx - h // 2: cx - h // 2 + h]
    sig = mad_sigma(img, lungs)
    return blend_patch(img, patch, cy, cx, rng, sig)


def tamper_copymove(img, lungs, rng):
    idx = np.argwhere(ndi.binary_erosion(lungs, iterations=20))
    s = int(rng.integers(20, 30))
    for _ in range(50):
        a = idx[rng.integers(len(idx))]
        b = idx[rng.integers(len(idx))]
        if np.abs(a - b).max() > s + 4:
            break
    out = img.copy()
    src = img[a[0] - s // 2: a[0] - s // 2 + s, a[1] - s // 2: a[1] - s // 2 + s]
    out[b[0] - s // 2: b[0] - s // 2 + s, b[1] - s // 2: b[1] - s // 2 + s] = src
    mask = np.zeros(img.shape, bool)
    mask[b[0] - s // 2: b[0] - s // 2 + s, b[1] - s // 2: b[1] - s // 2 + s] = True
    return out, mask


def tamper_noise(img, rng):
    """Injection de bruit globale (perturbation de type adversariale non ciblée)."""
    sig = rng.uniform(4, 12)
    n = rng.normal(0, sig, img.shape)
    body = img > -900
    out = img.astype(np.float32) + n * body
    return np.clip(np.round(out), -1024, 3071).astype(np.int16), body


class PPCAGenerator:
    """Modèle génératif linéaire-gaussien (PCA probabiliste) appris sur des patchs nodulaires authentiques.
    Sert de *proxy* aux générateurs profonds (GAN/diffusion) : il produit des contenus plausibles mais
    dont la signature de bruit n'est pas celle de la chaîne d'acquisition."""

    def __init__(self, k=24):
        self.k = k

    def fit(self, patches):
        X = patches.reshape(len(patches), -1).astype(np.float64)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[: self.k]
        self.std = S[: self.k] / np.sqrt(len(X) - 1)
        self.shape = patches.shape[1:]
        return self

    def sample(self, rng):
        z = rng.normal(0, 1, self.k) * self.std
        return (self.mu + z @ self.W).reshape(self.shape)


def tamper_generative(img, lungs, gen, rng):
    idx = np.argwhere(ndi.binary_erosion(lungs, iterations=18))
    cy, cx = idx[rng.integers(len(idx))]
    patch = gen.sample(rng)
    sig = mad_sigma(img, lungs)
    return blend_patch(img, patch, cy, cx, rng, sig)

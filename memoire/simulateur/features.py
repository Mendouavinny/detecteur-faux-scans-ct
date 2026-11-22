"""Extraction de caractéristiques forensiques (résidus de bruit, spectre, cohérence locale, duplication)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew

B = 16
K_SRM3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4.0
K_KV = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0


def residuals(img):
    f = img.astype(np.float32)
    return {
        "med": f - ndi.median_filter(f, 3),
        "srm3": ndi.convolve(f, K_SRM3),
        "kv": ndi.convolve(f, K_KV),
        "dx": np.diff(f, axis=1, append=f[:, -1:]),
    }


def ac1(r):
    a = r - r.mean()
    d = (a * a).mean() + 1e-6
    return float((a[:, 1:] * a[:, :-1]).mean() / d), float((a[1:] * a[:-1]).mean() / d)


def band_energy(r, nb=6):
    F = np.abs(np.fft.fftshift(np.fft.fft2(r - r.mean()))) ** 2
    h, w = r.shape
    yy, xx = np.mgrid[:h, :w]
    rad = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    e = np.array([F[(rad >= i / nb) & (rad < (i + 1) / nb)].mean() for i in range(nb)])
    return e / (e.sum() + 1e-9)


def lung_blocks(lungs):
    er = ndi.binary_erosion(lungs, iterations=2)
    out = []
    for y in range(0, lungs.shape[0] - B + 1, B // 2):
        for x in range(0, lungs.shape[1] - B + 1, B // 2):
            if er[y:y + B, x:x + B].all():
                out.append((y, x))
    return out


def block_feats(img, R, blocks):
    """Descripteurs locaux par bloc : niveau de bruit, autocorrélation, ratio spectral haut/moyen."""
    rows = []
    for y, x in blocks:
        r = R["med"][y:y + B, x:x + B]
        k = R["kv"][y:y + B, x:x + B]
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        a1x, a1y = ac1(k)
        e = band_energy(k, 4)
        rows.append([sig, a1x, a1y, e[3] / (e[1] + 1e-9), k.std(), kurtosis(k.ravel())])
    return np.array(rows, np.float32)


def duplicate_map(img, body, b=8):
    """Détection de blocs strictement identiques (copier-déplacer) par hachage."""
    from numpy.lib.stride_tricks import sliding_window_view
    dup = np.zeros(img.shape, bool)
    W = sliding_window_view(np.ascontiguousarray(img), (b, b))
    H, Wd = W.shape[:2]
    flat = np.ascontiguousarray(W.reshape(H * Wd, b * b))
    yy, xx = np.divmod(np.arange(H * Wd), Wd)
    ok = body[yy + b // 2, xx + b // 2] & (flat.std(1) >= 3)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return dup
    v = flat[idx].view(np.dtype((np.void, flat.dtype.itemsize * b * b))).ravel()
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    for g in np.where(cnt > 1)[0]:
        mem = idx[inv == g]
        ys, xs = yy[mem], xx[mem]
        if (np.abs(ys - ys[0]).max() > b) or (np.abs(xs - xs[0]).max() > b):
            for y, x in zip(ys, xs):
                dup[y:y + b, x:x + b] = True
    return dup


def robust_z(v):
    med = np.median(v, 0)
    mad = 1.4826 * np.median(np.abs(v - med), 0) + 1e-6
    return (v - med) / mad


def image_features(img, lungs):
    R = residuals(img)
    body = img > -900
    feats = []
    for k in ("med", "srm3", "kv", "dx"):
        v = R[k][body]
        feats += [v.std(), kurtosis(v), skew(v)]
    lv = R["kv"][lungs]
    feats += list(ac1(np.where(lungs, R["kv"], 0)))
    # spectre du résidu dans une fenêtre pulmonaire centrale
    ys, xs = np.where(ndi.binary_erosion(lungs, iterations=4))
    cy, cx = int(np.median(ys)), int(np.median(xs))
    win = R["kv"][max(cy - 32, 0):cy + 32, max(cx - 32, 0):cx + 32]
    feats += list(band_energy(win, 6))
    blocks = lung_blocks(lungs)
    bf = block_feats(img, R, blocks)
    z = np.abs(robust_z(bf))
    feats += list(z.max(0)) + list(np.percentile(z, 95, axis=0)) + list(bf.std(0) / (np.abs(bf.mean(0)) + 1e-6))
    dup = duplicate_map(img, body)
    feats += [dup.mean() * 1000, float(dup.any())]
    return np.array(feats, np.float32), (blocks, bf, dup)


def intensity_features(img, lungs):
    """Ligne de base « contenu » : statistiques d'intensité HU uniquement."""
    v = img[lungs].astype(np.float32)
    b = img[img > -900].astype(np.float32)
    return np.array([v.mean(), v.std(), *np.percentile(v, [5, 25, 50, 75, 95]),
                     b.mean(), b.std(), *np.histogram(b, bins=12, range=(-1000, 1000), density=True)[0]], np.float32)

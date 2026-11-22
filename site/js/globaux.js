// Les 40 descripteurs de la coupe entière (features.image_features du carnet), utilisés par les méta-classifieurs
// de l'exécution v7. Même arithmétique en simple précision que NumPy, comme pour les descripteurs des patchs.
import { pw32, mean, std, kurtosis, median, erode, crop, C_MAD, E6, E9 } from "./forensique.js";

const f32 = Math.fround;

function skew(a) {
  const m = mean(a), x2 = new Float64Array(a.length), x3 = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) { const d = f32(a[i] - m); x2[i] = f32(d * d); x3[i] = f32(d * d * d); }
  const m2 = mean(x2), m3 = mean(x3);
  return m2 === 0 ? NaN : f32(m3 / f32(Math.pow(m2, 1.5)));
}

function ac1hw(r, h, w) {
  const m = mean(r), a = new Float64Array(h * w), sq = new Float64Array(h * w);
  for (let i = 0; i < h * w; i++) { a[i] = f32(r[i] - m); sq[i] = f32(a[i] * a[i]); }
  const d = f32(mean(sq) + E6);
  const px = new Float64Array(h * (w - 1)), py = new Float64Array((h - 1) * w);
  let k = 0;
  for (let y = 0; y < h; y++) for (let x = 1; x < w; x++) px[k++] = f32(a[y * w + x] * a[y * w + x - 1]);
  k = 0;
  for (let y = 1; y < h; y++) for (let x = 0; x < w; x++) py[k++] = f32(a[y * w + x] * a[(y - 1) * w + x]);
  return [f32(mean(px) / d), f32(mean(py) / d)];
}

// transformée de Fourier discrète d'une ligne (radix 2 si possible, sinon directe)
function tfd(re, im) {
  const n = re.length;
  if ((n & (n - 1)) === 0) {
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      for (let i = 0; i < n; i += len) for (let k = 0; k < len / 2; k++) {
        const cr = Math.cos(ang * k), ci = Math.sin(ang * k), a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
    return;
  }
  const r0 = Float64Array.from(re), i0 = Float64Array.from(im);
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < n; t++) { const a = (-2 * Math.PI * k * t) / n; sr += r0[t] * Math.cos(a) - i0[t] * Math.sin(a); si += r0[t] * Math.sin(a) + i0[t] * Math.cos(a); }
    re[k] = sr; im[k] = si;
  }
}

/** band_energy pour un bloc h × w quelconque (fftshift, anneaux de rayon normalisé par min(h, w) / 2). */
export function energieAnneaux(r, h, w, nb) {
  const m = mean(r), re = new Float64Array(h * w), im = new Float64Array(h * w);
  for (let i = 0; i < h * w; i++) re[i] = f32(r[i] - m);
  for (let y = 0; y < h; y++) {
    const a = re.slice(y * w, y * w + w), b = im.slice(y * w, y * w + w); tfd(a, b);
    re.set(a, y * w); im.set(b, y * w);
  }
  for (let x = 0; x < w; x++) {
    const a = new Float64Array(h), b = new Float64Array(h);
    for (let y = 0; y < h; y++) { a[y] = re[y * w + x]; b[y] = im[y * w + x]; }
    tfd(a, b);
    for (let y = 0; y < h; y++) { re[y * w + x] = a[y]; im[y * w + x] = b[y]; }
  }
  const sh = Math.floor(h / 2), sw = Math.floor(w / 2), ray = Math.min(h, w) / 2, vals = Array.from({ length: nb }, () => []);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const rad = Math.sqrt((y - h / 2) ** 2 + (x - w / 2) ** 2) / ray;
    let k = -1;
    for (let i = 0; i < nb; i++) if (rad >= i / nb && rad < (i + 1) / nb) { k = i; break; }
    if (k < 0) continue;
    const p = ((y - sh + h) % h) * w + ((x - sw + w) % w);
    const am = f32(Math.hypot(re[p], im[p]));
    vals[k].push(f32(am * am));
  }
  const e = vals.map((v) => (v.length ? mean(v) : NaN));
  const t = f32(pw32(e, 0, nb) + E9);
  return e.map((v) => f32(v / t));
}

function zRobuste(rows, nc) {
  const z = rows.map(() => new Array(nc));
  for (let c = 0; c < nc; c++) {
    const col = rows.map((r) => r[c]), med = median(col);
    const mad = f32(f32(C_MAD * median(col.map((v) => Math.abs(f32(v - med))))) + E6);
    rows.forEach((r, i) => { z[i][c] = Math.abs(f32(f32(r[c] - med) / mad)); });
  }
  return z;
}

function percentile95(col) {
  const s = Float64Array.from(col).sort(), n = s.length, v = 0.95 * (n - 1), lo = Math.floor(v), g = v - lo;
  if (lo + 1 >= n) return s[n - 1];
  const a = s[lo], b = s[lo + 1], d = b - a;
  return f32(g >= 0.5 ? b - d * (1 - g) : a + d * g);
}

/**
 * Descripteurs de la coupe. hu : coupe (Int16Array), lung : masque pulmonaire, R : résidus, dup : carte de duplication.
 * Renvoie 40 valeurs (NaN remplacés par 0, comme np.nan_to_num dans le carnet).
 */
export function descripteursCoupe(hu, h, w, lung, R, dup) {
  const feats = [];
  const corps = [];
  for (let i = 0; i < h * w; i++) if (hu[i] > -900) corps.push(i);
  for (const k of ["med", "srm3", "kv", "dx"]) {
    const v = new Float64Array(corps.length);
    for (let j = 0; j < corps.length; j++) v[j] = R[k][corps[j]];
    feats.push(std(v), kurtosis(v), skew(v));
  }
  const kvp = new Float64Array(h * w);
  for (let i = 0; i < h * w; i++) kvp[i] = lung[i] ? R.kv[i] : 0;
  feats.push(...ac1hw(kvp, h, w));
  // spectre du résidu dans une fenêtre pulmonaire centrale
  let er = lung;
  for (let k = 0; k < 4; k++) er = erode(er, h, w);
  const ys = [], xs = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (er[y * w + x]) { ys.push(y); xs.push(x); }
  {
    // exécution v7 : centre de l'image si le poumon érodé est vide
    const medi = (a) => { const s = Float64Array.from(a).sort(), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
    const cy = ys.length ? Math.trunc(medi(ys)) : Math.floor(h / 2), cx = ys.length ? Math.trunc(medi(xs)) : Math.floor(w / 2);
    const y0 = Math.max(cy - 32, 0), y1 = Math.min(cy + 32, h), x0 = Math.max(cx - 32, 0), x1 = Math.min(cx + 32, w);
    const hh = y1 - y0, ww = x1 - x0, win = new Float64Array(hh * ww);
    for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) win[y * ww + x] = R.kv[(y0 + y) * w + x0 + x];
    feats.push(...energieAnneaux(win, hh, ww, 6));
  }
  // blocs 16 × 16 au pas de 8 entièrement dans le poumon érodé
  const B = 16;
  let er2 = lung;
  for (let k = 0; k < 2; k++) er2 = erode(er2, h, w);
  const bf = [];
  for (let y = 0; y + B <= h; y += B / 2) for (let x = 0; x + B <= w; x += B / 2) {
    let ok = true;
    for (let i = 0; i < B && ok; i++) for (let j = 0; j < B; j++) if (!er2[(y + i) * w + x + j]) { ok = false; break; }
    if (!ok) continue;
    const r = crop(R.med, w, y, x, B), k = crop(R.kv, w, y, x, B), mr = median(r);
    const sig = f32(C_MAD * median(r.map((v) => Math.abs(f32(v - mr)))));
    const [a1x, a1y] = ac1hw(k, B, B), e = energieAnneaux(k, B, B, 4);
    bf.push([sig, a1x, a1y, f32(e[3] / f32(e[1] + E9)), std(k), kurtosis(k)]);
  }
  if (!bf.length) bf.push([0, 0, 0, 0, 0, 0]);                // exécution v7 : un bloc nul si aucun bloc
  {
    const z = zRobuste(bf, 6);
    for (let c = 0; c < 6; c++) feats.push(Math.max(...z.map((r) => r[c])));
    for (let c = 0; c < 6; c++) feats.push(percentile95(z.map((r) => r[c])));
    for (let c = 0; c < 6; c++) {             // bf.std(0) / (|bf.mean(0)| + 1e-6) : sommes séquentielles sur l'axe 0
      let s = 0;
      for (const r of bf) s = f32(s + r[c]);
      const m = f32(s / bf.length);
      let q = 0;
      for (const r of bf) { const d = f32(r[c] - m); q = f32(q + f32(d * d)); }
      feats.push(f32(f32(Math.sqrt(f32(q / bf.length))) / f32(Math.abs(m) + E6)));
    }
  }
  let n = 0;
  for (let i = 0; i < h * w; i++) n += dup[i];
  feats.push(f32((n / (h * w)) * 1000), n ? 1 : 0);
  return feats.map((v) => (Number.isFinite(f32(v)) ? f32(v) : 0));
}

/** agg du carnet : maximum, moyenne des 3 et des 10 plus grandes valeurs, nombre au-dessus de 0,5. */
export function agg(p) {
  const s = Array.from(p).sort((a, b) => b - a);
  if (!s.length) return [0, 0, 0, 0];
  const m = (k) => { const t = s.slice(0, k); return pw64(t, 0, t.length) / t.length; };
  return [s[0], m(3), m(10), s.filter((v) => v > 0.5).length];
}

// somme « pairwise » de NumPy en double précision (moyenne des 3 et des 10 plus grandes valeurs)
function pw64(a, off, n) {
  if (n < 8) { let r = 0; for (let i = 0; i < n; i++) r += a[off + i]; return r; }
  if (n <= 128) {
    const r = a.slice(off, off + 8);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[off + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[off + i];
    return res;
  }
  let n2 = Math.floor(n / 2); n2 -= n2 % 8;
  return pw64(a, off, n2) + pw64(a, off + n2, n - n2);
}

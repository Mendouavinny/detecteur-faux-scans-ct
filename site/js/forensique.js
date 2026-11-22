// Branche forensique du détecteur, portée en JavaScript (navigateur, sans serveur).
// Reprise ligne à ligne de detecteur.py et features.py (application du mémoire) :
// lecture DICOM → unités Hounsfield → masque pulmonaire → patchs 64 × 64 (pas 32) → 27 descripteurs du bruit
// → classifieur HistGradientBoosting entraîné (arbres exportés) → score de la coupe = moyenne des 3 patchs les plus suspects.

export const P = 64, S = 32;

// ------------------------------------------------------------------ lecture DICOM (non compressé)
const TS_IMPLICIT = "1.2.840.10008.1.2";
const TS_EXPLICIT = "1.2.840.10008.1.2.1";
const LONG_VR = new Set(["OB", "OW", "OF", "SQ", "UT", "UN", "UC", "UR", "OD", "OL", "OV", "SV", "UV"]);

export function parseDicom(buf) {
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  const txt = (o, n) => {
    let s = "";
    for (let i = 0; i < n; i++) s += String.fromCharCode(u8[o + i]);
    return s.replace(/[\0 ]+$/, "").trim();
  };
  let off = 0;
  if (buf.byteLength > 132 && txt(128, 4) === "DICM") off = 132;
  let ts = TS_IMPLICIT;
  const el = {};
  const isVR = (o) => {
    const a = u8[o], b = u8[o + 1];
    return a >= 65 && a <= 90 && b >= 65 && b <= 90;
  };
  // groupe méta (toujours en VR explicite)
  while (off + 8 <= buf.byteLength && dv.getUint16(off, true) === 0x0002) {
    const g = dv.getUint16(off, true), e = dv.getUint16(off + 2, true);
    const vr = txt(off + 4, 2);
    let len, start;
    if (LONG_VR.has(vr)) { len = dv.getUint32(off + 8, true); start = off + 12; }
    else { len = dv.getUint16(off + 6, true); start = off + 8; }
    if (g === 2 && e === 0x10) ts = txt(start, len);
    off = start + len;
  }
  if (ts !== TS_IMPLICIT && ts !== TS_EXPLICIT) {
    throw new Error("Syntaxe de transfert non prise en charge (" + ts + ") : il faut un DICOM non compressé, " +
      "tel qu'il sort du scanner. Une image compressée a perdu une partie de son bruit.");
  }
  const explicit = ts === TS_EXPLICIT;

  function readElements(end, depth) {
    while (off + 8 <= end) {
      const g = dv.getUint16(off, true), e = dv.getUint16(off + 2, true);
      if (g === 0xfffe) {                       // délimiteurs d'items et de séquences
        const len = dv.getUint32(off + 4, true);
        off += 8;
        if (e === 0xe00d || e === 0xe0dd) return;
        if (e === 0xe000) {
          if (len === 0xffffffff) readElements(end, depth + 1);
          else off += len;
        }
        continue;
      }
      let vr = null, len, start;
      if (explicit && isVR(off + 4)) {
        vr = txt(off + 4, 2);
        if (LONG_VR.has(vr)) { len = dv.getUint32(off + 8, true); start = off + 12; }
        else { len = dv.getUint16(off + 6, true); start = off + 8; }
      } else { len = dv.getUint32(off + 4, true); start = off + 8; }
      off = start;
      if (len === 0xffffffff) {                 // séquence (ou pixels encapsulés) de longueur indéfinie
        if (g === 0x7fe0 && e === 0x0010) throw new Error("Pixels encapsulés (DICOM compressé) : non pris en charge.");
        readElements(end, depth + 1);
        continue;
      }
      if (depth === 0) el[(g << 16) | e] = { start, len, vr };
      off = start + len;
    }
  }
  readElements(buf.byteLength, 0);

  const u16 = (tag) => (el[tag] ? dv.getUint16(el[tag].start, true) : undefined);
  const str = (tag) => (el[tag] ? txt(el[tag].start, el[tag].len) : undefined);
  const rows = u16(0x00280010), cols = u16(0x00280011);
  const bits = u16(0x00280100) || 16, rep = u16(0x00280103) || 0;
  const pix = el[0x7fe00010];
  if (!rows || !cols || !pix) throw new Error("Ce fichier ne contient pas d'image DICOM lisible.");
  if (bits !== 16) throw new Error("Seules les images codées sur 16 bits sont prises en charge.");
  const slope = parseFloat((str(0x00281053) || "1").split("\\")[0]) || 1;
  const inter = parseFloat((str(0x00281052) || "0").split("\\")[0]) || 0;
  const n = rows * cols;
  const hu = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const v = rep ? dv.getInt16(pix.start + 2 * i, true) : dv.getUint16(pix.start + 2 * i, true);
    const x = Math.fround(Math.fround(v * Math.fround(slope)) + Math.fround(inter));   // calcul float32, comme NumPy
    hu[i] = Math.trunc(x);                                                            // astype(int16) : troncature
  }
  return {
    rows, cols, hu, ts, slope, inter,
    modality: str(0x00080060), patient: str(0x00100020), manufacturer: str(0x00080070),
    kernel: str(0x00181210), kvp: str(0x00180060), thickness: str(0x00180050),
  };
}

// ------------------------------------------------------------------ morphologie (scipy.ndimage, connexité 4)
function label4(mask, h, w) {
  const lab = new Int32Array(h * w);
  let n = 0;
  const stack = new Int32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    if (!mask[i] || lab[i]) continue;
    n++;
    let sp = 0;
    stack[sp++] = i; lab[i] = n;
    while (sp) {
      const p = stack[--sp], y = (p / w) | 0, x = p - y * w;
      if (x > 0 && mask[p - 1] && !lab[p - 1]) { lab[p - 1] = n; stack[sp++] = p - 1; }
      if (x < w - 1 && mask[p + 1] && !lab[p + 1]) { lab[p + 1] = n; stack[sp++] = p + 1; }
      if (y > 0 && mask[p - w] && !lab[p - w]) { lab[p - w] = n; stack[sp++] = p - w; }
      if (y < h - 1 && mask[p + w] && !lab[p + w]) { lab[p + w] = n; stack[sp++] = p + w; }
    }
  }
  return { lab, n };
}

function dilate(m, h, w) {           // élément structurant en croix, bord = 0
  const o = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    o[p] = m[p] || (x > 0 && m[p - 1]) || (x < w - 1 && m[p + 1]) || (y > 0 && m[p - w]) || (y < h - 1 && m[p + w]) ? 1 : 0;
  }
  return o;
}

export function erode(m, h, w) {            // bord = 0 (comme binary_erosion de SciPy)
  const o = new Uint8Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = y * w + x;
    o[p] = m[p] && x > 0 && m[p - 1] && x < w - 1 && m[p + 1] && y > 0 && m[p - w] && y < h - 1 && m[p + w] ? 1 : 0;
  }
  return o;
}

function fillHoles(m, h, w) {
  const bg = new Uint8Array(h * w);
  for (let i = 0; i < h * w; i++) bg[i] = m[i] ? 0 : 1;
  const out = new Uint8Array(h * w).fill(1);
  const stack = [];
  const push = (p) => { if (bg[p] && out[p]) { out[p] = 0; stack.push(p); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop(), y = (p / w) | 0, x = p - y * w;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w);
    if (y < h - 1) push(p + w);
  }
  return out;
}

export function lungMask(hu, h, w) {
  let m = new Uint8Array(h * w);
  for (let i = 0; i < h * w; i++) m[i] = hu[i] < -400 ? 1 : 0;
  let { lab, n } = label4(m, h, w);
  const border = new Set();
  for (let x = 0; x < w; x++) { border.add(lab[x]); border.add(lab[(h - 1) * w + x]); }
  for (let y = 0; y < h; y++) { border.add(lab[y * w]); border.add(lab[y * w + w - 1]); }
  for (let i = 0; i < h * w; i++) if (m[i] && border.has(lab[i])) m[i] = 0;
  ({ lab, n } = label4(m, h, w));
  if (n === 0) return m;
  const sizes = new Float64Array(n + 1);
  for (let i = 0; i < h * w; i++) if (lab[i]) sizes[lab[i]]++;
  const order = Array.from({ length: n }, (_, i) => i + 1).sort((a, b) => sizes[b] - sizes[a] || a - b).slice(0, 2);
  const keep = new Set(order.filter((k) => sizes[k] > 1500));
  for (let i = 0; i < h * w; i++) m[i] = keep.has(lab[i]) ? 1 : 0;
  for (let k = 0; k < 3; k++) m = dilate(m, h, w);
  for (let k = 0; k < 3; k++) m = erode(m, h, w);
  return fillHoles(m, h, w);
}

// ------------------------------------------------------------------ résidus (mode « reflect » de SciPy)
const refl = (i, n) => (i < 0 ? -i - 1 : i >= n ? 2 * n - i - 1 : i);
const K_SRM3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]].map((r) => r.map((v) => v / 4));
const K_KV = [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]
  .map((r) => r.map((v) => Math.fround(v / 12)));

function convolve(img, h, w, K) {
  const r = (K.length - 1) / 2, out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) {
      const yy = refl(y - i, h) * w;              // convolution : noyau retourné (symétrique ici)
      for (let j = -r; j <= r; j++) s += K[i + r][j + r] * img[yy + refl(x - j, w)];
    }
    out[y * w + x] = s;
  }
  return out;
}

export function residuals(hu, h, w) {
  const med = new Float32Array(h * w), dx = new Float32Array(h * w), v = new Float64Array(9);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let k = 0;
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) v[k++] = hu[refl(y + i, h) * w + refl(x + j, w)];
    for (let a = 1; a < 9; a++) { const t = v[a]; let b = a - 1; while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; b--; } v[b + 1] = t; }
    const p = y * w + x;
    med[p] = hu[p] - v[4];
    dx[p] = x < w - 1 ? hu[p + 1] - hu[p] : 0;
  }
  return { med, srm3: convolve(hu, h, w, K_SRM3), kv: convolve(hu, h, w, K_KV), dx };
}

// ------------------------------------------------------------------ statistiques en simple précision
// NumPy calcule ces descripteurs en float32 (somme « pairwise ») : on reproduit ses arrondis pour que
// les arbres du classifieur prennent les mêmes décisions qu'en Python.
const f32 = Math.fround;
// NumPy 2 (NEP 50) convertit les constantes Python en float32 avant l'opération
export const C_MAD = f32(1.4826), E6 = f32(1e-6), E9 = f32(1e-9);
export function pw32(a, off, n) {
  if (n < 8) { let r = 0; for (let i = 0; i < n; i++) r = f32(r + a[off + i]); return r; }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let j = 0; j < 8; j++) r[j] = f32(a[off + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = f32(r[j] + a[off + i + j]);
    let res = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
    for (; i < n; i++) res = f32(res + a[off + i]);
    return res;
  }
  let n2 = Math.floor(n / 2); n2 -= n2 % 8;
  return f32(pw32(a, off, n2) + pw32(a, off + n2, n - n2));
}
export const mean = (a) => f32(pw32(a, 0, a.length) / a.length);
export function std(a) {
  const m = mean(a), x = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) { const d = f32(a[i] - m); x[i] = f32(d * d); }
  return f32(Math.sqrt(mean(x)));
}
export function kurtosis(a) {
  const m = mean(a), x2 = new Float64Array(a.length), x4 = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) { const d = f32(a[i] - m); x2[i] = f32(d * d); x4[i] = f32(d ** 4); }
  const m2 = mean(x2), m4 = mean(x4);
  return m2 === 0 ? NaN : f32(f32(m4 / f32(m2 * m2)) - 3);
}
export function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : f32(f32(s[n / 2 - 1] + s[n / 2]) / 2);
}

export function ac1(r, n) {                         // r : patch n × n (float32)
  const m = mean(r), a = new Float64Array(n * n), sq = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) { a[i] = f32(r[i] - m); sq[i] = f32(a[i] * a[i]); }
  const d = f32(mean(sq) + E6);
  const px = new Float64Array(n * (n - 1)), py = new Float64Array((n - 1) * n);
  let k = 0;
  for (let y = 0; y < n; y++) for (let x = 1; x < n; x++) px[k++] = f32(a[y * n + x] * a[y * n + x - 1]);
  k = 0;
  for (let y = 1; y < n; y++) for (let x = 0; x < n; x++) py[k++] = f32(a[y * n + x] * a[(y - 1) * n + x]);
  return [f32(mean(px) / d), f32(mean(py) / d)];
}

// FFT 2D (radix 2) pour l'énergie par anneaux de fréquence
function fft1(re, im, n) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

const RING = {};
function rings(n, nb) {
  const key = n + "_" + nb;
  if (RING[key]) return RING[key];
  const idx = new Int8Array(n * n).fill(-1);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const rad = Math.sqrt((y - n / 2) ** 2 + (x - n / 2) ** 2) / (n / 2);
    for (let i = 0; i < nb; i++) if (rad >= i / nb && rad < (i + 1) / nb) { idx[y * n + x] = i; break; }
  }
  return (RING[key] = idx);
}

export function bandEnergy(r, n, nb = 6) {
  const m = mean(r), re = new Float64Array(n * n), im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) re[i] = f32(r[i] - m);
  const rr = new Float64Array(n), ri = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { rr[x] = re[y * n + x]; ri[x] = im[y * n + x]; }
    fft1(rr, ri, n);
    for (let x = 0; x < n; x++) { re[y * n + x] = rr[x]; im[y * n + x] = ri[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { rr[y] = re[y * n + x]; ri[y] = im[y * n + x]; }
    fft1(rr, ri, n);
    for (let y = 0; y < n; y++) { re[y * n + x] = rr[y]; im[y * n + x] = ri[y]; }
  }
  const idx = rings(n, nb), h = n / 2, vals = Array.from({ length: nb }, () => []);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {      // ordre du tableau décalé (fftshift)
    const k = idx[y * n + x];
    if (k < 0) continue;
    const p = ((y + h) % n) * n + ((x + h) % n);
    const a = f32(Math.hypot(re[p], im[p]));
    vals[k].push(f32(a * a));
  }
  const e = vals.map((v) => mean(v));
  const t = f32(pw32(e, 0, nb) + E9);
  return e.map((v) => f32(v / t));
}

// ------------------------------------------------------------------ copier-déplacer : fenêtres 8 × 8 identiques
export function duplicateMap(hu, h, w, b = 8) {
  const dup = new Uint8Array(h * w);
  const H = h - b + 1, W = w - b + 1;
  const buckets = new Map();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (!(hu[(y + b / 2) * w + x + b / 2] > -900)) continue;
    let s = 0, s2 = 0, h1 = 2166136261, h2 = 5381;
    for (let i = 0; i < b; i++) {
      const o = (y + i) * w + x;
      for (let j = 0; j < b; j++) {
        const v = hu[o + j];
        s += v; s2 += v * v;
        h1 = Math.imul(h1 ^ (v & 0xffff), 16777619);
        h2 = (Math.imul(h2, 33) + v) | 0;
      }
    }
    const m = s / (b * b);
    if (Math.sqrt(Math.max(0, s2 / (b * b) - m * m)) < 3) continue;
    const key = (h1 >>> 0) * 4294967296 + (h2 >>> 0);
    const l = buckets.get(key);
    if (l) l.push(y * W + x); else buckets.set(key, [y * W + x]);
  }
  const same = (a, c) => {
    const ya = (a / W) | 0, xa = a % W, yc = (c / W) | 0, xc = c % W;
    for (let i = 0; i < b; i++) for (let j = 0; j < b; j++) if (hu[(ya + i) * w + xa + j] !== hu[(yc + i) * w + xc + j]) return false;
    return true;
  };
  const groups = [];
  for (const l of buckets.values()) {
    if (l.length < 2) continue;
    const used = new Uint8Array(l.length);
    for (let i = 0; i < l.length; i++) {
      if (used[i]) continue;
      const g = [l[i]];
      for (let j = i + 1; j < l.length; j++) if (!used[j] && same(l[i], l[j])) { used[j] = 1; g.push(l[j]); }
      if (g.length < 2) continue;
      g.sort((a, c) => a - c);                       // premier membre = premier dans l'ordre de balayage
      const y0 = (g[0] / W) | 0, x0 = g[0] % W;
      let far = false;
      for (const q of g) if (Math.abs(((q / W) | 0) - y0) > b || Math.abs((q % W) - x0) > b) far = true;
      if (!far) continue;
      groups.push(g.map((q) => [(q / W) | 0, q % W]));
      for (const q of g) {
        const yy = (q / W) | 0, xx = q % W;
        for (let i = 0; i < b; i++) dup.fill(1, (yy + i) * w + xx, (yy + i) * w + xx + b);
      }
    }
  }
  return { dup, groups };
}

// ------------------------------------------------------------------ patchs et descripteurs (27 par patch)
export function gridCoords(lung, h, w) {
  const out = [];
  for (let y = 0; y + P <= h; y += S) for (let x = 0; x + P <= w; x += S) {
    let s = 0;
    for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) s += lung[(y + i) * w + x + j];
    if (s / (P * P) > 0.5) out.push([y, x]);
  }
  return out;
}

export function crop(a, w, y, x, n) {
  const o = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) o[i * n + j] = a[(y + i) * w + x + j];
  return o;
}

export const NOMS = ["σ du bruit (MAD)", "autocorrélation x", "autocorrélation y", "rapport hautes / moyennes fréquences",
  "écart-type résidu KV", "kurtosis résidu KV", "énergie anneau 1", "énergie anneau 2", "énergie anneau 3",
  "énergie anneau 4", "énergie anneau 5", "énergie anneau 6", "intensité moyenne", "écart-type intensité"];

export function patchFeatures(hu, h, w, lung, R) {
  const coords = gridCoords(lung, h, w);
  if (!coords.length) return { coords, F: [], dup: null };
  const raw = coords.map(([y, x]) => {
    const r = crop(R.med, w, y, x, P), k = crop(R.kv, w, y, x, P), blk = crop(hu, w, y, x, P);
    const mr = median(r);
    const sig = f32(C_MAD * median(r.map((v) => Math.abs(f32(v - mr)))));
    const [a1x, a1y] = ac1(k, P), e = bandEnergy(k, P, 6);
    return [sig, a1x, a1y, f32(e[5] / f32(e[2] + E9)), std(k), kurtosis(k), ...e, mean(blk), std(blk)];
  });
  const z = [];
  for (let c = 0; c < 12; c++) {
    const col = raw.map((r) => r[c]), med = median(col);
    const mad = f32(f32(C_MAD * median(col.map((v) => Math.abs(f32(v - med))))) + E6);
    z.push(col.map((v) => Math.abs(f32(f32(v - med) / mad))));
  }
  const { dup, groups } = duplicateMap(hu, h, w);
  const F = raw.map((r, i) => {
    const [y, x] = coords[i];
    let s = 0;
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) s += dup[(y + a) * w + x + b];
    const f = [...r, ...z.map((col) => col[i]), Math.fround(s / (P * P))];
    return f.map((v) => (Number.isFinite(v) ? v : 0));
  });
  return { coords, F, dup, groups, raw, z };
}

// ------------------------------------------------------------------ classifieur (arbres HistGradientBoosting exportés)
export function hgbProba(model, f) {
  let raw = model.baseline;
  for (const [feat, thr, left, right, leaf, val, missLeft] of model.trees) {
    let nd = 0;
    while (!leaf[nd]) {
      const v = f[feat[nd]];
      nd = Number.isNaN(v) ? (missLeft[nd] ? left[nd] : right[nd]) : v <= thr[nd] ? left[nd] : right[nd];
    }
    raw += val[nd];
  }
  return 1 / (1 + Math.exp(-raw));
}

// ------------------------------------------------------------------ analyse complète d'une coupe
export function analyser(hu, h, w, model, onStep = () => {}) {
  const t0 = performance.now();
  onStep("Masque pulmonaire");
  const lung = lungMask(hu, h, w);
  onStep("Résidus du bruit");
  const R = residuals(hu, h, w);
  onStep("Descripteurs des patchs");
  const { coords, F, dup, groups, raw, z } = patchFeatures(hu, h, w, lung, R);
  if (!coords.length) return { ok: false, lung, R, message: "Aucun patch pulmonaire exploitable dans cette coupe." };
  onStep("Classifieur forensique");
  const p = F.map((f) => hgbProba(model, f));
  const top = [...p].sort((a, b) => b - a);
  const score = top.slice(0, 3).reduce((a, b) => a + b, 0) / Math.min(3, top.length);
  // cartes par pixel (moyenne des patchs qui se recouvrent)
  const acc = new Float32Array(h * w), cnt = new Float32Array(h * w), accZ = new Float32Array(h * w);
  coords.forEach(([y, x], i) => {
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const q = (y + a) * w + x + b;
      acc[q] += p[i]; accZ[q] += z[0][i]; cnt[q]++;
    }
  });
  for (let q = 0; q < h * w; q++) if (cnt[q]) { acc[q] /= cnt[q]; accZ[q] /= cnt[q]; }
  let dupPx = 0;
  for (let q = 0; q < h * w; q++) dupPx += dup[q];
  return {
    ok: true, lung, R, coords, F, raw, z, p, score, carte: acc, carteZ: accZ, dup, groups, dupPx,
    duree: performance.now() - t0,
  };
}

// ------------------------------------------------------------------ carte du niveau local du bruit (visualisation)
// Médiane locale de |résidu KV| dans une fenêtre 11 × 11, divisée par sa médiane sur le poumon de la même coupe.
// Rapport < 1 : bruit manquant (région lissée ou générée) ; > 1 : bruit en excès (bruit ajouté).
// Cette carte sert à voir le bruit ; la décision reste celle du classifieur.
export function carteBruit(R, lung, h, w, demi = 5, pas = 2) {
  const ratio = new Float32Array(h * w).fill(1);
  const vals = [], pos = [], buf = new Float64Array((2 * demi + 1) ** 2);
  for (let y = demi; y < h - demi; y += pas) for (let x = demi; x < w - demi; x += pas) {
    if (!lung[y * w + x]) continue;
    let k = 0;
    for (let i = -demi; i <= demi; i++) for (let j = -demi; j <= demi; j++) buf[k++] = Math.abs(R.kv[(y + i) * w + x + j]);
    buf.sort();
    vals.push(buf[(k - 1) >> 1]); pos.push(y * w + x);
  }
  if (!vals.length) return { ratio, ref: 0 };
  const ref = Float64Array.from(vals).sort()[vals.length >> 1] || 1;
  pos.forEach((q, i) => {
    const r = vals[i] / ref;
    for (let a = 0; a < pas; a++) for (let b = 0; b < pas; b++) if (lung[q + a * w + b]) ratio[q + a * w + b] = r;
  });
  return { ratio, ref };
}

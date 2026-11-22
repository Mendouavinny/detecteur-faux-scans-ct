// Falsifications de démonstration, appliquées dans le navigateur à une coupe authentique (valeurs en HU).
// Elles reproduisent le principe des familles étudiées dans le mémoire, pas les générateurs entraînés.

function rng(seed) {                      // mulberry32 : tirages reproductibles
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function flou(hu, h, w, y0, y1, x0, x1, sigma) {
  // flou gaussien séparable limité à une fenêtre (bords répliqués)
  const rad = Math.ceil(3 * sigma), k = [];
  for (let i = -rad; i <= rad; i++) k.push(Math.exp(-(i * i) / (2 * sigma * sigma)));
  const s = k.reduce((a, b) => a + b, 0); k.forEach((v, i) => (k[i] = v / s));
  const H = y1 - y0, W = x1 - x0, tmp = new Float64Array(H * W), out = new Float64Array(H * W);
  const cl = (v, n) => Math.min(n - 1, Math.max(0, v));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let a = 0;
    for (let i = -rad; i <= rad; i++) a += k[i + rad] * hu[cl(y0 + y, h) * w + cl(x0 + x + i, w)];
    tmp[y * W + x] = a;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let a = 0;
    for (let i = -rad; i <= rad; i++) a += k[i + rad] * tmp[cl(y + i, H) * W + x];
    out[y * W + x] = a;
  }
  return out;
}

const masqueDoux = (d, r0, r1) => (d <= r0 ? 1 : d >= r1 ? 0 : 0.5 + 0.5 * Math.cos((Math.PI * (d - r0)) / (r1 - r0)));
const i16 = (v) => Math.max(-32768, Math.min(32767, Math.round(v)));

/** Point du poumon où un disque de rayon r tient entièrement, le plus proche du centre du plus grand poumon. */
export function cibleParDefaut(lung, h, w, r = 22) {
  let sy = 0, sx = 0, n = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w / 2; x++) if (lung[y * w + x]) { sy += y; sx += x; n++; }
  if (!n) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (lung[y * w + x]) { sy += y; sx += x; n++; }
  if (!n) return [h / 2, w / 2];
  const cy = sy / n, cx = sx / n;
  let best = null, bd = Infinity;
  for (let y = r; y < h - r; y += 3) for (let x = r; x < w - r; x += 3) {
    if (!lung[y * w + x]) continue;
    let ok = true;
    for (let a = 0; a < 16 && ok; a++) {
      const yy = Math.round(y + r * Math.sin((a * Math.PI) / 8)), xx = Math.round(x + r * Math.cos((a * Math.PI) / 8));
      if (!lung[yy * w + xx]) ok = false;
    }
    const d = (y - cy) ** 2 + (x - cx) ** 2;
    if (ok && d < bd) { bd = d; best = [y, x]; }
  }
  return best || [Math.round(cy), Math.round(cx)];
}

/** Nodule synthétique sans la texture du bruit : la signature des contenus générés (GAN, diffusion). */
export function noduleLisse(hu, h, w, cy, cx, r = 11) {
  const out = Int16Array.from(hu), R = Math.ceil(r * 2.2);
  const y0 = Math.max(0, cy - R), y1 = Math.min(h, cy + R + 1), x0 = Math.max(0, cx - R), x1 = Math.min(w, cx + R + 1);
  const lisse = flou(hu, h, w, y0, y1, x0, x1, 2.2), W = x1 - x0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const d = Math.hypot(y - cy, x - cx), m = masqueDoux(d, r * 1.35, r * 2.1);
    if (!m) continue;
    const blob = 820 / (1 + Math.exp((d - r) / 1.6));             // nodule : bord net mais régulier
    const v = lisse[(y - y0) * W + x - x0] + blob;
    out[y * w + x] = i16((1 - m) * hu[y * w + x] + m * Math.min(v, 80));
  }
  return out;
}

/** Copier-déplacer : une région de l'autre poumon recopiée à l'identique (bruit dupliqué). */
export function copierDeplacer(hu, h, w, cy, cx, lung, c = 40) {
  const out = Int16Array.from(hu);
  let sy = cy, sx = w - cx;                                        // région symétrique, dans l'autre poumon
  const dedans = (y, x) => y >= 0 && x >= 0 && y + c <= h && x + c <= w && lung[(y + c / 2) * w + x + c / 2];
  let ty = Math.round(cy - c / 2), tx = Math.round(cx - c / 2), py = Math.round(sy - c / 2), px = Math.round(sx - c / 2);
  if (!dedans(py, px) || Math.abs(px - tx) < c) { py = ty + (ty > h / 2 ? -70 : 70); px = tx; }
  py = Math.max(0, Math.min(h - c, py)); px = Math.max(0, Math.min(w - c, px));
  ty = Math.max(0, Math.min(h - c, ty)); tx = Math.max(0, Math.min(w - c, tx));
  for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) out[(ty + i) * w + tx + j] = hu[(py + i) * w + px + j];
  return { hu: out, source: [py, px, c], cible: [ty, tx, c] };
}

/** Camouflage : bruit blanc ajouté localement. Il imite le niveau du bruit, pas sa texture corrélée. */
export function bruitLocal(hu, h, w, cy, cx, r = 26, sigma = 34, seed = 7) {
  const out = Int16Array.from(hu), g = rng(seed);
  for (let y = Math.max(0, cy - r); y < Math.min(h, cy + r + 1); y++)
    for (let x = Math.max(0, cx - r); x < Math.min(w, cx + r + 1); x++) {
      const m = masqueDoux(Math.hypot(y - cy, x - cx), r * 0.7, r);
      if (m) out[y * w + x] = i16(hu[y * w + x] + m * sigma * gauss(g));
    }
  return out;
}

/** Retouche par lissage (inpainting classique) : la région remplie perd son bruit. */
export function lissage(hu, h, w, cy, cx, r = 22) {
  const out = Int16Array.from(hu), R = r + 14;
  const y0 = Math.max(0, cy - R), y1 = Math.min(h, cy + R + 1), x0 = Math.max(0, cx - R), x1 = Math.min(w, cx + R + 1);
  const l = flou(hu, h, w, y0, y1, x0, x1, 5), W = x1 - x0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const m = masqueDoux(Math.hypot(y - cy, x - cx), r * 0.75, r);
    if (m) out[y * w + x] = i16((1 - m) * hu[y * w + x] + m * l[(y - y0) * W + x - x0]);
  }
  return out;
}

/** Écart-type robuste du bruit dans le poumon, majoré de 20 % (mad_sigma du carnet). */
function sigmaPoumon(hu, h, w, lung) {
  const v = [];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x;
    if (!lung[p]) continue;
    const n = [hu[p - w - 1], hu[p - w], hu[p - w + 1], hu[p - 1], hu[p], hu[p + 1], hu[p + w - 1], hu[p + w], hu[p + w + 1]].sort((a, b) => a - b);
    v.push(hu[p] - n[4]);
  }
  if (!v.length) return 10;
  v.sort((a, b) => a - b);
  const m = v[v.length >> 1], d = v.map((x) => Math.abs(x - m)).sort((a, b) => a - b);
  return 1.4826 * d[d.length >> 1] * 1.2;
}

/** Colle le centre 40 × 40 d'un patch généré avec une fenêtre gaussienne et le bruit du poumon (blend_patch). */
function collerCentre(hu, h, w, lung, patch, cy, cx, seed = 3) {
  const out = Int16Array.from(hu), g = rng(seed), s = sigmaPoumon(hu, h, w, lung), c = 40;
  for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) {
    const y = cy - c / 2 + i, x = cx - c / 2 + j;
    if (y < 0 || x < 0 || y >= h || x >= w) continue;
    const win = Math.exp(-(((i - c / 2) / (c / 3.2)) ** 2 + ((j - c / 2) / (c / 3.2)) ** 2));
    const v = patch[(12 + i) * 64 + 12 + j] + s * gauss(g);
    const q = y * w + x;
    if (hu[q] < -1024) continue;                          // champ de vue conservé, comme l'application
    out[q] = Math.max(-1024, Math.min(3071, Math.round(hu[q] * (1 - win) + v * win)));
  }
  return out;
}

export const ATTAQUES = {
  gan: { nom: "Nodule injecté par le GAN", court: "nodule injecté par GAN", reseau: true },
  suppression: { nom: "Nodule effacé par le GAN", court: "nodule effacé par GAN", reseau: true, nodule: true },
  copie: { nom: "Copier-déplacer", court: "copier-déplacer" },
  bruit: { nom: "Bruit ajouté", court: "bruit ajouté" },
  lissage: { nom: "Retouche par lissage", court: "lissage" },
  nodule: { nom: "Nodule synthétique sans bruit", court: "nodule synthétique" },
};

/**
 * Falsifie la coupe. Les attaques par GAN utilisent le générateur pix2pix du mémoire (ganRemplir, fourni par
 * hybride.js) : il est téléchargé à la première utilisation.
 */
export async function falsifier(type, hu, h, w, lung, cy, cx, ganRemplir) {
  if (type === "gan" || type === "suppression") {
    const y = Math.max(32, Math.min(h - 32, cy)), x = Math.max(32, Math.min(w - 32, cx));
    const patch = await ganRemplir(y, x, type === "gan" ? 1 : 0);
    return { hu: collerCentre(hu, h, w, lung, patch, y, x), cible: [y, x] };
  }
  if (type === "nodule") return { hu: noduleLisse(hu, h, w, cy, cx) };
  if (type === "copie") return copierDeplacer(hu, h, w, cy, cx, lung);
  if (type === "bruit") return { hu: bruitLocal(hu, h, w, cy, cx) };
  if (type === "lissage") return { hu: lissage(hu, h, w, cy, cx) };
  return { hu };
}

// Détecteur hybride complet dans le navigateur, avec les modèles de l'exécution finale du mémoire (v7, 300 patients) :
// branche CNN, branche diffusion (facultative), fusion locale, puis méta-classifieur sur les 40 descripteurs de la coupe
// et les agrégats des patchs (comme le carnet, cellule v5 et m3d_v7.py). Les réseaux ONNX sont servis par ce site
// (dossier modele/v7, en morceaux de 4 Mo), gardés dans le cache du navigateur et exécutés avec ONNX Runtime Web (WebAssembly).

import { P, hgbProba } from "./forensique.js";
import { descripteursCoupe, agg } from "./globaux.js";

const BASE = new URL("../modele/v7/", import.meta.url).href;
const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE = "detecteur-reseaux-v7";
const LOT = 8;                                    // patchs par passage (mémoire et réactivité sur téléphone)
export const TAILLES = { "cnn.onnx": 17, "ddpm.onnx": 17, "gan.onnx": 17 };   // Mo, pour l'affichage

// rend la main au navigateur entre deux lots, sans le ralentissement des minuteries des onglets en arrière-plan
const pause = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = () => r(); c.port2.postMessage(0); });
const sig = (z) => 1 / (1 + Math.exp(-z));

// ------------------------------------------------------------ ONNX Runtime Web (chargé à la demande)
let ortPromesse = null;
function chargerOrt() {
  if (window.ort) return Promise.resolve(window.ort);
  if (!ortPromesse) {
    ortPromesse = new Promise((ok, ko) => {
      const s = document.createElement("script");
      s.src = ORT_CDN + "ort.wasm.min.js";
      s.crossOrigin = "anonymous";
      s.onload = () => {
        const ort = window.ort;
        ort.env.wasm.wasmPaths = ORT_CDN;
        ort.env.wasm.numThreads = 1;              // pas de SharedArrayBuffer sur un hébergement statique
        ort.env.logLevel = "error";
        ok(ort);
      };
      s.onerror = () => { ortPromesse = null; ko(new Error("ONNX Runtime n'a pas pu être chargé (connexion internet ?)")); };
      document.head.append(s);
    });
  }
  return ortPromesse;
}

// ------------------------------------------------------------ réseaux : téléchargement, cache
export async function reseauEnCache(nom) {
  try { const c = await caches.open(CACHE); return !!(await c.match(BASE + nom)); } catch { return false; }
}

// Un réseau est servi en morceaux de 4 Mo (modele/v7/morceaux.json) : sur une connexion qui coupe, seul le morceau
// interrompu est retéléchargé, et les morceaux déjà reçus restent en cache même si la page est fermée.
let manifeste = null;
const attendre = (ms) => new Promise((ok) => setTimeout(ok, ms));
const hex = (b) => Array.from(new Uint8Array(b), (x) => x.toString(16).padStart(2, "0")).join("");

async function lireFlux(url, progres) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const lecteur = r.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await lecteur.read();
    if (done) break;
    parts.push(value); n += value.length; progres?.(n);
  }
  const b = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { b.set(p, o); o += p.length; }
  return b;
}

async function octetsReseau(nom, progres) {
  const c = globalThis.caches ? await caches.open(CACHE).catch(() => null) : null;
  const deja = c && (await c.match(BASE + nom));
  if (deja) return new Uint8Array(await deja.arrayBuffer());
  if (!manifeste) manifeste = await fetch(BASE + "morceaux.json").then((r) => { if (!r.ok) throw new Error("Manifeste des réseaux introuvable"); return r.json(); });
  const m = manifeste[nom], tout = new Uint8Array(m.taille);
  let recu = 0, o = 0;
  for (const p of m.morceaux) {
    const url = BASE + p, enCache = c && (await c.match(url));
    let b = enCache ? new Uint8Array(await enCache.arrayBuffer()) : null;
    for (let essai = 0; !b && essai < 6; essai++) {
      try { b = await lireFlux(url, (n) => progres?.(recu + n, m.taille)); }
      catch { b = null; await attendre(Math.min(8000, 1000 * 2 ** essai)); }
    }
    if (!b) throw new Error("Téléchargement interrompu (" + nom + "). Vérifiez la connexion et relancez : la partie déjà reçue est gardée.");
    if (c && !enCache) await c.put(url, new Response(b)).catch(() => {});
    tout.set(b, o); o += b.length; recu += b.length; progres?.(recu, m.taille);
  }
  const empreinte = crypto.subtle ? hex(await crypto.subtle.digest("SHA-256", tout)) : m.sha256;
  if (o !== m.taille || empreinte !== m.sha256) {
    if (c) for (const p of m.morceaux) await c.delete(BASE + p).catch(() => {});
    throw new Error("Réseau " + nom + " reçu incomplet ou altéré. Relancez l'analyse pour le retélécharger.");
  }
  if (c) {
    await c.put(BASE + nom, new Response(tout, { headers: { "Content-Type": "application/octet-stream" } })).catch(() => {});
    for (const p of m.morceaux) await c.delete(BASE + p).catch(() => {});
  }
  return tout;
}

/** Paramètres de l'exécution v7 : arbres des classifieurs et fusions (fichiers JSON du site). */
let config = null;
export async function modelesV7() {
  if (config) return config;
  const lire = (n) => fetch(BASE + n).then((r) => { if (!r.ok) throw new Error("Modèle introuvable : " + n); return r.json(); });
  const [forensique, meta, meta2, fusion] = await Promise.all(["hgb_forensique.json", "hgb_meta.json", "hgb_meta2.json", "fusion.json"].map(lire));
  return (config = { ...fusion, forensique, meta, meta2 });      // fusion.json décrit aussi meta/meta2 : les arbres passent après
}

const sessions = {};
export async function session(nom, progres) {
  if (sessions[nom]) return sessions[nom];
  const ort = await chargerOrt();
  const octets = await octetsReseau(nom, progres);
  sessions[nom] = await ort.InferenceSession.create(octets, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return sessions[nom];
}

// ------------------------------------------------------------ branches
function patchs(hu, w, coords, f) {
  const X = new Float32Array(coords.length * P * P);
  coords.forEach(([y, x], k) => {
    for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) X[k * P * P + i * P + j] = f(hu[(y + i) * w + x + j]);
  });
  return X;
}

/** Branche CNN : probabilité par patch et carte de segmentation (sigmoïdes), comme Detecteur.cnn_patch. */
export async function brancheCNN(hu, w, coords, etape) {
  const ort = await chargerOrt(), s = await session("cnn.onnx");
  const X = patchs(hu, w, coords, (v) => Math.fround(v / 1000));
  const p = new Float64Array(coords.length), seg = new Float32Array(coords.length * P * P);
  for (let b = 0; b < coords.length; b += LOT) {
    const n = Math.min(LOT, coords.length - b);
    const x = new ort.Tensor("float32", X.subarray(b * P * P, (b + n) * P * P), [n, 1, P, P]);
    const out = await s.run({ x });
    const lo = out.logit.data, se = out.seg.data;
    for (let i = 0; i < n; i++) p[b + i] = sig(lo[i]);
    for (let i = 0; i < n * P * P; i++) seg[b * P * P + i] = sig(se[i]);
    etape?.(Math.min(coords.length, b + n), coords.length);
    await pause();
  }
  return { p, seg };
}

function gauss(seed) {             // tirages normaux reproductibles (mulberry32 + Box-Muller)
  let s = seed >>> 0, garde = null;
  const u = () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => {
    if (garde !== null) { const g = garde; garde = null; return g; }
    let a = 0; while (!a) a = u();
    const b = u(), r = Math.sqrt(-2 * Math.log(a));
    garde = r * Math.sin(2 * Math.PI * b);
    return r * Math.cos(2 * Math.PI * b);
  };
}

/** Branche diffusion : erreur de reconstruction du DDPM à t* = 150, moyenne de 3 tirages, puis écart robuste. */
export async function brancheDiffusion(hu, w, coords, cfg, etape) {
  const ort = await chargerOrt(), s = await session("ddpm.onnx");
  const { abar, win_lo, win_hi, t_star, tirages } = cfg;
  const X0 = patchs(hu, w, coords, (v) => Math.fround(Math.min(1, Math.max(0, (v - win_lo) / (win_hi - win_lo))) * 2 - 1));
  const sa = Math.fround(Math.sqrt(abar)), sb = Math.fround(Math.sqrt(1 - abar));
  const g = gauss(0), N = coords.length, err = new Float64Array(N);
  const eps = new Float32Array(tirages * N * P * P);
  for (let i = 0; i < eps.length; i++) eps[i] = g();
  let fait = 0;
  for (let b = 0; b < N; b += LOT) {
    const n = Math.min(LOT, N - b);
    const t = new ort.Tensor("int64", new BigInt64Array(n).fill(BigInt(t_star)), [n]);
    for (let k = 0; k < tirages; k++) {
      const xt = new Float32Array(n * P * P);
      for (let i = 0; i < n * P * P; i++) xt[i] = sa * X0[b * P * P + i] + sb * eps[(k * N + b) * P * P + i];
      const out = await s.run({ x: new ort.Tensor("float32", xt, [n, 1, P, P]), t });
      const e = out.eps.data;
      for (let q = 0; q < n; q++) {
        let acc = 0;
        for (let i = 0; i < P * P; i++) {
          const j = q * P * P + i, x0h = (xt[j] - sb * e[j]) / sa, d = x0h - X0[b * P * P + j];
          acc += d * d;
        }
        err[b + q] += acc / (P * P) / tirages;
      }
      fait += n; etape?.(fait, N * tirages);
      await pause();
    }
  }
  const med = mediane(err), mad = mediane(Array.from(err, (v) => Math.abs(v - med)));
  return Array.from(err, (v) => (v - med) / (1.4826 * mad + 1e-9));
}

function mediane(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0;
}
const top3 = (p) => { const s = Array.from(p).sort((a, b) => b - a).slice(0, 3); return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0; };

/**
 * Analyse complète (exécution v7). res : analyse forensique de la même coupe (patchs, probabilités, résidus).
 * Sans diffusion : fusion locale « bruit + CNN », puis méta-classifieur à 52 entrées (AUC 0,879 dans le mémoire).
 * Avec diffusion : fusion locale à 4 entrées, puis méta-classifieur à 60 entrées (trois branches, AUC 0,870).
 */
export async function analyseComplete(hu, w, h, res, cfg, { avecDiffusion = false, etape = () => {} } = {}) {
  const t0 = performance.now(), coords = res.coords;
  const g = descripteursCoupe(hu, h, w, res.lung, res.R, res.dup);
  etape("cnn", 0, coords.length);
  const { p: pCnn, seg } = await brancheCNN(hu, w, coords, (a, b) => etape("cnn", a, b));
  let sz = null, pLoc, entree, meta;
  if (avecDiffusion) {
    etape("diffusion", 0, coords.length * cfg.ddpm.tirages);
    sz = await brancheDiffusion(hu, w, coords, cfg.ddpm, (a, b) => etape("diffusion", a, b));
    const L = cfg.local, az = sz.map(Math.abs);
    pLoc = res.p.map((pf, i) => sig(L.coef[0] * pf + L.coef[1] * pCnn[i] + L.coef[2] * sz[i] + L.coef[3] * az[i] + L.intercept));
    entree = [...g, ...agg(pLoc), ...agg(res.p), ...agg(pCnn), ...agg(sz), ...agg(az)];
    meta = cfg.meta;
  } else {
    const L = cfg.loc2;
    pLoc = res.p.map((pf, i) => sig(L.coef[0] * pf + L.coef[1] * pCnn[i] + L.intercept));
    entree = [...g, ...agg(pLoc), ...agg(res.p), ...agg(pCnn)];
    meta = cfg.meta2;
  }
  const score = hgbProba(meta, entree.map((v) => (Number.isFinite(v) ? v : 0)));
  const carteFusion = new Float32Array(w * h), carteCnn = new Float32Array(w * h), cnt = new Float32Array(w * h);
  coords.forEach(([y, x], k) => {
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const q = (y + a) * w + x + b;
      carteFusion[q] += pLoc[k]; carteCnn[q] += seg[k * P * P + a * P + b]; cnt[q]++;
    }
  });
  for (let q = 0; q < w * h; q++) if (cnt[q]) { carteFusion[q] /= cnt[q]; carteCnn[q] /= cnt[q]; }
  return {
    score, branches: { forensique: top3(res.p), cnn: top3(pCnn), diffusion: sz ? top3(sz.map(Math.abs)) : NaN, locale: top3(pLoc) },
    pCnn, pLoc, sz, g, entree, carteFusion, carteCnn, avecDiffusion, duree: performance.now() - t0,
  };
}

// ------------------------------------------------------------ attaques par GAN (pix2pix entraîné dans le carnet)
const CENTRE = (() => { const m = new Float32Array(P * P); for (let i = P / 4; i < (3 * P) / 4; i++) for (let j = P / 4; j < (3 * P) / 4; j++) m[i * P + j] = 1; return m; })();

/** Remplit le centre d'un patch avec le générateur : cond = 1 ajoute un nodule, cond = 0 en efface un (gan_fill). */
export async function ganRemplir(hu, w, cy, cx, cond, cfg, progres) {
  const ort = await chargerOrt(), s = await session("gan.onnx", progres);
  const { win_lo: lo, win_hi: hi } = cfg.ddpm;
  const y = new Float32Array(P * P);
  for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
    const v = hu[(cy - P / 2 + i) * w + cx - P / 2 + j];
    y[i * P + j] = Math.fround(Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * 2 - 1);
  }
  const x = new Float32Array(3 * P * P);
  for (let i = 0; i < P * P; i++) { x[i] = y[i] * (1 - CENTRE[i]); x[P * P + i] = CENTRE[i]; x[2 * P * P + i] = cond; }
  const out = (await s.run({ x: new ort.Tensor("float32", x, [1, 3, P, P]) })).y.data;
  const patch = new Float32Array(P * P);
  for (let i = 0; i < P * P; i++) patch[i] = ((y[i] * (1 - CENTRE[i]) + out[i] * CENTRE[i] + 1) / 2) * (hi - lo) + lo;
  return patch;
}

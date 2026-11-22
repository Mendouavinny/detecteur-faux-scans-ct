// Détecteur hybride complet dans le navigateur : branche CNN, branche diffusion, fusion locale et fusion robuste.
// Reprise de detecteur.py (Detecteur.branch_outputs, fused_local, branch_vector, fusion). Les réseaux ONNX sont ceux
// de l'application Python ; ils viennent du dépôt Hugging Face du mémoire (morceaux de 4 Mo, empreinte SHA-256
// vérifiée), puis restent dans le cache du navigateur. Le calcul se fait avec ONNX Runtime Web (WebAssembly).

import { P } from "./forensique.js";

const DEPOT = "https://huggingface.co/MENDOUA/detecteur-faux-scans-ct-modeles/resolve/main/";
const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE = "detecteur-reseaux-v1";
const LOT = 8;                                    // patchs par passage (mémoire et réactivité sur téléphone)
export const TAILLES = { "cnn.onnx": 17, "ddpm.onnx": 17, "gan.onnx": 17 };   // Mo, pour l'affichage

const pause = () => new Promise((r) => setTimeout(r, 0));
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

// ------------------------------------------------------------ réseaux : téléchargement, vérification, cache
let manifeste = null;
async function lireManifeste() {
  if (manifeste) return manifeste;
  const c = await caches.open(CACHE).catch(() => null);
  const url = DEPOT + "morceaux.json";
  let r = c && (await c.match(url));
  if (!r) {
    r = await fetch(url);
    if (!r.ok) throw new Error("Le dépôt des modèles est injoignable (" + r.status + ").");
    if (c) await c.put(url, r.clone());
  }
  return (manifeste = await r.json());
}

async function sha256(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(h), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function reseauEnCache(nom) {
  try { const c = await caches.open(CACHE); return !!(await c.match("reseau/" + nom)); } catch { return false; }
}

async function octetsReseau(nom, progres) {
  const c = await caches.open(CACHE).catch(() => null);
  const cle = "reseau/" + nom;
  const deja = c && (await c.match(cle));
  if (deja) return new Uint8Array(await deja.arrayBuffer());
  const m = (await lireManifeste())[nom];
  if (!m) throw new Error("Réseau absent du dépôt : " + nom);
  const parts = [];
  let recu = 0;
  for (let i = 0; i < m.morceaux.length; i++) {
    let r;
    for (let essai = 0; essai < 3; essai++) {
      try { r = await fetch(DEPOT + m.morceaux[i]); if (r.ok) break; } catch (e) { r = null; }
      await new Promise((ok) => setTimeout(ok, 1500));
    }
    if (!r || !r.ok) throw new Error("Téléchargement interrompu (" + nom + "). Vérifiez la connexion et relancez.");
    const b = new Uint8Array(await r.arrayBuffer());
    parts.push(b); recu += b.length;
    progres?.(i + 1, m.morceaux.length);
  }
  const tout = new Uint8Array(recu);
  let o = 0;
  for (const p of parts) { tout.set(p, o); o += p.length; }
  if ((await sha256(tout)) !== m.sha256) throw new Error("Empreinte SHA-256 incorrecte pour " + nom + " : fichier refusé.");
  if (c) await c.put(cle, new Response(tout, { headers: { "Content-Type": "application/octet-stream" } })).catch(() => {});
  return tout;
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
 * Analyse complète : reprend l'analyse forensique déjà faite (res), ajoute CNN et diffusion, puis les deux fusions.
 * avecDiffusion = false : la branche diffusion n'est pas calculée (s = 0). Son poids est presque nul
 * (−0,011 dans la fusion locale, 0 dans la fusion robuste) ; l'interface l'indique.
 */
export async function analyseComplete(hu, w, h, res, cfg, { avecDiffusion = true, etape = () => {} } = {}) {
  const t0 = performance.now(), coords = res.coords;
  etape("cnn", 0, coords.length);
  const { p: pCnn, seg } = await brancheCNN(hu, w, coords, (a, b) => etape("cnn", a, b));
  let sDiff = new Array(coords.length).fill(0);
  if (avecDiffusion) {
    etape("diffusion", 0, coords.length * cfg.ddpm.tirages);
    sDiff = await brancheDiffusion(hu, w, coords, cfg.ddpm, (a, b) => etape("diffusion", a, b));
  }
  const L = cfg.local, pLoc = res.p.map((pf, i) => sig(L.coef[0] * pf + L.coef[1] * pCnn[i] + L.coef[2] * sDiff[i] + L.intercept));
  const v = [top3(res.p), top3(pCnn), top3(sDiff), top3(pLoc)];
  const F = cfg.fusion;
  let z = F.t[F.t.length - 1];
  F.cols.forEach((c, k) => { z += F.t[k] * ((v[c] - F.mu[k]) / F.sd[k]); });
  const score = sig(z);
  // cartes : fusion locale et segmentation du CNN (moyenne des patchs qui se recouvrent)
  const carteFusion = new Float32Array(w * h), carteCnn = new Float32Array(w * h), cnt = new Float32Array(w * h);
  coords.forEach(([y, x], k) => {
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const q = (y + a) * w + x + b;
      carteFusion[q] += pLoc[k]; carteCnn[q] += seg[k * P * P + a * P + b]; cnt[q]++;
    }
  });
  for (let q = 0; q < w * h; q++) if (cnt[q]) { carteFusion[q] /= cnt[q]; carteCnn[q] /= cnt[q]; }
  return {
    score, branches: { forensique: v[0], cnn: v[1], diffusion: v[2], locale: v[3] }, pCnn, pLoc, sDiff,
    carteFusion, carteCnn, avecDiffusion, duree: performance.now() - t0,
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

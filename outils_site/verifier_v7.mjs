// Contrôle de parité des modèles v7 sans navigateur (Node) : branche forensique, 40 descripteurs de la coupe,
// méta-classifieurs « bruit + CNN » et « trois branches ». Les sorties des réseaux (p_cnn, s diffusion) sont
// reprises de reference.json : les réseaux ONNX se vérifient dans le navigateur (site/verification.html).
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseDicom, analyser, hgbProba } from "../site/js/forensique.js";
import { descripteursCoupe, agg } from "../site/js/globaux.js";

const ici = path.dirname(fileURLToPath(import.meta.url));
const lire = (p) => JSON.parse(fs.readFileSync(path.join(ici, "../site/modele/v7", p)));
const ref = lire("reference.json"), forensique = lire("hgb_forensique.json"), meta = lire("hgb_meta.json"), meta2 = lire("hgb_meta2.json");
const fusion = lire("fusion.json");
const sig = (z) => 1 / (1 + Math.exp(-z));
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const maxRel = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / Math.max(1e-6, Math.abs(b[i]))), 0);
const pire = { pf: 0, g: 0, m2: 0, m: 0 };
for (const [f, r] of Object.entries(ref)) {
  const b = fs.readFileSync(path.join(ici, "../site/exemples", f));
  const d = parseDicom(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const hu = d.hu.map((v) => Math.min(3071, Math.max(-1024, v)));
  const a = analyser(hu, d.rows, d.cols, forensique);
  if (a.coords.length !== r.n) { console.log(f, "nombre de patchs", a.coords.length, "≠", r.n); pire.pf = Infinity; continue; }
  const pf = Array.from(a.p), pc = r.p_cnn, sz = r.sz, az = sz.map(Math.abs);
  const g = descripteursCoupe(hu, d.rows, d.cols, a.lung, a.R, a.dup);
  const L2 = fusion.loc2, L = fusion.local;
  const f2 = pf.map((p, i) => sig(L2.coef[0] * p + L2.coef[1] * pc[i] + L2.intercept));
  const fl = pf.map((p, i) => sig(L.coef[0] * p + L.coef[1] * pc[i] + L.coef[2] * sz[i] + L.coef[3] * az[i] + L.intercept));
  const nz = (v) => v.map((x) => (Number.isFinite(x) ? x : 0));
  const s2 = hgbProba(meta2, nz([...g, ...agg(f2), ...agg(pf), ...agg(pc)]));
  const s3 = hgbProba(meta, nz([...g, ...agg(fl), ...agg(pf), ...agg(pc), ...agg(sz), ...agg(az)]));
  const e = { pf: maxAbs(pf, r.p_for), g: maxRel(g, r.g), m2: Math.abs(s2 - r.meta2), m: Math.abs(s3 - r.meta) };
  for (const k in pire) pire[k] = Math.max(pire[k], e[k]);
  console.log(f.padEnd(30), `patchs ${r.n}`.padEnd(11), `p bruit ${e.pf.toExponential(1)}`, ` descripteurs ${e.g.toExponential(1)}`,
    ` bruit+CNN ${s2.toFixed(4)} / ${r.meta2.toFixed(4)}`, ` trois branches ${s3.toFixed(4)} / ${r.meta.toFixed(4)}`);
}
console.log(`Écarts max : p bruit ${pire.pf.toExponential(1)}, descripteurs (relatif) ${pire.g.toExponential(1)}, ` +
  `bruit+CNN ${pire.m2.toExponential(1)}, trois branches ${pire.m.toExponential(1)}`);

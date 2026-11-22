// Vérifie que la branche forensique en JavaScript (site/js/forensique.js) donne les mêmes probabilités que Python.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseDicom, analyser } from "../site/js/forensique.js";

const ici = path.dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(fs.readFileSync(path.join(ici, "reference_python.json")));
const model = JSON.parse(fs.readFileSync(path.join(ici, "../site/modele/hgb_forensique.json")));
let pire = 0;
for (const [f, r] of Object.entries(ref)) {
  const b = fs.readFileSync(path.join(ici, "../site/exemples", f));
  const d = parseDicom(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const a = analyser(d.hu, d.rows, d.cols, model);
  const ecart = Math.max(...a.p.map((v, i) => Math.abs(v - r.p[i])));
  pire = Math.max(pire, ecart);
  console.log(f.padEnd(30), "score JS", a.score.toFixed(4), "Python", r.score.toFixed(4), "écart max par patch", ecart.toExponential(1));
}
console.log(pire < 1e-9 ? "Parité exacte." : "Écarts détectés : " + pire);

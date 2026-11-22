// Interface de l'application : accueil (volet image / bruit) et laboratoire d'analyse.
import { parseDicom, analyser, lungMask, residuals, carteBruit } from "./forensique.js";
import { falsifier, cibleParDefaut, noduleLisse, ATTAQUES } from "./attaques.js";

const $ = (s) => document.querySelector(s);
const fr = (v, d = 3) => v.toFixed(d).replace(".", ",");
const reduit = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pause = () => new Promise((r) => setTimeout(r, 0));
const TAU1 = 0.5, TAU2 = 0.85;

// ------------------------------------------------------------ rendu
const FEN = { lo: -1350, hi: 150 };                        // fenêtre pulmonaire : largeur 1500, centre −600
function gris(hu, i) { const v = (hu[i] - FEN.lo) / (FEN.hi - FEN.lo); return v <= 0 ? 0 : v >= 1 ? 255 : (v * 255) | 0; }

function dessinerImage(ctx, hu, w, h) {
  const im = ctx.createImageData(w, h), d = im.data;
  for (let i = 0; i < w * h; i++) { const g = gris(hu, i); d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = g; d[4 * i + 3] = 255; }
  return im;
}

function echelleResidu(kv, lung) {
  const v = [];
  for (let i = 0; i < kv.length; i += 7) if (lung[i]) v.push(Math.abs(kv[i]));
  v.sort((a, b) => a - b);
  return (v[(v.length * 0.5) | 0] || 8) * 4.5;
}

function dessinerResidu(ctx, kv, w, h, s) {
  const im = ctx.createImageData(w, h), d = im.data;
  for (let i = 0; i < w * h; i++) {
    let g = 128 + (127 * kv[i]) / s;
    g = g < 0 ? 0 : g > 255 ? 255 : g;
    d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = g; d[4 * i + 3] = 255;
  }
  return im;
}

function melanger(im, i, rgb, a) {
  const d = im.data;
  d[4 * i] += (rgb[0] - d[4 * i]) * a; d[4 * i + 1] += (rgb[1] - d[4 * i + 1]) * a; d[4 * i + 2] += (rgb[2] - d[4 * i + 2]) * a;
}
const ROUGE = [222, 64, 46], FROID = [86, 172, 222], AMBRE = [226, 166, 49];

// ------------------------------------------------------------ accueil : volet image / bruit
async function accueil(model, exemples) {
  const fig = $("#film-accueil"), curseur = $("#accueil-curseur");
  const buf = await (await fetch("exemples/LIDC-IDRI-0085_coupe086.dcm")).arrayBuffer();
  const d = parseDicom(buf), w = d.cols, h = d.rows;
  const lung = lungMask(d.hu, h, w);
  const [cy, cx] = cibleParDefaut(lung, h, w);
  const hu = noduleLisse(d.hu, h, w, cy, cx, 15);
  const R = residuals(hu, h, w);
  $("#accueil-image").getContext("2d").putImageData(dessinerImage($("#accueil-image").getContext("2d"), hu, w, h), 0, 0);
  const ctxB = $("#accueil-bruit").getContext("2d");
  ctxB.putImageData(dessinerResidu(ctxB, R.kv, w, h, echelleResidu(R.kv, lung)), 0, 0);
  // repère discret autour de la région injectée, côté bruit seulement
  ctxB.strokeStyle = "rgba(226,166,49,.9)"; ctxB.lineWidth = 1.5; ctxB.setLineDash([4, 4]);
  ctxB.beginPath(); ctxB.arc(cx, cy, 40, 0, 2 * Math.PI); ctxB.stroke();
  const poser = (p) => { fig.style.setProperty("--pos", p + "%"); curseur.value = p; };
  curseur.addEventListener("input", () => poser(+curseur.value));
  // un seul mouvement à l'ouverture : le volet balaie la coupe, comme l'anneau du scanner, puis s'arrête au milieu
  const fin = Math.round(((cx - 4) / w) * 100);    // le volet s'arrête sur le nodule : moitié image, moitié bruit
  if (reduit) { poser(fin); return; }
  poser(100);
  let t0; const D = 1800;
  const pas = (t) => {
    if (t0 === undefined) t0 = t;
    const u = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - u, 3);
    poser(100 - (100 - fin) * e);
    if (u < 1) requestAnimationFrame(pas);
  };
  setTimeout(() => requestAnimationFrame(pas), 350);
}

// ------------------------------------------------------------ laboratoire
const etat = { couche: "image", attaque: "aucune", cible: null, source: null, nom: "", meta: null, sha: "" };

const AIDES = {
  image: "La coupe en fenêtre pulmonaire. Les carrés ambrés marquent les trois patchs les plus suspects pour le classifieur.",
  residu: "Le résidu du filtre KV 5 × 5 : l'anatomie disparaît, il reste le bruit. Une région lisse ou trop granuleuse s'y voit à l'œil.",
  bruit: "Niveau local du bruit dans le poumon, rapporté au niveau habituel de cette coupe. En bleu, le bruit manque (région lissée ou générée) ; en ambre, il y en a trop (bruit ajouté).",
  proba: "Probabilité de falsification de chaque patch de 64 × 64 pixels selon le classifieur entraîné. Les patchs se recouvrent : la carte est leur moyenne.",
  copie: "Fenêtres de 8 × 8 pixels strictement identiques et éloignées : la signature d'un copier-déplacer. Un trait relie la source à la copie.",
};

function legende(couche) {
  const L = $("#legende");
  if (couche === "bruit") L.innerHTML = '<span>bruit manquant</span><span class="degrade" style="background:linear-gradient(90deg,rgb(86,172,222),rgba(86,172,222,0) 42%,rgba(226,166,49,0) 58%,rgb(226,166,49))"></span><span>bruit en excès</span>';
  else if (couche === "proba") L.innerHTML = '<span>0</span><span class="degrade" style="background:linear-gradient(90deg,rgba(222,64,46,0),rgb(222,64,46))"></span><span>1</span>';
  else L.innerHTML = "";
}

function rendre() {
  const c = $("#labo-canvas"), ctx = c.getContext("2d"), a = etat.res, hu = etat.huVu, w = etat.w, h = etat.h;
  if (!hu) return;
  if (c.width !== w) { c.width = w; c.height = h; }
  let im;
  if (etat.couche === "residu" && a) im = dessinerResidu(ctx, a.R.kv, w, h, echelleResidu(a.R.kv, a.lung));
  else im = dessinerImage(ctx, hu, w, h);
  if (a && a.ok && etat.couche === "proba") {
    for (let i = 0; i < w * h; i++) if (a.carte[i] > 0.02) melanger(im, i, ROUGE, Math.min(0.85, a.carte[i] * 0.9));
  }
  if (a && etat.couche === "bruit") {
    const cb = etat.carteB;
    for (let i = 0; i < w * h; i++) {
      if (!a.lung[i]) continue;
      const l = Math.log2(Math.max(cb.ratio[i], 1e-3));
      if (l < -0.6) melanger(im, i, FROID, Math.min(0.9, (-l - 0.6) * 0.45));
      else if (l > 0.6) melanger(im, i, AMBRE, Math.min(0.9, (l - 0.6) * 0.45));
    }
  }
  if (a && a.ok && etat.couche === "copie") {
    for (let i = 0; i < w * h; i++) if (a.dup[i]) melanger(im, i, AMBRE, 0.75);
  }
  ctx.putImageData(im, 0, 0);
  if (a && a.ok && etat.couche === "image") {
    const idx = a.p.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 3);
    ctx.lineWidth = 2; ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.font = "12px Annot, monospace"; ctx.fillStyle = "rgba(226,166,49,.95)";
    for (const [v, i] of idx) {
      const [y, x] = a.coords[i];
      ctx.strokeRect(x + 1, y + 1, 62, 62);
      ctx.fillText(fr(v, 2), x + 4, y + 14);
    }
  }
  if (a && a.ok && etat.couche === "copie" && a.groups.length) {
    ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    const vus = new Set();
    for (const g of a.groups) {
      const [y0, x0] = g[0], [y1, x1] = g[g.length - 1], k = `${y1 - y0},${x1 - x0}`;
      if (vus.has(k) || vus.size > 6) continue;
      vus.add(k);
      ctx.beginPath(); ctx.moveTo(x0 + 4, y0 + 4); ctx.lineTo(x1 + 4, y1 + 4); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  if (etat.cible && etat.attaque !== "aucune" && etat.couche !== "copie") {
    const [cy, cx] = etat.cible;
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.arc(cx, cy, 36, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
  $("#aide-film").textContent = AIDES[etat.couche] + (etat.attaque !== "aucune" ? " Cliquez sur le poumon pour déplacer la falsification." : "");
  legende(etat.couche);
}

function verdictDe(s) { return s >= TAU2 ? ["Alerte", "alerte"] : s >= TAU1 ? ["À vérifier", "verifier"] : ["Conforme", "conforme"]; }

function afficherVerdict() {
  const a = etat.res, mot = $("#verdict-mot");
  if (!a || !a.ok) {
    mot.textContent = "Non analysable"; mot.className = "verdict-mot";
    $("#score").textContent = "—"; $("#m-patchs").textContent = "0";
    $("#jauge-valeur").style.left = "0%";
    $("#comparaison").hidden = true;
    $("#aide-film").textContent = a ? a.message + " Il faut une coupe thoracique où les poumons sont visibles." : "";
    return;
  }
  const [txt, cl] = verdictDe(a.score);
  mot.textContent = txt; mot.className = "verdict-mot " + cl;
  $("#score").textContent = fr(a.score);
  $("#jauge-valeur").style.left = (a.score * 100).toFixed(1) + "%";
  $("#m-patchs").textContent = a.coords.length;
  $("#m-copie").textContent = a.dupPx ? `oui, ${a.dupPx} pixels` : "non";
  $("#m-duree").textContent = Math.round(a.duree) + " ms";
  const comp = $("#comparaison"), av = $("#jauge-avant");
  if (etat.attaque !== "aucune" && etat.resOrig && etat.resOrig.ok) {
    const s0 = etat.resOrig.score, [t0] = verdictDe(s0);
    comp.hidden = false;
    comp.innerHTML = `Coupe d'origine : <b>${fr(s0)}</b> (${t0.toLowerCase()}).<br>Après ${ATTAQUES[etat.attaque].court} : <b>${fr(a.score)}</b> (${txt.toLowerCase()}).` +
      (a.score < TAU1 ? "<br>Cette falsification échappe ici à la branche du bruit seule ; regardez la couche « Niveau du bruit »." : "");
    av.hidden = false; av.style.left = (s0 * 100).toFixed(1) + "%";
  } else { comp.hidden = true; av.hidden = true; }
}

async function calculer() {
  const voile = $("#calcul");
  voile.hidden = false;
  const etape = (t) => (voile.firstElementChild.textContent = t + "…");
  etape("Analyse"); await pause();
  const { w, h } = etat;
  if (etat.attaque !== "aucune") {
    if (!etat.resOrig) etat.resOrig = analyser(etat.hu, h, w, etat.model);
    if (!etat.cible) etat.cible = cibleParDefaut(etat.resOrig.lung, h, w);
    const r = falsifier(etat.attaque, etat.hu, h, w, etat.resOrig.lung, etat.cible[0], etat.cible[1]);
    etat.huVu = r.hu;
  } else etat.huVu = etat.hu;
  await pause();
  etat.res = analyser(etat.huVu, h, w, etat.model, etape);
  if (etat.attaque === "aucune") etat.resOrig = etat.res;
  etat.carteB = etat.res.R ? carteBruit(etat.res.R, etat.res.lung, h, w) : null;
  voile.hidden = true;
  $("#film-labo").classList.toggle("sans-cible", etat.attaque === "aucune");
  rendre(); afficherVerdict();
}

async function charger(buf, nom, meta = {}) {
  let d;
  try { d = parseDicom(buf); }
  catch (e) { montrerErreur(e.message); return; }
  Object.assign(etat, { hu: d.hu, w: d.cols, h: d.rows, nom, meta: d, resOrig: null, cible: null });
  const hash = await crypto.subtle.digest("SHA-256", buf).catch(() => null);
  const hex = hash ? Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("") : "";
  $("#m-sha").textContent = hex ? hex.slice(0, 12) + "…" + hex.slice(-8) : "indisponible";
  $("#m-sha").title = hex;
  $("#annot-hg").innerHTML = `${meta.patient || d.patient || "patient anonyme"}<br>${meta.coupe ? "coupe " + meta.coupe : nom}`;
  $("#annot-hd").innerHTML = `${(d.manufacturer || "").toLowerCase() || "fabricant inconnu"}<br>noyau ${d.kernel || "?"}`;
  $("#annot-bg").innerHTML = `${d.cols} × ${d.rows}<br>F 1500 C −600`;
  await calculer();
}

function montrerErreur(msg) {
  $("#verdict-mot").textContent = "Fichier refusé"; $("#verdict-mot").className = "verdict-mot";
  $("#aide-film").textContent = msg;
}

function annotBas() {
  const t = { image: "image", residu: "résidu KV 5×5", bruit: "niveau du bruit", proba: "patchs suspects", copie: "copies exactes" }[etat.couche];
  $("#annot-bd").innerHTML = (etat.attaque !== "aucune" ? `falsifiée : ${ATTAQUES[etat.attaque].court}<br>` : "") + t;
}

async function laboratoire(model, exemples) {
  etat.model = model;
  const sel = $("#choix-exemple");
  for (const e of exemples) {
    const o = document.createElement("option");
    o.value = e.fichier;
    o.textContent = `${e.patient}, coupe ${e.coupe}${e.nodules ? " (nodule réel)" : ""}`;
    sel.append(o);
  }
  const ex = (f) => exemples.find((e) => e.fichier === f);
  const ouvrirExemple = async (f) => {
    const buf = await (await fetch("exemples/" + f)).arrayBuffer();
    await charger(buf, f, ex(f));
    annotBas();
  };
  sel.value = "LIDC-IDRI-0085_coupe086.dcm";
  sel.addEventListener("change", () => ouvrirExemple(sel.value));

  const ouvrirFichier = async (file) => {
    if (!file) return;
    const buf = await file.arrayBuffer();
    sel.value = "";
    await charger(buf, file.name);
    annotBas();
  };
  $("#fichier").addEventListener("change", (e) => ouvrirFichier(e.target.files[0]));
  const film = $("#film-labo"), depot = $("#depot");
  film.addEventListener("dragover", (e) => { e.preventDefault(); depot.hidden = false; });
  film.addEventListener("dragleave", () => (depot.hidden = true));
  film.addEventListener("drop", (e) => { e.preventDefault(); depot.hidden = true; ouvrirFichier(e.dataTransfer.files[0]); });

  document.querySelectorAll("#attaques input").forEach((r) => r.addEventListener("change", async () => {
    etat.attaque = r.value; annotBas(); await calculer();
  }));
  film.addEventListener("click", async (e) => {
    if (etat.attaque === "aucune" || !etat.resOrig) return;
    const c = $("#labo-canvas").getBoundingClientRect();
    const x = Math.round(((e.clientX - c.left) / c.width) * etat.w), y = Math.round(((e.clientY - c.top) / c.height) * etat.h);
    if (!etat.resOrig.lung[y * etat.w + x]) { $("#aide-film").textContent = "Placez la falsification dans le poumon : c'est là que le détecteur cherche."; return; }
    etat.cible = [y, x]; await calculer();
  });
  const onglets = document.querySelectorAll(".couches button");
  onglets.forEach((b) => b.addEventListener("click", () => {
    onglets.forEach((o) => o.setAttribute("aria-selected", String(o === b)));
    etat.couche = b.dataset.couche; annotBas(); rendre();
  }));
  document.querySelector(".couches").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const l = [...onglets], i = l.findIndex((o) => o.getAttribute("aria-selected") === "true");
    const n = l[(i + (e.key === "ArrowRight" ? 1 : l.length - 1)) % l.length]; n.focus(); n.click();
  });
  await ouvrirExemple(sel.value);
}

// ------------------------------------------------------------ démarrage
(async () => {
  try {
    const [model, exemples] = await Promise.all([
      fetch("modele/hgb_forensique.json").then((r) => r.json()),
      fetch("exemples/index.json").then((r) => r.json()),
    ]);
    await accueil(model, exemples);
    await laboratoire(model, exemples);
  } catch (e) {
    console.error(e);
    const a = $("#aide-film");
    if (a) a.textContent = "Le chargement a échoué (" + e.message + "). Cette page doit être servie par un serveur web, pas ouverte comme fichier local.";
  }
})();

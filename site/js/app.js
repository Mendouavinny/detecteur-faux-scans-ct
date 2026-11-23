// Application : dépôt de coupes DICOM (une coupe ou une série), analyse, constats en clair, graphiques,
// rapport PDF, historique chaîné, page « Comprendre ». Tout le calcul se fait dans le navigateur.
import { parseDicom, analyser, lungMask, residuals, carteBruit, P } from "./forensique.js";
import { falsifier, cibleParDefaut, noduleLisse, ATTAQUES } from "./attaques.js";
import * as G from "./graphes.js";
import { imprimerRapport, exporterCSV } from "./rapport.js";

const $ = (s) => document.querySelector(s);
const fr = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",").replace("-", "−") : "—");
const reduit = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pause = () => new Promise((r) => setTimeout(r, 0));
let H = null;                                            // module du détecteur complet, chargé à la demande
const hybride = async () => (H ||= await import("./hybride.js"));
const mediane = (v) => { const s = Float64Array.from(v).sort(); return s.length ? s[s.length >> 1] : NaN; };

const stock = {
  lire(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  ecrire(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* navigation privée */ } },
};

// ------------------------------------------------------------ rendu des images
const FEN = { lo: -1350, hi: 150 };                        // fenêtre pulmonaire : largeur 1500, centre −600
function dessinerImage(ctx, hu, w, h) {
  const im = ctx.createImageData(w, h), d = im.data;
  for (let i = 0; i < w * h; i++) {
    const v = (hu[i] - FEN.lo) / (FEN.hi - FEN.lo), g = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255) | 0;
    d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = g; d[4 * i + 3] = 255;
  }
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
const ROUGE = [222, 64, 46], FROID = [63, 151, 207], AMBRE = [226, 166, 49];

// ------------------------------------------------------------ page « Comprendre » : volet image / bruit
let voletPret = false;
async function voletAccueil() {
  if (voletPret) return; voletPret = true;
  const fig = $("#film-accueil"), curseur = $("#accueil-curseur");
  const buf = await (await fetch("exemples/LIDC-IDRI-0085_coupe086.dcm")).arrayBuffer();
  const d = parseDicom(buf), w = d.cols, h = d.rows;
  const lung = lungMask(d.hu, h, w);
  const [cy, cx] = cibleParDefaut(lung, h, w);
  const hu = noduleLisse(d.hu, h, w, cy, cx, 15);
  const R = residuals(hu, h, w);
  const ctxI = $("#accueil-image").getContext("2d"), ctxB = $("#accueil-bruit").getContext("2d");
  ctxI.putImageData(dessinerImage(ctxI, hu, w, h), 0, 0);
  ctxB.putImageData(dessinerResidu(ctxB, R.kv, w, h, echelleResidu(R.kv, lung)), 0, 0);
  ctxB.strokeStyle = "rgba(226,166,49,.9)"; ctxB.lineWidth = 1.5; ctxB.setLineDash([4, 4]);
  ctxB.beginPath(); ctxB.arc(cx, cy, 40, 0, 2 * Math.PI); ctxB.stroke();
  const poser = (p) => { fig.style.setProperty("--pos", p + "%"); curseur.value = p; };
  curseur.addEventListener("input", () => poser(+curseur.value));
  const fin = Math.round(((cx - 4) / w) * 100);
  if (reduit || !("IntersectionObserver" in window)) { poser(fin); return; }
  poser(100);
  let t0; const D = 1800;
  const pas = (t) => {
    if (t0 === undefined) t0 = t;
    const u = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - u, 3);
    poser(100 - (100 - fin) * e);
    if (u < 1) requestAnimationFrame(pas);
  };
  const vu = new IntersectionObserver((e) => {     // le volet balaie la coupe quand la figure devient visible
    if (!e[0].isIntersecting) return;
    vu.disconnect(); setTimeout(() => requestAnimationFrame(pas), 250);
  }, { threshold: 0.45 });
  vu.observe(fig);
}

// ------------------------------------------------------------ état
const etat = {
  couche: "image", attaque: "aucune", cible: null, mode: "complet", nom: "", sha: "", meta: null, exemple: null,
  tau1: stock.lire("tau1", 0.5), tau2: stock.lire("tau2", 0.85), jeton: 0, jetonSerie: 0, serie: null, sources: null,
  dejaJournalise: false,
};

const AIDES = {
  image: "La coupe en fenêtre pulmonaire. Les cadres ambrés marquent les trois zones les plus suspectes, avec leur score.",
  residu: "Le bruit seul : un filtre retire l'anatomie. Une région lisse ou trop granuleuse s'y voit à l'œil.",
  bruit: "Niveau local du bruit dans le poumon, comparé au niveau habituel de cette coupe. En bleu, le bruit manque (région lissée ou générée) ; en ambre, il y en a trop (bruit ajouté).",
  proba: "Probabilité de falsification de chaque zone de 64 × 64 pixels. Les zones se recouvrent : la carte est leur moyenne.",
  probaComplet: "Probabilité de falsification de chaque zone après la fusion de l'analyse du bruit et du réseau de neurones.",
  cnn: "La région que le réseau de neurones juge falsifiée (carte de segmentation).",
  copie: "Pixels recopiés à l'identique à un autre endroit de la coupe : la signature d'un copier-déplacer. Un trait relie la source à la copie.",
};
const couleurEtat = (s) => (s > etat.tau2 ? G.COUL.alerte : s > etat.tau1 ? G.COUL.verifier : G.COUL.sain);

function legende(couche) {
  const L = $("#legende");
  if (couche === "bruit") L.innerHTML = '<span>bruit manquant</span><span class="degrade" style="background:linear-gradient(90deg,rgb(63,151,207),rgba(63,151,207,0) 42%,rgba(226,166,49,0) 58%,rgb(226,166,49))"></span><span>bruit en excès</span>';
  else if (couche === "proba" || couche === "cnn") L.innerHTML = '<span>0</span><span class="degrade" style="background:linear-gradient(90deg,rgba(222,64,46,0),rgb(222,64,46))"></span><span>1</span>';
  else L.innerHTML = "";
}

function dessinerCouche(ctx, couche) {
  const a = etat.res, hu = etat.huVu, w = etat.w, h = etat.h, comp = etat.complet;
  const im = couche === "residu" && a ? dessinerResidu(ctx, a.R.kv, w, h, echelleResidu(a.R.kv, a.lung)) : dessinerImage(ctx, hu, w, h);
  if (a && a.ok && couche === "proba") {
    const carte = comp ? comp.carteFusion : a.carte;
    for (let i = 0; i < w * h; i++) if (carte[i] > 0.02) melanger(im, i, ROUGE, Math.min(0.85, carte[i] * 0.9));
  }
  if (a && a.ok && couche === "cnn" && comp) {
    for (let i = 0; i < w * h; i++) if (comp.carteCnn[i] > 0.05) melanger(im, i, ROUGE, Math.min(0.85, comp.carteCnn[i] * 0.9));
  }
  if (a && a.lung && couche === "bruit" && etat.carteB) {
    const r = etat.carteB.ratio;
    for (let i = 0; i < w * h; i++) {
      if (!a.lung[i]) continue;
      const l = Math.log2(Math.max(r[i], 1e-3));
      if (l < -0.6) melanger(im, i, FROID, Math.min(0.9, (-l - 0.6) * 0.45));
      else if (l > 0.6) melanger(im, i, AMBRE, Math.min(0.9, (l - 0.6) * 0.45));
    }
  }
  if (a && a.ok && couche === "copie") for (let i = 0; i < w * h; i++) if (a.dup[i]) melanger(im, i, AMBRE, 0.75);
  ctx.putImageData(im, 0, 0);
  if (a && a.ok && couche === "image") {
    for (const { i, v } of zonesSuspectes()) {
      const [y, x] = a.coords[i];
      ctx.lineWidth = 2; ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.fillStyle = "rgba(226,166,49,.95)"; ctx.font = "12px Annot, monospace";
      ctx.strokeRect(x + 1, y + 1, 62, 62); ctx.fillText(fr(v, 2), x + 4, y + 14);
    }
  }
  if (a && a.ok && couche === "copie" && a.groups.length) {
    ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    const vus = new Set();
    for (const g of a.groups) {
      const [y0, x0] = g[0], [y1, x1] = g[g.length - 1], k = `${y1 - y0},${x1 - x0}`;
      if (vus.has(k) || vus.size > 6) continue;
      vus.add(k); ctx.beginPath(); ctx.moveTo(x0 + 4, y0 + 4); ctx.lineTo(x1 + 4, y1 + 4); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  if (etat.cible && etat.attaque !== "aucune" && couche !== "copie") {
    const [cy, cx] = etat.cible;
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.arc(cx, cy, 36, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
}

function rendre() {
  const c = $("#labo-canvas");
  if (!etat.huVu) return;
  if (c.width !== etat.w || c.height !== etat.h) { c.width = etat.w; c.height = etat.h; }
  dessinerCouche(c.getContext("2d"), etat.couche);
  const aide = etat.couche === "proba" && etat.complet ? AIDES.probaComplet : AIDES[etat.couche];
  const placer = etat.attaque !== "aucune" && etat.attaque !== "suppression" ? " Touchez le poumon pour déplacer la falsification simulée." : "";
  $("#aide-film").textContent = aide + placer;
  legende(etat.couche);
  annotBas();
}
function imageCouche(couche) {
  const c = document.createElement("canvas"); c.width = etat.w; c.height = etat.h;
  dessinerCouche(c.getContext("2d"), couche);
  return c.toDataURL("image/png");
}

const verdictDe = (s) => (s > etat.tau2 ? ["Alerte", "alerte"] : s > etat.tau1 ? ["À vérifier", "verifier"] : ["Conforme", "conforme"]);
const probaZones = () => (etat.complet ? Array.from(etat.complet.pLoc) : etat.res && etat.res.ok ? Array.from(etat.res.p) : []);
function zonesSuspectes(n = 3) {
  return probaZones().map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, n);
}

// ------------------------------------------------------------ verdict
function afficherVerdict() {
  const a = etat.res, mot = $("#verdict-mot"), comp = etat.complet, carte = $(".verdict-carte");
  $("#verdict-etiquette").textContent = comp ? (comp.avecDiffusion ? "Verdict de l'analyse complète, avec diffusion" : "Verdict de l'analyse complète") : "Verdict de l'analyse rapide";
  $("#score-def").textContent = comp ? "probabilité de falsification (0 : authentique, 1 : falsifiée)" : "score de falsification (moyenne des 3 zones les plus suspectes)";
  if (!a || !a.ok) {
    mot.textContent = "Non analysable"; mot.className = "verdict-mot"; carte.dataset.etat = "";
    $("#score").textContent = "—"; $("#m-patchs").textContent = "0"; $("#m-copie").textContent = "—";
    $("#jauge-valeur").style.left = "0%"; $("#comparaison").hidden = true; $("#branches").hidden = true;
    if (a) $("#aide-film").textContent = a.message + " Il faut une coupe thoracique où les poumons sont visibles.";
    afficherAnalyse();
    return;
  }
  const score = comp ? comp.score : a.score, [txt, cl] = verdictDe(score);
  mot.textContent = txt; mot.className = "verdict-mot " + cl; carte.dataset.etat = cl;
  $("#score").textContent = fr(score);
  $("#jauge-valeur").style.left = (score * 100).toFixed(1) + "%";
  $("#m-patchs").textContent = a.coords.length;
  $("#m-copie").textContent = a.dupPx ? `oui, ${a.dupPx} pixels` : "non";
  $("#m-duree").textContent = Math.round(a.duree + (comp ? comp.duree : 0)) + " ms";
  const br = $("#branches");
  if (comp) {
    br.hidden = false;
    br.innerHTML = [
      ["Analyse du bruit", comp.branches.forensique, "3 zones les plus suspectes"],
      ["Réseau de neurones", comp.branches.cnn, "3 zones les plus suspectes"],
      ["Branche diffusion", comp.avecDiffusion ? comp.branches.diffusion : NaN, comp.avecDiffusion ? "écart robuste de reconstruction" : "non calculée"],
      ["Fusion des zones", comp.branches.locale, "3 zones les plus suspectes"],
    ].map(([n, v, t]) => `<div><dt>${n}<small>${t}</small></dt><dd>${fr(v)}</dd></div>`).join("");
  } else br.hidden = true;
  const cp = $("#comparaison"), av = $("#jauge-avant"), ref = etat.refOrig;
  if (etat.attaque !== "aucune" && ref !== undefined) {
    const [t0] = verdictDe(ref);
    cp.hidden = false;
    cp.innerHTML = `Simulation. Coupe d'origine : <b>${fr(ref)}</b> (${t0.toLowerCase()}).<br>Après ${ATTAQUES[etat.attaque].court} : <b>${fr(score)}</b> (${txt.toLowerCase()}).` +
      (score <= etat.tau1 ? "<br>Cette falsification passe sous le seuil" + (comp ? "." : " de l'analyse rapide : essayez l'analyse complète, ou regardez le niveau du bruit.") : "");
    av.hidden = false; av.style.left = (ref * 100).toFixed(1) + "%";
  } else if (etat.exemple && etat.attaque === "aucune" && comp && etat.exemple.python) {
    const r0 = comp.avecDiffusion ? etat.exemple.python.trois_branches : etat.exemple.python.bruit_cnn;
    cp.hidden = false;
    cp.innerHTML = `Contrôle : le même détecteur en Python donne <b>${fr(r0)}</b> pour cette coupe` + (comp.avecDiffusion ? " (la diffusion tire un autre bruit aléatoire : petits écarts possibles)." : ".");
    av.hidden = true;
  } else { cp.hidden = true; av.hidden = true; }
  afficherAnalyse();
}

// ------------------------------------------------------------ constats en langage clair
function statsBruit() {
  const a = etat.res, r = etat.carteB && etat.carteB.ratio;
  if (!a || !a.lung || !r) return null;
  let n = 0, manque = 0, exces = 0;
  const vals = [];
  for (let i = 0; i < r.length; i += 3) {
    if (!a.lung[i]) continue;
    const l = Math.log2(Math.max(r[i], 1e-3));
    n++; vals.push(l);
    if (l < -1) manque++; else if (l > 1) exces++;
  }
  return n ? { vals, manque: (100 * manque) / n, exces: (100 * exces) / n } : null;
}
function lieu(i) {
  const [y, x] = etat.res.coords[i], cx = x + P / 2, cy = y + P / 2;
  const cote = cx < etat.w / 2 ? "poumon droit du patient (à gauche de l'image)" : "poumon gauche du patient (à droite de l'image)";
  const haut = cy < etat.h / 3 ? "en haut" : cy < (2 * etat.h) / 3 ? "au milieu" : "en bas";
  return `dans le ${cote}, ${haut} de l'image`;
}
function constats() {
  const a = etat.res, comp = etat.complet;
  if (!a || !a.ok) return [a ? `La coupe n'a pas pu être analysée : ${a.message}` : "Aucune coupe analysée."];
  const score = comp ? comp.score : a.score, [txt, cl] = verdictDe(score), L = [];
  const seuil = cl === "conforme" ? `sous le seuil de vigilance (${fr(etat.tau1, 2)})` : cl === "verifier" ? `au-dessus du seuil de vigilance (${fr(etat.tau1, 2)})` : `au-dessus du seuil d'alerte (${fr(etat.tau2, 2)})`;
  L.push(`<b>${txt}.</b> Le score de falsification de la coupe est de <b>${fr(score)}</b>, ${seuil}.`);
  const z = zonesSuspectes(probaZones().length), au = z.filter((q) => q.v > etat.tau1).length, top = z[0];
  if (top) L.push(au
    ? `<b>${au} zone${au > 1 ? "s" : ""}</b> sur ${z.length} dépasse${au > 1 ? "nt" : ""} le seuil de vigilance. La plus suspecte (score ${fr(top.v, 2)}) se trouve ${lieu(top.i)}.`
    : `Aucune des ${z.length} zones du poumon ne dépasse le seuil de vigilance ; la plus haute (score ${fr(top.v, 2)}) se trouve ${lieu(top.i)}.`);
  const sb = statsBruit();
  if (sb) L.push(`Bruit au moins deux fois plus faible que l'habituel de la coupe sur <b>${fr(sb.manque, 1)} %</b> du poumon, au moins deux fois plus fort sur <b>${fr(sb.exces, 1)} %</b>.`);
  L.push(a.dupPx ? `<b>${a.dupPx} pixels</b> sont recopiés à l'identique ailleurs dans la coupe : c'est la signature d'un copier-déplacer.` : "Aucune zone n'est recopiée à l'identique dans la coupe.");
  if (comp) {
    const fo = comp.branches.forensique > etat.tau1, cn = comp.branches.cnn > etat.tau1;
    L.push(fo && cn ? "L'analyse du bruit et le réseau de neurones signalent tous deux des zones suspectes."
      : fo ? "Seule l'analyse du bruit signale des zones suspectes ; le réseau de neurones ne dépasse pas le seuil."
      : cn ? "Seul le réseau de neurones signale des zones suspectes ; l'analyse du bruit ne dépasse pas le seuil."
      : "Ni l'analyse du bruit ni le réseau de neurones ne dépassent le seuil sur leurs zones les plus suspectes.");
  } else L.push(cl === "conforme" ? "Analyse rapide (bruit seul). L'analyse complète, plus fiable, ajoute le réseau de neurones : bouton « Complète » en haut."
    : "Analyse rapide (bruit seul) : elle donne plus de fausses alertes que l'analyse complète. Confirmez avec le bouton « Complète » en haut avant de conclure.");
  if (etat.attaque !== "aucune") L.push(`Simulation : la coupe a été falsifiée par l'application (${ATTAQUES[etat.attaque].court}) pour tester le détecteur.`);
  return L;
}

// ------------------------------------------------------------ graphiques de la coupe
function construireGraphes(largeur) {
  const a = etat.res, comp = etat.complet, p = probaZones();
  const seuils = [{ v: etat.tau1, nom: "vigilance", couleur: G.COUL.verifier }, { v: etat.tau2, nom: "alerte", couleur: G.COUL.alerte }];
  const lg = (id) => largeur || Math.max(280, Math.min(620, ($(id) && $(id).clientWidth) || 480));
  const g = {};
  g.zones = G.histogramme({
    valeurs: p, min: 0, max: 1, pas: 0.05, largeur: lg("#g-zones"), seuils, graduations: [0, 0.25, 0.5, 0.75, 1], formatX: (v) => fr(v, 2),
    couleurDe: (c) => (c > etat.tau2 ? G.COUL.alerte : c > etat.tau1 ? G.COUL.verifier : G.COUL.barre), unite: "zones",
  });
  const lignes = comp ? [
    { nom: "Analyse du bruit", valeur: comp.branches.forensique },
    { nom: "Réseau de neurones", valeur: comp.branches.cnn },
    { nom: "Fusion des zones", valeur: comp.branches.locale },
    { nom: "Score de la coupe", valeur: comp.score, fort: true, couleur: couleurEtat(comp.score) },
  ] : [
    { nom: "Analyse du bruit", valeur: a.score, fort: true, couleur: couleurEtat(a.score) },
    { nom: "Réseau de neurones", valeur: NaN },
  ];
  g.methodes = G.barres({ lignes: lignes.map((l) => ({ ...l, absent: !Number.isFinite(l.valeur) })), largeur: lg("#g-methodes"), seuils: seuils.map(({ v, couleur }) => ({ v, couleur })) });
  const sb = statsBruit();
  g.bruit = sb ? G.histogramme({
    valeurs: sb.vals, min: -2, max: 2, pas: 0.25, enPourcent: true, largeur: lg("#g-bruit"), graduations: [-2, -1, 0, 1, 2],
    formatX: (v) => "×" + fr(2 ** v, v < 0 ? 2 : 0), couleurDe: (c) => (c < -1 ? G.COUL.froid : c > 1 ? G.COUL.exces : G.COUL.gris),
    infoDe: (x0, x1, n, pct) => `bruit ×${fr(2 ** x0, 2)} à ×${fr(2 ** x1, 2)} de l'habituel : ${fr(pct, 1)} % du poumon`,
  }) : "";
  const sig = a.raw.map((r) => r[0]), med = mediane(sig) || 1, top = new Set(zonesSuspectes().map((z) => z.i));
  g.nuage = G.nuage({
    points: p.map((v, i) => {
      const [y, x] = a.coords[i];
      return { x: Math.log2(Math.max(sig[i] / med, 1e-3)), y: v, fort: top.has(i), couleur: top.has(i) ? couleurEtat(v) : G.COUL.barre,
        info: `Zone centrée en (${x + P / 2}, ${y + P / 2}) : bruit ×${fr(sig[i] / med, 2)} de l'habituel, score ${fr(v, 3)}` };
    }),
    xMin: -2, xMax: 2, gradX: [-2, -1, 0, 1, 2], formatX: (v) => "×" + fr(2 ** v, v < 0 ? 2 : 0), largeur: lg("#g-nuage"),
    titreX: "bruit de la zone (×1 = habituel dans la coupe)", titreY: "score de la zone", lignesX: [{ v: 0 }],
    seuilsY: seuils.map(({ v, couleur }) => ({ v, couleur })),
  });
  return g;
}

function afficherAnalyse() {
  const ul = $("#constats");
  ul.innerHTML = constats().map((c) => `<li>${c}</li>`).join("");
  const a = etat.res;
  if (!a || !a.ok) { for (const id of ["#g-zones", "#g-methodes", "#g-bruit", "#g-nuage", "#zones-suspectes"]) $(id).innerHTML = '<p class="vide">Pas de données pour cette coupe.</p>'; return; }
  const g = construireGraphes();
  $("#g-zones").innerHTML = g.zones; $("#g-methodes").innerHTML = g.methodes; $("#g-bruit").innerHTML = g.bruit; $("#g-nuage").innerHTML = g.nuage;
  $("#g-methodes-aide").textContent = etat.complet
    ? "Moyenne des trois zones les plus suspectes pour chaque méthode, puis le score final de la coupe donné par le méta-classifieur." + (etat.complet.avecDiffusion ? ` Branche diffusion : écart de ${fr(etat.complet.branches.diffusion, 2)} (ce n'est pas une probabilité).` : "")
    : "En analyse rapide, seule l'analyse du bruit est calculée. Passez en analyse complète pour ajouter le réseau de neurones.";
  // les trois zones les plus suspectes : image et bruit
  const box = $("#zones-suspectes"); box.innerHTML = "";
  const s = echelleResidu(a.R.kv, a.lung);
  zonesSuspectes().forEach(({ i, v }, k) => {
    const [y, x] = a.coords[i];
    const el = document.createElement("div"); el.className = "zone";
    const c1 = document.createElement("canvas"), c2 = document.createElement("canvas");
    for (const c of [c1, c2]) { c.width = P; c.height = P; }
    const x1 = c1.getContext("2d"), im1 = x1.createImageData(P, P), x2 = c2.getContext("2d"), im2 = x2.createImageData(P, P);
    for (let r = 0; r < P; r++) for (let q = 0; q < P; q++) {
      const j = (y + r) * etat.w + x + q, o = 4 * (r * P + q);
      const u = (etat.huVu[j] - FEN.lo) / (FEN.hi - FEN.lo), gi = u <= 0 ? 0 : u >= 1 ? 255 : (u * 255) | 0;
      let gr = 128 + (127 * a.R.kv[j]) / s; gr = gr < 0 ? 0 : gr > 255 ? 255 : gr;
      im1.data[o] = im1.data[o + 1] = im1.data[o + 2] = gi; im1.data[o + 3] = 255;
      im2.data[o] = im2.data[o + 1] = im2.data[o + 2] = gr; im2.data[o + 3] = 255;
    }
    x1.putImageData(im1, 0, 0); x2.putImageData(im2, 0, 0);
    const [txt, cl] = verdictDe(v);
    el.innerHTML = `<div class="zone-images"></div><div class="zone-texte"><p class="zone-rang">Zone ${k + 1}</p><p class="zone-score ${cl}">${fr(v, 2)} <small>${txt.toLowerCase()}</small></p><p class="zone-lieu">${lieu(i).replace(/^dans le /, "")}</p></div>`;
    el.querySelector(".zone-images").append(c1, c2);
    box.append(el);
  });
}

// ------------------------------------------------------------ progression du détecteur complet
function progression(texte, f) {
  const p = $("#progression");
  if (texte === null) { p.hidden = true; return; }
  p.hidden = false; $("#progression-texte").textContent = texte;
  const b = $("#progression-barre");
  if (f === undefined) b.removeAttribute("value"); else b.value = f;
}

async function preparerReseaux(montrer, jeton, jetonDe) {
  const M = await hybride();
  for (const nom of ["cnn.onnx", etat.avecDiffusion ? "ddpm.onnx" : null].filter(Boolean)) {
    const lib = `Téléchargement du réseau ${nom.replace(".onnx", "").toUpperCase()} (${M.TAILLES[nom]} Mo, une seule fois)…`;
    if (!(await M.reseauEnCache(nom))) { montrer(lib, 0); await M.session(nom, (i, n) => montrer(lib, n ? i / n : undefined)); }
    else { montrer("Chargement des réseaux…"); await M.session(nom); }
    if (jeton !== jetonDe()) return null;
  }
  if (!etat.cfgV7) { montrer("Chargement des classifieurs…"); etat.cfgV7 = await M.modelesV7(); }
  return M;
}

async function lancerComplet(jeton) {
  const M = await preparerReseaux(progression, jeton, () => etat.jeton);
  if (!M) return null;
  const comp = await M.analyseComplete(etat.huVu, etat.w, etat.h, etat.res, etat.cfgV7, {
    avecDiffusion: etat.avecDiffusion,
    etape: (b, i, n) => { if (jeton === etat.jeton) progression(b === "cnn" ? `Réseau de neurones : ${i} / ${n} zones` : `Branche diffusion : ${i} / ${n} passages`, n ? i / n : 0); },
  });
  return jeton === etat.jeton ? comp : null;
}

async function ganRemplir(y, x, cond) {
  const M = await hybride();
  if (!(await M.reseauEnCache("gan.onnx"))) progression("Téléchargement du générateur GAN (17 Mo, une seule fois)…", 0);
  else progression("Chargement du générateur GAN…");
  return M.ganRemplir(etat.hu, etat.w, y, x, cond, etat.cfgV7 || (etat.cfgV7 = await M.modelesV7()), (i, n) => progression("Téléchargement du générateur GAN (17 Mo, une seule fois)…", i / n));
}

function cibleNodule() {
  const n = etat.exemple && etat.exemple.nodules && etat.exemple.nodules[0];
  return n ? [Math.round(n[0]), Math.round(n[1])] : null;
}

// ------------------------------------------------------------ analyse de la coupe affichée
async function calculer() {
  const jeton = ++etat.jeton, voile = $("#calcul"), { w, h } = etat;
  voile.hidden = false;
  const etape = (t) => (voile.lastElementChild.textContent = t + "…");
  etape("Analyse"); await pause();
  etat.complet = null;
  try {
    if (!etat.resOrig) etat.resOrig = analyser(etat.hu, h, w, etat.model);
    if (etat.attaque === "suppression") {
      etat.cible = cibleNodule();
      if (!etat.cible) throw new Error("Pour effacer un nodule, choisissez un exemple « avec nodule ».");
    }
    if (etat.attaque !== "aucune" && !etat.cible) etat.cible = cibleParDefaut(etat.resOrig.lung, h, w);
    if (etat.attaque !== "aucune") {
      const r = await falsifier(etat.attaque, etat.hu, h, w, etat.resOrig.lung, etat.cible[0], etat.cible[1], ganRemplir);
      if (jeton !== etat.jeton) return;
      etat.huVu = r.hu; if (r.cible) etat.cible = r.cible;
    } else etat.huVu = etat.hu;
    await pause();
    etat.res = etat.attaque === "aucune" ? etat.resOrig : analyser(etat.huVu, h, w, etat.model, etape);
    etat.carteB = etat.res.R ? carteBruit(etat.res.R, etat.res.lung, h, w) : null;
    etat.refOrig = etat.resOrig.ok ? etat.resOrig.score : undefined;
    voile.hidden = true;
    rendre(); afficherVerdict();
    if (etat.mode === "complet" && etat.res.ok) {
      if (etat.attaque !== "aucune" && etat.resOrig.ok) {
        if (etat.compOrig === undefined || etat.compOrigDiff !== etat.avecDiffusion) {
          const sauve = etat.huVu, resF = etat.res;
          etat.huVu = etat.hu; etat.res = etat.resOrig;
          const c0 = await lancerComplet(jeton);
          etat.huVu = sauve; etat.res = resF;
          if (!c0) return;
          etat.compOrig = c0.score; etat.compOrigDiff = etat.avecDiffusion;
        }
        etat.refOrig = etat.compOrig;
      }
      const comp = await lancerComplet(jeton);
      if (!comp) return;
      etat.complet = comp;
      if (etat.attaque === "aucune") { etat.compOrig = comp.score; etat.compOrigDiff = etat.avecDiffusion; }
      progression(null);
      rendre(); afficherVerdict();
      majCoupeSerie(comp.score);
    } else progression(null);
    $(".couches [data-couche=cnn]").disabled = !etat.complet;
    if (!etat.complet && etat.couche === "cnn") choisirCouche("image");
    if (etat.attaque !== "aucune" || !etat.dejaJournalise) journaliser();
    etat.dejaJournalise = true;
  } catch (e) {
    console.error(e);
    voile.hidden = true;
    if (jeton !== etat.jeton) return;
    progression(null);
    $("#aide-film").textContent = e.message;
    if (etat.mode === "complet" && etat.res && etat.res.ok) {
      $("#comparaison").hidden = false;
      $("#comparaison").innerHTML = "L'analyse complète n'a pas pu tourner : " + e.message + " Le verdict affiché est celui de l'analyse rapide.";
    }
  }
}

async function charger(buf, nom, meta = null) {
  let d;
  try { d = parseDicom(buf); } catch (e) { montrerErreur(nom, e.message); return; }
  const hu = d.hu.map((v) => (v < -1024 ? -1024 : v > 3071 ? 3071 : v));   // lecture écrêtée de l'exécution v7
  Object.assign(etat, { hu, w: d.cols, h: d.rows, nom, meta: d, exemple: meta, resOrig: null, res: null, cible: null, compOrig: undefined });
  const effacer = document.querySelector("input[name=attaque][value=suppression]"), avecNodule = !!cibleNodule();
  effacer.disabled = !avecNodule;
  effacer.parentElement.title = avecNodule ? "" : "Disponible sur les exemples « avec nodule »";
  if (!avecNodule && etat.attaque === "suppression") { etat.attaque = "aucune"; document.querySelector("input[name=attaque][value=aucune]").checked = true; }
  const hash = await crypto.subtle.digest("SHA-256", buf).catch(() => null);
  const hex = hash ? Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("") : "";
  etat.sha = hex;
  $("#m-sha").textContent = hex ? hex.slice(0, 12) + "…" + hex.slice(-8) : "indisponible";
  $("#m-sha").title = hex;
  $("#annot-hg").innerHTML = `${(meta && meta.patient) || d.patient || "patient anonyme"}<br>${meta ? "coupe " + meta.coupe : Number.isFinite(d.instance) ? "coupe " + d.instance : nom}`;
  $("#annot-hd").innerHTML = `${(d.manufacturer || "fabricant inconnu").toLowerCase()}<br>noyau ${d.kernel || "?"}`;
  $("#annot-bg").innerHTML = `${d.cols} × ${d.rows}<br>F 1500 C −600`;
  majDossier();
  await calculer();
}

function montrerErreur(nom, msg) {
  etat.res = null; etat.huVu = null;
  $("#verdict-mot").textContent = "Fichier refusé"; $("#verdict-mot").className = "verdict-mot"; $(".verdict-carte").dataset.etat = "";
  $("#score").textContent = "—";
  $("#aide-film").textContent = msg;
  $("#constats").innerHTML = `<li><b>${nom}</b> n'a pas pu être lu : ${msg}</li>`;
  const c = $("#labo-canvas"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function annotBas() {
  const t = { image: "image", residu: "bruit (résidu KV 5×5)", bruit: "niveau du bruit", proba: "zones suspectes", cnn: "carte du réseau", copie: "copies exactes" }[etat.couche];
  $("#annot-bd").innerHTML = (etat.attaque !== "aucune" ? `simulation : ${ATTAQUES[etat.attaque].court}<br>` : "") + t;
}

function choisirCouche(c) {
  document.querySelectorAll(".couches button").forEach((o) => o.setAttribute("aria-selected", String(o.dataset.couche === c)));
  etat.couche = c; rendre();
}

const dateDicom = (s) => (s && /^\d{8}$/.test(s) ? `${s.slice(6, 8)}/${s.slice(4, 6)}/${s.slice(0, 4)}` : "");
function majDossier() {
  const d = etat.meta, S = etat.serie, ex = etat.exemple;
  if (S) {
    $("#dossier-etiquette").textContent = "Série";
    $("#dossier-titre").textContent = `${S.items.length} coupes analysées`;
  } else {
    $("#dossier-etiquette").textContent = ex ? "Exemple LIDC-IDRI" : "Coupe";
    $("#dossier-titre").textContent = ex ? `${ex.patient}, coupe ${ex.coupe}` : etat.nom;
  }
  const infos = d ? [d.patient && `patient ${d.patient}`, dateDicom(d.date), d.etude || d.serie, d.manufacturer, d.kernel && `noyau ${d.kernel}`].filter(Boolean) : [];
  $("#dossier-infos").textContent = infos.join(", ");
}

// ------------------------------------------------------------ séries
function trierSerie(items) {
  const tous = (k) => items.every((x) => x.ok !== false && Number.isFinite(x[k]));
  const cle = tous("instance") ? "instance" : tous("localisation") ? "localisation" : null;
  return [...items].sort((a, b) => (cle ? a[cle] - b[cle] : a.nom.localeCompare(b.nom, "fr", { numeric: true })));
}

async function lancerSerie(sources) {
  const jeton = ++etat.jetonSerie;
  etat.jeton++;                                             // arrête une analyse de coupe en cours
  const S = etat.serie = { items: sources.map((s, k) => ({ ...s, k, ok: null, score: NaN })), choisie: -1 };
  etat.sources = sources;
  montrerResultats();
  $("#serie").hidden = false; $("#export-csv").hidden = false;
  majDossier();
  const pr = $("#serie-progression"), prT = $("#serie-progression-texte"), prB = $("#serie-progression-barre");
  pr.hidden = false;
  const n = S.items.length;
  let M = null, echec = "";
  for (let k = 0; k < n; k++) {
    if (jeton !== etat.jetonSerie) return;
    const it = S.items[k];
    prT.textContent = `Analyse de la coupe ${k + 1} sur ${n}…`; prB.value = k / n;
    try {
      const buf = await it.lire(), d = parseDicom(buf), w = d.cols, h = d.rows;
      const hu = d.hu.map((v) => (v < -1024 ? -1024 : v > 3071 ? 3071 : v));
      const hash = await crypto.subtle.digest("SHA-256", buf).catch(() => null);
      it.sha = hash ? Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("") : "";
      Object.assign(it, { instance: d.instance, localisation: d.localisation, patient: d.patient, date: d.date, fabricant: d.manufacturer, noyau: d.kernel });
      const res = analyser(hu, h, w, etat.model);
      it.ok = res.ok; it.n = res.ok ? res.coords.length : 0; it.message = res.message;
      it.score = res.ok ? res.score : NaN; it.modele = "analyse rapide";
      if (res.ok && etat.mode === "complet" && !echec) {
        try {
          M = M || await preparerReseaux((t, f) => { prT.textContent = t; if (f !== undefined) prB.value = f; }, jeton, () => etat.jetonSerie);
          if (!M) return;
          prT.textContent = `Coupe ${k + 1} sur ${n} : réseau de neurones…`;
          const comp = await M.analyseComplete(hu, w, h, res, etat.cfgV7, { avecDiffusion: etat.avecDiffusion });
          if (jeton !== etat.jetonSerie) return;
          it.score = comp.score; it.modele = comp.avecDiffusion ? "analyse complète avec diffusion" : "analyse complète";
        } catch (e) { echec = e.message; console.error(e); }
      }
      if (res.ok) await ajouterJournal({ fichier: it.nom, empreinte: it.sha, falsification: "", modele: it.modele, score: it.score });
    } catch (e) { it.ok = false; it.message = e.message; }
    if (k % 3 === 2 || k === n - 1) afficherSerie();
    await pause();
  }
  pr.hidden = true;
  if (echec) { const al = $("#serie-avis") || Object.assign(document.createElement("p"), { id: "serie-avis", className: "serie-avis" }); al.textContent = `L'analyse complète n'a pas pu tourner (${echec}) : les scores affichés sont ceux de l'analyse rapide.`; $("#serie-progression").after(al); }
  else if ($("#serie-avis")) $("#serie-avis").remove();
  S.items = trierSerie(S.items);
  afficherSerie();
  const ok = S.items.map((x, i) => ({ x, i })).filter(({ x }) => x.ok);
  if (!ok.length) { montrerErreur("La série", "aucune coupe n'a pu être analysée (fichiers non DICOM, compressés, ou sans poumon visible)."); return; }
  const pire = ok.reduce((a, b) => (b.x.score > a.x.score ? b : a));
  await choisirCoupe(pire.i);
}

function afficherSerie() {
  const S = etat.serie;
  if (!S) return;
  const it = S.items, faits = it.filter((x) => x.ok !== null);
  const c = { conforme: 0, verifier: 0, alerte: 0, non: 0 };
  for (const x of faits) { if (!x.ok) c.non++; else c[verdictDe(x.score)[1]]++; }
  $("#serie-resume").innerHTML =
    `<span class="pastille-etat conforme"><b>${c.conforme}</b> conforme${c.conforme > 1 ? "s" : ""}</span>` +
    `<span class="pastille-etat verifier"><b>${c.verifier}</b> à vérifier</span>` +
    `<span class="pastille-etat alerte"><b>${c.alerte}</b> alerte${c.alerte > 1 ? "s" : ""}</span>` +
    (c.non ? `<span class="pastille-etat non"><b>${c.non}</b> non analysable${c.non > 1 ? "s" : ""}</span>` : "");
  const larg = Math.max(300, $("#graphe-serie").clientWidth || 900);
  $("#graphe-serie").innerHTML = G.serie({
    coupes: it.map((x, i) => ({
      ok: !!x.ok, score: x.score, couleur: x.ok ? couleurEtat(x.score) : G.COUL.gris,
      etiquette: Number.isFinite(x.instance) ? String(x.instance) : String(i + 1),
      info: x.ok === null ? `${x.nom} : en attente` : x.ok ? `${x.nom}${Number.isFinite(x.instance) ? " (coupe " + x.instance + ")" : ""} : ${fr(x.score)}, ${verdictDe(x.score)[0].toLowerCase()}` : `${x.nom} : non analysable (${x.message || "erreur"})`,
    })),
    largeur: larg, seuils: [{ v: etat.tau1, couleur: G.COUL.verifier, nom: "vigilance" }, { v: etat.tau2, couleur: G.COUL.alerte, nom: "alerte" }], choisie: S.choisie,
  });
  $("#serie-position").textContent = S.choisie >= 0 ? `Coupe ${S.choisie + 1} sur ${it.length} : ${it[S.choisie].nom}` : "";
  $("#coupe-prec").disabled = S.choisie <= 0; $("#coupe-suiv").disabled = S.choisie < 0 || S.choisie >= it.length - 1;
}

async function choisirCoupe(i) {
  const S = etat.serie;
  if (!S || !S.items[i]) return;
  S.choisie = i; afficherSerie();
  const it = S.items[i];
  if (!it.ok) { montrerErreur(it.nom, it.message || "coupe non analysable."); return; }
  etat.attaque = "aucune"; document.querySelector("input[name=attaque][value=aucune]").checked = true;
  etat.dejaJournalise = true;
  await charger(await it.lire(), it.nom, it.meta || null);
}
function majCoupeSerie(score) {
  const S = etat.serie;
  if (!S || S.choisie < 0 || etat.attaque !== "aucune") return;
  const it = S.items[S.choisie];
  if (it.modele === "analyse rapide") { it.score = score; it.modele = etat.avecDiffusion ? "analyse complète avec diffusion" : "analyse complète"; afficherSerie(); }
}

// ------------------------------------------------------------ fichiers déposés ou choisis
const NON_DICOM = /(^|\/)(DICOMDIR|\.DS_Store|Thumbs\.db)$|\.(txt|xml|json|jpe?g|png|gif|bmp|pdf|html?|ini|exe|dll|zip|rar|7z|csv|log|inf|autorun)$/i;
function sourcesDepuisFichiers(fichiers) {
  return [...fichiers]
    .filter((f) => !NON_DICOM.test(f.chemin || f.webkitRelativePath || f.name) && f.size > 2048)
    .map((f) => ({ nom: f.name, chemin: f.chemin || f.webkitRelativePath || f.name, lire: () => f.arrayBuffer(), meta: null }))
    .sort((a, b) => a.chemin.localeCompare(b.chemin, "fr", { numeric: true }));
}
async function fichiersDepuisDepot(dt) {
  const entrees = [...(dt.items || [])].map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
  if (!entrees.length) return [...dt.files];
  const out = [];
  const lire = async (e) => {
    if (e.isFile) out.push(await new Promise((ok, ko) => e.file((f) => { try { Object.defineProperty(f, "chemin", { value: e.fullPath }); } catch { /* lecture seule */ } ok(f); }, ko)));
    else if (e.isDirectory) {
      const r = e.createReader();
      let lot;
      do { lot = await new Promise((ok, ko) => r.readEntries(ok, ko)); for (const x of lot) await lire(x); } while (lot.length);
    }
  };
  for (const e of entrees) await lire(e);
  return out;
}
async function analyserSources(sources) {
  if (!sources.length) { alerteAccueil("Aucun fichier DICOM trouvé dans ce choix. Il faut des fichiers .dcm (ou sans extension) tels qu'ils sortent du scanner."); return; }
  if (sources.length === 1) {
    etat.jetonSerie++; etat.serie = null; etat.sources = sources;
    $("#serie").hidden = true; $("#export-csv").hidden = true;
    montrerResultats();
    etat.attaque = "aucune"; document.querySelector("input[name=attaque][value=aucune]").checked = true;
    etat.dejaJournalise = false;
    await charger(await sources[0].lire(), sources[0].nom, sources[0].meta);
  } else await lancerSerie(sources);
}
function alerteAccueil(msg) {
  let p = $("#accueil-alerte");
  if (!p) { p = document.createElement("p"); p.id = "accueil-alerte"; p.className = "accueil-alerte"; p.setAttribute("role", "alert"); $("#zone-depot").append(p); }
  p.textContent = msg;
}

function montrerResultats() {
  $("#accueil").hidden = true; $("#resultats").hidden = false;
  if (location.hash !== "#analyser") history.replaceState(null, "", "#analyser");
  afficherPage();
  window.scrollTo(0, 0);
}
function nouvelleAnalyse() {
  etat.jeton++; etat.jetonSerie++; etat.serie = null; etat.sources = null;
  progression(null);
  $("#resultats").hidden = true; $("#accueil").hidden = false;
  $("#fichier").value = ""; $("#dossier").value = "";
  const al = $("#accueil-alerte"); if (al) al.remove();
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------ rapport et export
function rapport() {
  const a = etat.res, comp = etat.complet, d = etat.meta;
  if (!a || !a.ok || !d) return;
  const score = comp ? comp.score : a.score, [txt, cl] = verdictDe(score);
  const g = construireGraphes(360);
  const S = etat.serie;
  imprimerRapport({
    titre: etat.exemple ? `${etat.exemple.patient}, coupe ${etat.exemple.coupe}` : etat.nom,
    nomFichier: etat.nom, verdict: txt, classe: cl, score: fr(score),
    definition: comp ? "probabilité de falsification donnée par le méta-classifieur (0 : authentique, 1 : falsifiée)" : "moyenne des 3 zones les plus suspectes de l'analyse du bruit",
    tau1: fr(etat.tau1, 2), tau2: fr(etat.tau2, 2),
    simulation: etat.attaque !== "aucune" ? ATTAQUES[etat.attaque].court : "",
    fichier: [
      ["Fichier", etat.nom], ["Identifiant patient (DICOM)", d.patient], ["Date de l'examen", dateDicom(d.date)], ["Description", d.etude || d.serie],
      ["Fabricant", d.manufacturer], ["Noyau de reconstruction", d.kernel], ["Numéro de coupe", Number.isFinite(d.instance) ? d.instance : ""],
      ["Image", `${d.cols} × ${d.rows} pixels`], ["Zones du poumon analysées", a.coords.length],
      ["Analyse", comp ? (comp.avecDiffusion ? "complète : bruit, réseau de neurones et diffusion" : "complète : bruit et réseau de neurones") : "rapide : analyse du bruit"],
      ["Empreinte SHA-256 du fichier", etat.sha],
    ],
    constats: constats(),
    imageCoupe: imageCouche("image"), imageCarte: imageCouche("proba"),
    graphes: [
      { titre: "Score des zones du poumon", svg: g.zones }, { titre: "Avis de chaque méthode", svg: g.methodes },
      { titre: "Niveau du bruit dans le poumon", svg: g.bruit }, { titre: "Bruit et score de chaque zone", svg: g.nuage },
    ].filter((x) => x.svg),
    serie: S ? {
      svg: G.serie({ coupes: S.items.map((x, i) => ({ ok: !!x.ok, score: x.score, couleur: x.ok ? couleurEtat(x.score) : G.COUL.gris, etiquette: Number.isFinite(x.instance) ? String(x.instance) : String(i + 1), info: "" })), largeur: 680, hauteur: 180, seuils: [{ v: etat.tau1, couleur: G.COUL.verifier, nom: "vigilance" }, { v: etat.tau2, couleur: G.COUL.alerte, nom: "alerte" }], choisie: S.choisie }),
      lignes: S.items.map((x, i) => ({ ordre: i + 1, nom: x.nom, score: x.ok ? fr(x.score) : "—", verdict: x.ok ? verdictDe(x.score)[0] : "non analysable" })),
    } : null,
  });
}
function exporterSerie() {
  const S = etat.serie;
  if (!S) return;
  exporterCSV(S.items.map((x, i) => ({
    ordre: i + 1, fichier: x.chemin || x.nom, coupe: Number.isFinite(x.instance) ? x.instance : "", score: x.ok ? x.score.toFixed(4) : "",
    verdict: x.ok ? verdictDe(x.score)[0] : "non analysable", analyse: x.modele || "", zones: x.n ?? "", empreinte_sha256: x.sha || "", remarque: x.ok ? "" : x.message || "",
  })), "analyse-serie.csv");
}

// ------------------------------------------------------------ historique chaîné (module de sécurité)
const CLE_J = "journal-audit-v1";
const trie = (o) => JSON.stringify(Object.keys(o).sort().reduce((a, k) => ((a[k] = o[k]), a), {}));
async function hacher(t) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(t));
  return Array.from(new Uint8Array(h), (b) => b.toString(16).padStart(2, "0")).join("");
}
async function ajouterJournal({ fichier, empreinte, falsification, modele, score }) {
  if (!crypto.subtle) return;
  const j = stock.lire(CLE_J, []);
  const rec = {
    time: new Date().toISOString().slice(0, 19), fichier, empreinte, falsification, modele,
    score: Math.round(score * 10000) / 10000, verdict: verdictDe(score)[0],
    prev_hash: j.length ? j[j.length - 1].entry_hash : "0".repeat(64),
  };
  rec.entry_hash = await hacher(trie(rec));
  j.push(rec);
  stock.ecrire(CLE_J, j.slice(-500));
  afficherJournal();
}
async function journaliser() {
  if (!etat.res || !etat.res.ok) return;
  const score = etat.complet ? etat.complet.score : etat.res.score;
  await ajouterJournal({
    fichier: etat.nom, empreinte: etat.sha, falsification: etat.attaque === "aucune" ? "" : ATTAQUES[etat.attaque].court,
    modele: etat.complet ? (etat.complet.avecDiffusion ? "analyse complète avec diffusion" : "analyse complète") : "analyse rapide", score,
  });
}
function afficherJournal() {
  const j = stock.lire(CLE_J, []), tb = $("#j-lignes");
  if (!j.length) { tb.innerHTML = '<tr><td colspan="6" class="vide">Aucune analyse pour l\'instant.</td></tr>'; return; }
  const cl = { Conforme: "conforme", "À vérifier": "verifier", Alerte: "alerte" };
  tb.innerHTML = j.slice(-100).reverse().map((r) =>
    `<tr><td>${r.time.slice(8, 10)}/${r.time.slice(5, 7)} ${r.time.slice(11, 16)}</td><td>${r.fichier.replace(/\.dcm$/i, "")}${r.falsification ? "<small>simulation : " + r.falsification + "</small>" : ""}</td><td>${r.modele.replace("branche du bruit", "analyse rapide")}</td><td>${fr(r.score)}</td><td><span class="pastille-etat ${cl[r.verdict] || ""}">${r.verdict}</span></td><td class="sha">${r.entry_hash.slice(0, 10)}…</td></tr>`).join("");
}
async function verifierJournal() {
  const j = stock.lire(CLE_J, []);
  let prev = "0".repeat(64);
  for (let i = 0; i < j.length; i++) {
    const { entry_hash, ...rec } = j[i];
    if (rec.prev_hash !== prev || (await hacher(trie(rec))) !== entry_hash) return `Chaîne rompue à l'entrée ${i + 1} : l'historique a été modifié.`;
    prev = entry_hash;
  }
  return j.length ? `Chaîne intacte : ${j.length} entrée${j.length > 1 ? "s" : ""} vérifiée${j.length > 1 ? "s" : ""}.` : "L'historique est vide.";
}

// ------------------------------------------------------------ pages
const PAGES = ["analyser", "historique", "comprendre"];
function afficherPage() {
  const id = (location.hash || "#analyser").slice(1), page = PAGES.includes(id) ? id : "analyser";
  document.querySelectorAll(".page").forEach((p) => (p.hidden = p.dataset.page !== page));
  document.querySelectorAll(".onglets a").forEach((a) => (a.dataset.page === page ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (page === "comprendre") voletAccueil().catch(() => {});
  if (page === "analyser" && etat.res) requestAnimationFrame(() => { afficherAnalyse(); afficherSerie(); });
}

// ------------------------------------------------------------ mise en place
function installerInterface(exemples) {
  const sel = $("#choix-exemple");
  for (const e of exemples) {
    const o = document.createElement("option");
    o.value = e.fichier; o.textContent = `${e.patient}, coupe ${e.coupe}${e.nodules.length ? " (avec nodule)" : ""}`;
    sel.append(o);
  }
  const source = (e) => ({ nom: e.fichier, chemin: e.fichier, meta: e, lire: () => fetch("exemples/" + e.fichier).then((r) => r.arrayBuffer()) });
  $("#ouvrir-exemple").addEventListener("click", () => analyserSources([source(exemples.find((e) => e.fichier === sel.value))]));
  $("#serie-exemples").addEventListener("click", () => analyserSources(exemples.map(source)));

  // dépôt de fichiers
  $("#fichier").addEventListener("change", (e) => analyserSources(sourcesDepuisFichiers(e.target.files)));
  $("#dossier").addEventListener("change", (e) => analyserSources(sourcesDepuisFichiers(e.target.files)));
  if (!("webkitdirectory" in document.createElement("input")) || matchMedia("(pointer: coarse)").matches) $(".choix-dossier").hidden = true;
  const zone = $("#zone-depot");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("survol"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("survol"));
  zone.addEventListener("drop", async (e) => { e.preventDefault(); zone.classList.remove("survol"); analyserSources(sourcesDepuisFichiers(await fichiersDepuisDepot(e.dataTransfer))); });
  const film = $("#film-labo"), depot = $("#depot");
  film.addEventListener("dragover", (e) => { e.preventDefault(); depot.hidden = false; });
  film.addEventListener("dragleave", () => (depot.hidden = true));
  film.addEventListener("drop", async (e) => { e.preventDefault(); depot.hidden = true; analyserSources(sourcesDepuisFichiers(await fichiersDepuisDepot(e.dataTransfer))); });

  // simulation de falsification
  document.querySelectorAll("input[name=attaque]").forEach((r) => r.addEventListener("change", async () => {
    if (!r.checked || !etat.hu) return;
    etat.attaque = r.value; etat.cible = null;
    await calculer();
  }));
  film.addEventListener("click", async (e) => {
    if (etat.attaque === "aucune" || etat.attaque === "suppression" || !etat.resOrig) return;
    const c = $("#labo-canvas").getBoundingClientRect();
    const x = Math.round(((e.clientX - c.left) / c.width) * etat.w), y = Math.round(((e.clientY - c.top) / c.height) * etat.h);
    if (!etat.resOrig.lung[y * etat.w + x]) { $("#aide-film").textContent = "Placez la falsification dans le poumon : c'est là que le détecteur cherche."; return; }
    etat.cible = [y, x]; await calculer();
  });

  // type d'analyse
  etat.mode = stock.lire("mode", "complet");
  etat.avecDiffusion = stock.lire("diffusion-v2", false);
  $("#avec-diffusion").checked = etat.avecDiffusion;
  document.querySelector(`input[name=mode][value=${etat.mode}]`).checked = true;
  $(".modele").classList.toggle("complet", etat.mode === "complet");
  const relancer = async () => {
    if (etat.serie && etat.sources) await lancerSerie(etat.sources);
    else if (etat.hu) await calculer();
  };
  document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", async () => {
    etat.mode = r.value; stock.ecrire("mode", r.value);
    $(".modele").classList.toggle("complet", etat.mode === "complet");
    etat.dejaJournalise = false;
    await relancer();
  }));
  $("#avec-diffusion").addEventListener("change", async (e) => {
    etat.avecDiffusion = e.target.checked; stock.ecrire("diffusion-v2", etat.avecDiffusion);
    etat.dejaJournalise = false;
    if (etat.mode === "complet") await relancer();
  });
  $("#annuler").addEventListener("click", () => {
    etat.jeton++; progression(null);
    document.querySelector("input[name=mode][value=rapide]").checked = true;
    etat.mode = "rapide"; stock.ecrire("mode", "rapide");
    $(".modele").classList.remove("complet");
    etat.complet = null; rendre(); afficherVerdict();
  });

  // seuils
  const majSeuils = () => {
    $("#v-tau1").textContent = fr(etat.tau1, 2); $("#v-tau2").textContent = fr(etat.tau2, 2);
    $("#seuil1").style.left = etat.tau1 * 100 + "%"; $("#seuil1").textContent = fr(etat.tau1, 2);
    $("#seuil2").style.left = etat.tau2 * 100 + "%"; $("#seuil2").textContent = fr(etat.tau2, 2);
    $(".jauge").style.setProperty("--t1", etat.tau1 * 100 + "%"); $(".jauge").style.setProperty("--t2", etat.tau2 * 100 + "%");
  };
  $("#tau1").value = etat.tau1; $("#tau2").value = etat.tau2; majSeuils();
  const apresSeuil = () => { majSeuils(); if (etat.res) afficherVerdict(); afficherSerie(); };
  $("#tau1").addEventListener("input", (e) => { etat.tau1 = Math.min(+e.target.value, etat.tau2 - 0.01); stock.ecrire("tau1", etat.tau1); apresSeuil(); });
  $("#tau2").addEventListener("input", (e) => { etat.tau2 = Math.max(+e.target.value, etat.tau1 + 0.01); stock.ecrire("tau2", etat.tau2); apresSeuil(); });

  // couches de l'image
  const onglets = document.querySelectorAll(".couches button");
  onglets.forEach((b) => b.addEventListener("click", () => choisirCouche(b.dataset.couche)));
  $(".couches").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const l = [...onglets].filter((o) => !o.disabled), i = l.findIndex((o) => o.getAttribute("aria-selected") === "true");
    const n = l[(i + (e.key === "ArrowRight" ? 1 : l.length - 1)) % l.length]; n.focus(); n.click();
  });

  // série
  $("#graphe-serie").addEventListener("click", (e) => { const b = e.target.closest("[data-coupe]"); if (b) choisirCoupe(+b.dataset.coupe); });
  $("#coupe-prec").addEventListener("click", () => choisirCoupe(etat.serie.choisie - 1));
  $("#coupe-suiv").addEventListener("click", () => choisirCoupe(etat.serie.choisie + 1));
  $("#serie-arreter").addEventListener("click", () => {
    etat.jetonSerie++; $("#serie-progression").hidden = true;
    const S = etat.serie;
    if (!S) return;
    S.items = trierSerie(S.items.filter((x) => x.ok !== null));
    afficherSerie(); majDossier();
    const ok = S.items.map((x, i) => ({ x, i })).filter(({ x }) => x.ok);
    if (ok.length) choisirCoupe(ok.reduce((a, b) => (b.x.score > a.x.score ? b : a)).i);
  });

  // actions du dossier
  $("#rapport-pdf").addEventListener("click", rapport);
  $("#export-csv").addEventListener("click", exporterSerie);
  $("#nouvelle").addEventListener("click", nouvelleAnalyse);

  // historique
  afficherJournal();
  $("#j-verifier").addEventListener("click", async () => ($("#j-etat").textContent = await verifierJournal()));
  $("#j-exporter").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(stock.lire(CLE_J, []), null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "historique-analyses.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  $("#j-effacer").addEventListener("click", () => { stock.ecrire(CLE_J, []); afficherJournal(); $("#j-etat").textContent = "Historique effacé."; });

  // graphiques à la largeur de l'écran
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (etat.res && !$("#resultats").hidden) { afficherAnalyse(); afficherSerie(); } }, 180); });
  G.activerInfobulles($("#infobulle"));
}

// ------------------------------------------------------------ installation (application web progressive)
function installation() {
  if ("serviceWorker" in navigator && location.protocol !== "file:") navigator.serviceWorker.register("sw.js").catch(() => {});
  let invite = null;
  addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); invite = e; $("#installer").hidden = false; });
  $("#installer").addEventListener("click", async () => {
    if (!invite) return;
    invite.prompt(); await invite.userChoice.catch(() => {}); invite = null; $("#installer").hidden = true;
  });
  addEventListener("appinstalled", () => ($("#installer").hidden = true));
}

// ------------------------------------------------------------ démarrage
(async () => {
  installation();
  addEventListener("hashchange", afficherPage);
  afficherPage();
  try {
    const [model, exemples] = await Promise.all([
      fetch("modele/v7/hgb_forensique.json").then((r) => r.json()),
      fetch("exemples/index.json").then((r) => r.json()),
    ]);
    etat.model = model;
    installerInterface(exemples);
  } catch (e) {
    console.error(e);
    alerteAccueil("Le chargement a échoué (" + e.message + "). Rechargez la page.");
  }
})();

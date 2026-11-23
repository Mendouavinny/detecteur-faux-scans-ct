// Interface de l'application : accueil (volet image / bruit), laboratoire d'analyse, journal d'audit, installation.
import { parseDicom, analyser, lungMask, residuals, carteBruit } from "./forensique.js";
import { falsifier, cibleParDefaut, noduleLisse, ATTAQUES } from "./attaques.js";

const $ = (s) => document.querySelector(s);
const fr = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",").replace("-", "−") : "—");
const reduit = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pause = () => new Promise((r) => setTimeout(r, 0));
let H = null;                                            // module du détecteur complet, chargé à la demande
const hybride = async () => (H ||= await import("./hybride.js"));

// ------------------------------------------------------------ stockage local (peut être refusé)
const stock = {
  lire(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  ecrire(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* navigation privée */ } },
};

// ------------------------------------------------------------ rendu
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
const ROUGE = [222, 64, 46], FROID = [86, 172, 222], AMBRE = [226, 166, 49];

// ------------------------------------------------------------ accueil : volet image / bruit
async function accueil() {
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
  // un seul mouvement à l'ouverture : le volet balaie la coupe, comme l'anneau du scanner, et s'arrête sur le nodule
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
  const vu = new IntersectionObserver((e) => {   // le volet balaie la coupe quand on arrive sur la figure
    if (!e[0].isIntersecting) return;
    vu.disconnect(); setTimeout(() => requestAnimationFrame(pas), 250);
  }, { threshold: 0.45 });
  vu.observe(fig);
}

// ------------------------------------------------------------ laboratoire
const etat = {
  couche: "image", attaque: "aucune", cible: null, mode: "rapide", nom: "", sha: "", meta: null, exemple: null,
  tau1: stock.lire("tau1", 0.5), tau2: stock.lire("tau2", 0.85), jeton: 0,
};

const AIDES = {
  image: "La coupe en fenêtre pulmonaire. Les carrés ambrés marquent les trois patchs les plus suspects.",
  residu: "Le résidu du filtre KV 5 × 5 : l'anatomie disparaît, il reste le bruit. Une région lisse ou trop granuleuse s'y voit à l'œil.",
  bruit: "Niveau local du bruit dans le poumon, rapporté au niveau habituel de cette coupe. En bleu, le bruit manque (région lissée ou générée) ; en ambre, il y en a trop (bruit ajouté).",
  proba: "Probabilité de falsification de chaque patch de 64 × 64 pixels. Les patchs se recouvrent : la carte est leur moyenne.",
  probaComplet: "Probabilité de falsification de chaque patch après la fusion locale du bruit, du CNN et de la diffusion.",
  cnn: "Carte de segmentation du décodeur U-Net de la branche CNN : la région que le réseau juge falsifiée.",
  copie: "Fenêtres de 8 × 8 pixels strictement identiques et éloignées : la signature d'un copier-déplacer. Un trait relie la source à la copie.",
};

function legende(couche) {
  const L = $("#legende");
  if (couche === "bruit") L.innerHTML = '<span>bruit manquant</span><span class="degrade" style="background:linear-gradient(90deg,rgb(86,172,222),rgba(86,172,222,0) 42%,rgba(226,166,49,0) 58%,rgb(226,166,49))"></span><span>bruit en excès</span>';
  else if (couche === "proba" || couche === "cnn") L.innerHTML = '<span>0</span><span class="degrade" style="background:linear-gradient(90deg,rgba(222,64,46,0),rgb(222,64,46))"></span><span>1</span>';
  else L.innerHTML = "";
}

function rendre() {
  const c = $("#labo-canvas"), ctx = c.getContext("2d"), a = etat.res, hu = etat.huVu, w = etat.w, h = etat.h;
  if (!hu) return;
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const comp = etat.complet;
  let im = etat.couche === "residu" && a ? dessinerResidu(ctx, a.R.kv, w, h, echelleResidu(a.R.kv, a.lung)) : dessinerImage(ctx, hu, w, h);
  if (a && a.ok && etat.couche === "proba") {
    const carte = comp ? comp.carteFusion : a.carte;
    for (let i = 0; i < w * h; i++) if (carte[i] > 0.02) melanger(im, i, ROUGE, Math.min(0.85, carte[i] * 0.9));
  }
  if (a && a.ok && etat.couche === "cnn" && comp) {
    for (let i = 0; i < w * h; i++) if (comp.carteCnn[i] > 0.05) melanger(im, i, ROUGE, Math.min(0.85, comp.carteCnn[i] * 0.9));
  }
  if (a && a.lung && etat.couche === "bruit" && etat.carteB) {
    const r = etat.carteB.ratio;
    for (let i = 0; i < w * h; i++) {
      if (!a.lung[i]) continue;
      const l = Math.log2(Math.max(r[i], 1e-3));
      if (l < -0.6) melanger(im, i, FROID, Math.min(0.9, (-l - 0.6) * 0.45));
      else if (l > 0.6) melanger(im, i, AMBRE, Math.min(0.9, (l - 0.6) * 0.45));
    }
  }
  if (a && a.ok && etat.couche === "copie") for (let i = 0; i < w * h; i++) if (a.dup[i]) melanger(im, i, AMBRE, 0.75);
  ctx.putImageData(im, 0, 0);
  if (a && a.ok && etat.couche === "image") {
    const p = comp ? comp.pLoc : a.p;
    const idx = Array.from(p, (v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 3);
    ctx.lineWidth = 2; ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.fillStyle = "rgba(226,166,49,.95)"; ctx.font = "12px Annot, monospace";
    for (const [v, i] of idx) { const [y, x] = a.coords[i]; ctx.strokeRect(x + 1, y + 1, 62, 62); ctx.fillText(fr(v, 2), x + 4, y + 14); }
  }
  if (a && a.ok && etat.couche === "copie" && a.groups.length) {
    ctx.strokeStyle = "rgba(226,166,49,.95)"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    const vus = new Set();
    for (const g of a.groups) {
      const [y0, x0] = g[0], [y1, x1] = g[g.length - 1], k = `${y1 - y0},${x1 - x0}`;
      if (vus.has(k) || vus.size > 6) continue;
      vus.add(k); ctx.beginPath(); ctx.moveTo(x0 + 4, y0 + 4); ctx.lineTo(x1 + 4, y1 + 4); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  if (etat.cible && etat.attaque !== "aucune" && etat.couche !== "copie") {
    const [cy, cx] = etat.cible;
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.arc(cx, cy, 36, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
  const aide = etat.couche === "proba" && comp ? AIDES.probaComplet : AIDES[etat.couche];
  const placer = etat.attaque !== "aucune" && etat.attaque !== "suppression" ? " Touchez le poumon pour déplacer la falsification." : "";
  $("#aide-film").textContent = aide + placer;
  legende(etat.couche);
  annotBas();
}

const verdictDe = (s) => (s > etat.tau2 ? ["Alerte", "alerte"] : s > etat.tau1 ? ["À vérifier", "verifier"] : ["Conforme", "conforme"]);

function afficherVerdict() {
  const a = etat.res, mot = $("#verdict-mot"), comp = etat.complet;
  $("#verdict-etiquette").textContent = comp ? "Verdict du détecteur complet" : "Verdict de la branche du bruit";
  $("#score-def").textContent = comp ? "probabilité de falsification donnée par le méta-classifieur (0 : authentique, 1 : falsifiée)" : "score de la coupe (moyenne des 3 patchs les plus suspects)";
  const carte = $(".verdict-carte");
  if (!a || !a.ok) {
    mot.textContent = "Non analysable"; mot.className = "verdict-mot"; carte.dataset.etat = "";
    $("#score").textContent = "—"; $("#m-patchs").textContent = "0"; $("#m-copie").textContent = "—";
    $("#jauge-valeur").style.left = "0%"; $("#comparaison").hidden = true; $("#branches").hidden = true;
    if (a) $("#aide-film").textContent = a.message + " Il faut une coupe thoracique où les poumons sont visibles.";
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
      ["Branche du bruit", comp.branches.forensique, "probabilité, 3 patchs les plus suspects"],
      ["Branche CNN", comp.branches.cnn, "probabilité, 3 patchs les plus suspects"],
      ["Branche diffusion", comp.avecDiffusion ? comp.branches.diffusion : NaN, comp.avecDiffusion ? "|écart robuste| de l'erreur de reconstruction" : "non calculée"],
      ["Fusion locale", comp.branches.locale, "probabilité, 3 patchs les plus suspects"],
    ].map(([n, v, t]) => `<div><dt>${n}<small>${t}</small></dt><dd>${fr(v)}</dd></div>`).join("");
  } else br.hidden = true;
  const cp = $("#comparaison"), av = $("#jauge-avant"), ref = etat.refOrig;
  if (etat.attaque !== "aucune" && ref !== undefined) {
    const [t0] = verdictDe(ref);
    cp.hidden = false;
    cp.innerHTML = `Coupe d'origine : <b>${fr(ref)}</b> (${t0.toLowerCase()}).<br>Après ${ATTAQUES[etat.attaque].court} : <b>${fr(score)}</b> (${txt.toLowerCase()}).` +
      (score <= etat.tau1 ? "<br>Cette falsification passe sous le seuil" + (comp ? "." : " de la branche du bruit seule : essayez le détecteur complet, ou regardez la couche « Niveau du bruit ».") : "");
    av.hidden = false; av.style.left = (ref * 100).toFixed(1) + "%";
  } else if (etat.exemple && etat.attaque === "aucune" && comp && etat.exemple.python) {
    const ref = comp.avecDiffusion ? etat.exemple.python.trois_branches : etat.exemple.python.bruit_cnn;
    cp.hidden = false;
    cp.innerHTML = `Score du même détecteur en Python pour cette coupe : <b>${fr(ref)}</b>` + (comp.avecDiffusion ? " (la diffusion tire un autre bruit aléatoire : petits écarts possibles)." : ".");
    av.hidden = true;
  } else { cp.hidden = true; av.hidden = true; }
}

// ------------------------------------------------------------ progression du détecteur complet
function progression(texte, f) {
  const p = $("#progression");
  if (texte === null) { p.hidden = true; return; }
  p.hidden = false; $("#progression-texte").textContent = texte;
  const b = $("#progression-barre");
  if (f === undefined) b.removeAttribute("value"); else b.value = f;
}

async function lancerComplet(jeton) {
  const M = await hybride();
  const res = etat.res, w = etat.w, h = etat.h;
  const besoin = ["cnn.onnx", etat.avecDiffusion ? "ddpm.onnx" : null].filter(Boolean);
  for (const nom of besoin) {
    if (!(await M.reseauEnCache(nom))) {
      progression(`Téléchargement du réseau ${nom.replace(".onnx", "").toUpperCase()} (${M.TAILLES[nom]} Mo, une seule fois)…`, 0);
      await M.session(nom, (i, n) => progression(`Téléchargement du réseau ${nom.replace(".onnx", "").toUpperCase()} (${M.TAILLES[nom]} Mo, une seule fois)…`, n ? i / n : undefined));
    } else {
      progression("Chargement des réseaux…");
      await M.session(nom);
    }
    if (jeton !== etat.jeton) return null;
  }
  if (!etat.cfgV7) { progression("Chargement des classifieurs…"); etat.cfgV7 = await M.modelesV7(); }
  const comp = await M.analyseComplete(etat.huVu, w, h, res, etat.cfgV7, {
    avecDiffusion: etat.avecDiffusion,
    etape: (b, i, n) => { if (jeton === etat.jeton) progression(b === "cnn" ? `Branche CNN : ${i} / ${n} patchs` : `Branche diffusion : ${i} / ${n} passages`, n ? i / n : 0); },
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

async function calculer() {
  const jeton = ++etat.jeton, voile = $("#calcul"), { w, h } = etat;
  voile.hidden = false;
  const etape = (t) => (voile.firstElementChild.textContent = t + "…");
  etape("Analyse"); await pause();
  etat.complet = null;
  try {
    // 1. coupe d'origine (analysée une fois) et falsification éventuelle
    if (!etat.resOrig) etat.resOrig = analyser(etat.hu, h, w, etat.model);
    if (etat.attaque === "suppression") {
      etat.cible = cibleNodule();
      if (!etat.cible) throw new Error("Pour effacer un nodule, choisissez une coupe « avec nodule » dans la liste.");
    }
    if (etat.attaque !== "aucune" && !etat.cible) etat.cible = cibleParDefaut(etat.resOrig.lung, h, w);
    if (etat.attaque !== "aucune") {
      const r = await falsifier(etat.attaque, etat.hu, h, w, etat.resOrig.lung, etat.cible[0], etat.cible[1], ganRemplir);
      if (jeton !== etat.jeton) return;
      etat.huVu = r.hu; if (r.cible) etat.cible = r.cible;
    } else etat.huVu = etat.hu;
    // 2. branche du bruit (toujours) ; carte du niveau du bruit
    await pause();
    etat.res = etat.attaque === "aucune" ? etat.resOrig : analyser(etat.huVu, h, w, etat.model, etape);
    etat.carteB = etat.res.R ? carteBruit(etat.res.R, etat.res.lung, h, w) : null;
    etat.refOrig = etat.resOrig.ok ? etat.resOrig.score : undefined;
    voile.hidden = true;
    rendre(); afficherVerdict();
    // 3. détecteur complet si demandé (et score d'origine complet pour la comparaison)
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
    } else progression(null);
    $(".couches [data-couche=cnn]").disabled = !etat.complet;
    if (!etat.complet && etat.couche === "cnn") choisirCouche("image");
    journaliser();
  } catch (e) {
    console.error(e);
    voile.hidden = true;
    if (jeton !== etat.jeton) return;
    progression(null);
    $("#aide-film").textContent = e.message;
    if (etat.mode === "complet" && etat.res && etat.res.ok) {
      $("#comparaison").hidden = false;
      $("#comparaison").innerHTML = "Le détecteur complet n'a pas pu tourner : " + e.message + " Le verdict affiché est celui de la branche du bruit.";
    }
  }
}

async function charger(buf, nom, meta = null) {
  let d;
  try { d = parseDicom(buf); } catch (e) { montrerErreur(e.message); return; }
  const hu = d.hu.map((v) => (v < -1024 ? -1024 : v > 3071 ? 3071 : v));   // lecture écrêtée de l'exécution v7
  Object.assign(etat, { hu, w: d.cols, h: d.rows, nom, meta: d, exemple: meta, resOrig: null, cible: null, compOrig: undefined });
  const effacer = document.querySelector("input[name=attaque][value=suppression]"), avecNodule = !!cibleNodule();
  effacer.disabled = !avecNodule;
  effacer.parentElement.title = avecNodule ? "" : "Choisissez une coupe « avec nodule » pour essayer l'effacement";
  if (!avecNodule && etat.attaque === "suppression") {
    etat.attaque = "aucune"; document.querySelector("input[name=attaque][value=aucune]").checked = true;
  }
  const hash = await crypto.subtle.digest("SHA-256", buf).catch(() => null);
  const hex = hash ? Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("") : "";
  etat.sha = hex;
  $("#m-sha").textContent = hex ? hex.slice(0, 12) + "…" + hex.slice(-8) : "indisponible";
  $("#m-sha").title = hex;
  $("#annot-hg").innerHTML = `${(meta && meta.patient) || d.patient || "patient anonyme"}<br>${meta ? "coupe " + meta.coupe : nom}`;
  $("#annot-hd").innerHTML = `${(d.manufacturer || "fabricant inconnu").toLowerCase()}<br>noyau ${d.kernel || "?"}`;
  $("#annot-bg").innerHTML = `${d.cols} × ${d.rows}<br>F 1500 C −600`;
  await calculer();
}

function montrerErreur(msg) {
  $("#verdict-mot").textContent = "Fichier refusé"; $("#verdict-mot").className = "verdict-mot"; $(".verdict-carte").dataset.etat = "";
  $("#aide-film").textContent = msg;
}

function annotBas() {
  const t = { image: "image", residu: "résidu KV 5×5", bruit: "niveau du bruit", proba: "patchs suspects", cnn: "carte du CNN", copie: "copies exactes" }[etat.couche];
  $("#annot-bd").innerHTML = (etat.attaque !== "aucune" ? `falsifiée : ${ATTAQUES[etat.attaque].court}<br>` : "") + t;
}

function choisirCouche(c) {
  document.querySelectorAll(".couches button").forEach((o) => o.setAttribute("aria-selected", String(o.dataset.couche === c)));
  etat.couche = c; rendre();
}

// ------------------------------------------------------------ journal d'audit chaîné (module de sécurité)
const CLE_J = "journal-audit-v1";
const trie = (o) => JSON.stringify(Object.keys(o).sort().reduce((a, k) => ((a[k] = o[k]), a), {}));
async function hacher(t) {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(t));
  return Array.from(new Uint8Array(h), (b) => b.toString(16).padStart(2, "0")).join("");
}
async function journaliser() {
  if (!etat.res || !etat.res.ok || !crypto.subtle) return;
  const j = stock.lire(CLE_J, []);
  const score = etat.complet ? etat.complet.score : etat.res.score;
  const rec = {
    time: new Date().toISOString().slice(0, 19), fichier: etat.nom, empreinte: etat.sha,
    falsification: etat.attaque === "aucune" ? "" : ATTAQUES[etat.attaque].court,
    modele: etat.complet ? (etat.complet.avecDiffusion ? "complet" : "complet sans diffusion") : "branche du bruit",
    score: Math.round(score * 10000) / 10000, verdict: verdictDe(score)[0],
    prev_hash: j.length ? j[j.length - 1].entry_hash : "0".repeat(64),
  };
  rec.entry_hash = await hacher(trie(rec));
  j.push(rec);
  stock.ecrire(CLE_J, j.slice(-200));
  afficherJournal();
}
function afficherJournal() {
  const j = stock.lire(CLE_J, []), tb = $("#j-lignes");
  if (!j.length) { tb.innerHTML = '<tr><td colspan="6" class="vide">Aucune analyse pour l\'instant.</td></tr>'; return; }
  tb.innerHTML = j.slice(-12).reverse().map((r) =>
    `<tr><td>${r.time.slice(11, 19)}</td><td>${r.fichier.replace(".dcm", "")}${r.falsification ? "<small>" + r.falsification + "</small>" : ""}</td><td>${r.modele}</td><td>${fr(r.score)}</td><td>${r.verdict}</td><td class="sha">${r.entry_hash.slice(0, 10)}…</td></tr>`).join("");
}
async function verifierJournal() {
  const j = stock.lire(CLE_J, []);
  let prev = "0".repeat(64);
  for (let i = 0; i < j.length; i++) {
    const { entry_hash, ...rec } = j[i];
    if (rec.prev_hash !== prev || (await hacher(trie(rec))) !== entry_hash) return `Chaîne rompue à l'entrée ${i + 1} : le journal a été modifié.`;
    prev = entry_hash;
  }
  return j.length ? `Chaîne intacte : ${j.length} entrée${j.length > 1 ? "s" : ""} vérifiée${j.length > 1 ? "s" : ""}.` : "Le journal est vide.";
}

// ------------------------------------------------------------ mise en place du laboratoire
async function laboratoire(model, cfg, exemples) {
  Object.assign(etat, { model, cfg });
  const sel = $("#choix-exemple");
  for (const e of exemples) {
    const o = document.createElement("option");
    o.value = e.fichier;
    o.textContent = `${e.patient}, coupe ${e.coupe}${e.nodules.length ? " (avec nodule)" : ""}`;
    sel.append(o);
  }
  const ex = (f) => exemples.find((e) => e.fichier === f);
  const ouvrirExemple = async (f) => {
    const buf = await (await fetch("exemples/" + f)).arrayBuffer();
    await charger(buf, f, ex(f));
  };
  sel.value = "LIDC-IDRI-0085_coupe086.dcm";
  sel.addEventListener("change", () => ouvrirExemple(sel.value));

  const ouvrirFichier = async (file) => {
    if (!file) return;
    const buf = await file.arrayBuffer();
    sel.value = "";
    await charger(buf, file.name);
  };
  $("#fichier").addEventListener("change", (e) => ouvrirFichier(e.target.files[0]));
  const film = $("#film-labo"), depot = $("#depot");
  film.addEventListener("dragover", (e) => { e.preventDefault(); depot.hidden = false; });
  film.addEventListener("dragleave", () => (depot.hidden = true));
  film.addEventListener("drop", (e) => { e.preventDefault(); depot.hidden = true; ouvrirFichier(e.dataTransfer.files[0]); });

  document.querySelectorAll("input[name=attaque]").forEach((r) => r.addEventListener("change", async () => {
    if (!r.checked) return;
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

  // modèle : branche du bruit ou détecteur complet
  etat.mode = stock.lire("mode", "rapide");
  etat.avecDiffusion = stock.lire("diffusion-v2", false);        // la diffusion multiplie la durée par 20 environ
  $("#avec-diffusion").checked = etat.avecDiffusion;
  document.querySelector(`input[name=mode][value=${etat.mode}]`).checked = true;
  document.querySelector(".modele").classList.toggle("complet", etat.mode === "complet");
  document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", async () => {
    etat.mode = r.value; stock.ecrire("mode", r.value);
    document.querySelector(".modele").classList.toggle("complet", etat.mode === "complet");
    await calculer();
  }));
  $("#avec-diffusion").addEventListener("change", async (e) => {
    etat.avecDiffusion = e.target.checked; stock.ecrire("diffusion-v2", etat.avecDiffusion);
    if (etat.mode === "complet") await calculer();
  });
  $("#annuler").addEventListener("click", () => {
    etat.jeton++; progression(null);
    document.querySelector("input[name=mode][value=rapide]").checked = true;
    etat.mode = "rapide"; stock.ecrire("mode", "rapide");
    document.querySelector(".modele").classList.remove("complet");
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
  $("#tau1").addEventListener("input", (e) => { etat.tau1 = Math.min(+e.target.value, etat.tau2 - 0.01); stock.ecrire("tau1", etat.tau1); majSeuils(); afficherVerdict(); });
  $("#tau2").addEventListener("input", (e) => { etat.tau2 = Math.max(+e.target.value, etat.tau1 + 0.01); stock.ecrire("tau2", etat.tau2); majSeuils(); afficherVerdict(); });

  // couches
  const onglets = document.querySelectorAll(".couches button");
  onglets.forEach((b) => b.addEventListener("click", () => choisirCouche(b.dataset.couche)));
  document.querySelector(".couches").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const l = [...onglets].filter((o) => !o.disabled), i = l.findIndex((o) => o.getAttribute("aria-selected") === "true");
    const n = l[(i + (e.key === "ArrowRight" ? 1 : l.length - 1)) % l.length]; n.focus(); n.click();
  });

  // journal
  afficherJournal();
  $("#j-verifier").addEventListener("click", async () => ($("#j-etat").textContent = await verifierJournal()));
  $("#j-exporter").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(stock.lire(CLE_J, []), null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "journal-audit.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  $("#j-effacer").addEventListener("click", () => { stock.ecrire(CLE_J, []); afficherJournal(); $("#j-etat").textContent = "Journal effacé."; });

  await ouvrirExemple(sel.value);
}

// ------------------------------------------------------------ installation (application web progressive)
function installation() {
  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
  let invite = null;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); invite = e; $("#installer").hidden = false; });
  $("#installer").addEventListener("click", async () => {
    if (!invite) return;
    invite.prompt(); await invite.userChoice.catch(() => {}); invite = null; $("#installer").hidden = true;
  });
  window.addEventListener("appinstalled", () => ($("#installer").hidden = true));
}

// ------------------------------------------------------------ démarrage
(async () => {
  installation();
  try {
    const [model, cfg, exemples] = await Promise.all([
      fetch("modele/v7/hgb_forensique.json").then((r) => r.json()),
      fetch("modele/v7/fusion.json").then((r) => r.json()),
      fetch("exemples/index.json").then((r) => r.json()),
    ]);
    await Promise.all([laboratoire(model, cfg, exemples), accueil()]);
  } catch (e) {
    console.error(e);
    const a = $("#aide-film");
    if (a) a.textContent = "Le chargement a échoué (" + e.message + "). Rechargez la page.";
  }
})();

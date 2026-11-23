// Graphiques de l'analyse, en SVG et sans bibliothèque. Chaque fonction rend une chaîne SVG à la largeur donnée
// (le texte garde sa taille réelle) ; les marques portent un attribut data-info lu par l'infobulle.
export const COUL = {
  marine: "#12293a", barre: "#4d6879", gris: "#9aaab4", grille: "#e6ecef", axe: "#c9d4da", texte: "#5a6b75", encre: "#0e1b24",
  sain: "#1f9d72", verifier: "#e39b22", alerte: "#d9472f", froid: "#3f97cf", exces: "#d98a12",
};
export const fr = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",").replace("-", "−") : "—");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const t = (x, y, txt, o = {}) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${o.taille || 11.5}" fill="${o.couleur || COUL.texte}" text-anchor="${o.ancre || "start"}"${o.gras ? ' font-weight="600"' : ""}${o.mono ? ' font-family="Annot, monospace"' : ""}>${esc(txt)}</text>`;
// barre aux extrémités arrondies du côté des données (4 px), ancrée sur la ligne de base
function barreV(x, y, l, h, couleur, info, extra = "") {
  if (h <= 0.5) return "";
  const r = Math.min(4, l / 2, h);
  const d = `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + l - r} Q${x + l},${y} ${x + l},${y + r} V${y + h} Z`;
  return `<path d="${d}" fill="${couleur}"${info ? ` data-info="${esc(info)}"` : ""}${extra}/>`;
}
function barreH(x, y, l, h, couleur, info) {
  if (l <= 0.5) return "";
  const r = Math.min(4, h / 2, l);
  const d = `M${x},${y} H${x + l - r} Q${x + l},${y} ${x + l},${y + r} V${y + h - r} Q${x + l},${y + h} ${x + l - r},${y + h} H${x} Z`;
  return `<path d="${d}" fill="${couleur}"${info ? ` data-info="${esc(info)}"` : ""}/>`;
}
const svg = (l, h, corps, titre) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${l} ${h}" width="${l}" height="${h}" role="img" aria-label="${esc(titre)}" style="max-width:100%;height:auto;font-family:Inter,system-ui,sans-serif">${corps}</svg>`;
const pasJoli = (m) => { const e = 10 ** Math.floor(Math.log10(Math.max(m, 1))); return [1, 2, 5, 10].map((k) => k * e).find((p) => m / p <= 4) || e * 10; };

// ------------------------------------------------------------ histogramme
// valeurs → classes de largeur pas entre min et max ; couleurDe(centre) donne la couleur de chaque classe
export function histogramme({ valeurs, min = 0, max = 1, pas = 0.05, largeur = 480, hauteur = 210, seuils = [], couleurDe = () => COUL.barre,
  graduations = [], formatX = (v) => fr(v, 1), enPourcent = false, unite = "zones", infoDe }) {
  const n = Math.round((max - min) / pas), c = new Array(n).fill(0);
  for (const v of valeurs) { if (!Number.isFinite(v)) continue; const k = Math.min(n - 1, Math.max(0, Math.floor((v - min) / pas))); c[k]++; }
  const tot = valeurs.length || 1, y = enPourcent ? c.map((k) => (100 * k) / tot) : c;
  const g = 40, d = 12, h = 16, b = 30, L = largeur - g - d, H = hauteur - h - b;
  const ymax = Math.max(...y, enPourcent ? 1 : 1), pasY = pasJoli(ymax), haut = Math.ceil(ymax / pasY) * pasY || 1;
  let s = "";
  for (let v = 0; v <= haut + 1e-9; v += pasY) {
    const yy = h + H - (v / haut) * H;
    s += `<line x1="${g}" x2="${g + L}" y1="${yy}" y2="${yy}" stroke="${v === 0 ? COUL.axe : COUL.grille}"/>`;
    s += t(g - 6, yy + 4, enPourcent ? fr(v, v < 10 && pasY < 1 ? 1 : 0) + " %" : String(Math.round(v)), { ancre: "end", taille: 11 });
  }
  const lb = L / n;
  for (let k = 0; k < n; k++) {
    const hh = (y[k] / haut) * H, x = g + k * lb + 1, a = min + k * pas;
    const info = infoDe ? infoDe(a, a + pas, c[k], y[k]) : `${formatX(a)} à ${formatX(a + pas)} : ${c[k]} ${unite}`;
    s += barreV(x, h + H - hh, Math.max(1, lb - 2), hh, couleurDe(a + pas / 2), info);
    s += `<rect x="${g + k * lb}" y="${h}" width="${lb}" height="${H}" fill="transparent" data-info="${esc(info)}"/>`;
  }
  for (const v of graduations) s += t(g + ((v - min) / (max - min)) * L, h + H + 18, formatX(v), { ancre: "middle", taille: 11 });
  for (const { v, nom, couleur } of seuils) {
    const x = g + ((v - min) / (max - min)) * L;
    s += `<line x1="${x}" x2="${x}" y1="${h - 4}" y2="${h + H}" stroke="${couleur}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += t(x + 4, h + 6, nom, { taille: 10.5, couleur });
  }
  return svg(largeur, hauteur, s, "Histogramme");
}

// ------------------------------------------------------------ barres horizontales (0 à 1)
export function barres({ lignes, largeur = 480, seuils = [] }) {
  const lh = 34, h = 12, nom = Math.min(190, largeur * 0.42), d = 46, L = largeur - nom - d - 8, H = lignes.length * lh;
  let s = "";
  lignes.forEach((l, i) => {
    const y = h + i * lh;
    s += t(nom - 10, y + 15, l.nom, { ancre: "end", gras: l.fort, couleur: l.fort ? COUL.encre : COUL.texte, taille: 12.5 });
    if (l.absent) {
      s += `<rect x="${nom}" y="${y + 4}" width="${L}" height="14" rx="4" fill="none" stroke="${COUL.axe}" stroke-dasharray="4 3" data-info="${esc(l.nom + " : non calculé en analyse rapide")}"/>`;
      s += t(nom + 10, y + 15, "non calculé en analyse rapide", { taille: 11.5 });
      return;
    }
    s += `<rect x="${nom}" y="${y + 4}" width="${L}" height="14" rx="4" fill="${COUL.grille}"/>`;
    s += barreH(nom, y + 4, Math.max(0, Math.min(1, l.valeur)) * L, 14, l.couleur || COUL.barre, l.info || `${l.nom} : ${fr(l.valeur, 3)}`);
    s += t(nom + L + 8, y + 16, fr(l.valeur, 3), { gras: l.fort, couleur: COUL.encre, taille: 12.5 });
  });
  for (const { v, couleur } of seuils) {
    const x = nom + v * L;
    s += `<line x1="${x}" x2="${x}" y1="${h - 2}" y2="${h + H - 8}" stroke="${couleur}" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += t(x, h + H + 6, fr(v, 2), { ancre: "middle", taille: 10.5, couleur, mono: true });
  }
  return svg(largeur, h + H + 12, s, "Scores par méthode");
}

// ------------------------------------------------------------ nuage de points
export function nuage({ points, xMin, xMax, yMin = 0, yMax = 1, largeur = 480, hauteur = 230, titreX, titreY, gradX = [], gradY = [0, 0.5, 1],
  formatX = (v) => fr(v, 1), lignesX = [], seuilsY = [] }) {
  const g = 44, d = 14, h = 14, b = 40, L = largeur - g - d, H = hauteur - h - b;
  const X = (v) => g + ((Math.min(xMax, Math.max(xMin, v)) - xMin) / (xMax - xMin)) * L, Y = (v) => h + H - ((v - yMin) / (yMax - yMin)) * H;
  let s = "";
  for (const v of gradY) { s += `<line x1="${g}" x2="${g + L}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === yMin ? COUL.axe : COUL.grille}"/>`; s += t(g - 6, Y(v) + 4, fr(v, 1), { ancre: "end", taille: 11 }); }
  for (const v of gradX) s += t(X(v), h + H + 17, formatX(v), { ancre: "middle", taille: 11 });
  for (const { v, nom } of lignesX) { s += `<line x1="${X(v)}" x2="${X(v)}" y1="${h}" y2="${h + H}" stroke="${COUL.axe}" stroke-dasharray="3 3"/>`; if (nom) s += t(X(v) + 4, h + 10, nom, { taille: 10.5 }); }
  for (const { v, couleur } of seuilsY) s += `<line x1="${g}" x2="${g + L}" y1="${Y(v)}" y2="${Y(v)}" stroke="${couleur}" stroke-width="1.3" stroke-dasharray="4 3"/>`;
  for (const p of [...points].sort((a, b2) => (a.fort ? 1 : 0) - (b2.fort ? 1 : 0))) {
    s += `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${p.fort ? 6 : 4.5}" fill="${p.couleur || COUL.barre}" fill-opacity="${p.fort ? 1 : .55}" stroke="#fff" stroke-width="${p.fort ? 2 : 1}" data-info="${esc(p.info)}"/>`;
  }
  if (titreX) s += t(g + L / 2, hauteur - 4, titreX, { ancre: "middle", taille: 11.5 });
  if (titreY) s += `<text transform="translate(12 ${h + H / 2}) rotate(-90)" font-size="11.5" fill="${COUL.texte}" text-anchor="middle">${esc(titreY)}</text>`;
  return svg(largeur, hauteur, s, "Nuage de points");
}

// ------------------------------------------------------------ score de chaque coupe d'une série
export function serie({ coupes, largeur = 900, hauteur = 220, seuils = [], choisie = -1 }) {
  const g = 40, d = 12, h = 14, b = 30, L = largeur - g - d, H = hauteur - h - b, n = Math.max(1, coupes.length), lb = L / n;
  let s = "";
  for (const v of [0, 0.5, 1]) { const yy = h + H - v * H; s += `<line x1="${g}" x2="${g + L}" y1="${yy}" y2="${yy}" stroke="${v ? COUL.grille : COUL.axe}"/>`; s += t(g - 6, yy + 4, fr(v, 1), { ancre: "end", taille: 11 }); }
  coupes.forEach((c, i) => {
    const x = g + i * lb, hh = Math.max(2, (c.ok ? c.score : 0) * H), ep = Math.max(2, Math.min(lb - 2, 28));
    const xb = x + (lb - ep) / 2;
    if (i === choisie) s += `<rect x="${x + 0.5}" y="${h}" width="${lb - 1}" height="${H}" rx="6" fill="#edf2f5"/>`;
    s += c.ok ? barreV(xb, h + H - hh, ep, hh, c.couleur, null) : `<line x1="${xb}" x2="${xb + ep}" y1="${h + H - 1}" y2="${h + H - 1}" stroke="${COUL.gris}" stroke-width="3"/>`;
    s += `<rect x="${x}" y="${h}" width="${lb}" height="${H + b - 8}" fill="transparent" data-info="${esc(c.info)}" data-coupe="${i}" style="cursor:pointer"/>`;
  });
  for (const { v, couleur, nom } of seuils) {
    const yy = h + H - v * H;
    s += `<line x1="${g}" x2="${g + L}" y1="${yy}" y2="${yy}" stroke="${couleur}" stroke-width="1.3" stroke-dasharray="4 3"/>`;
    s += t(g + L - 2, yy - 4, nom, { ancre: "end", taille: 10.5, couleur });
  }
  const pasE = Math.max(1, Math.ceil(n / Math.floor(L / 34)));
  coupes.forEach((c, i) => { if (i % pasE === 0 || i === choisie) s += t(g + i * lb + lb / 2, h + H + 17, c.etiquette, { ancre: "middle", taille: 10.5, gras: i === choisie, couleur: i === choisie ? COUL.encre : COUL.texte }); });
  return svg(largeur, hauteur, s, "Score de chaque coupe");
}

// ------------------------------------------------------------ infobulle commune
export function activerInfobulles(bulle) {
  const montrer = (e) => {
    const m = e.target.closest && e.target.closest("[data-info]");
    if (!m || !m.closest("svg")) { bulle.hidden = true; return; }
    bulle.textContent = m.getAttribute("data-info");
    bulle.hidden = false;
    const r = bulle.getBoundingClientRect(), x = Math.min(innerWidth - r.width - 8, e.clientX + 14), y = Math.max(8, e.clientY - r.height - 12);
    bulle.style.transform = `translate(${x}px, ${y}px)`;
  };
  document.addEventListener("pointermove", montrer);
  document.addEventListener("pointerdown", montrer);
  document.addEventListener("scroll", () => (bulle.hidden = true), { passive: true });
}

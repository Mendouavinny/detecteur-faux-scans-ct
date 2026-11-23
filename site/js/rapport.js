// Rapport d'analyse : PDF fabriqué dans le navigateur (pages dessinées sur canvas, puis un PDF minimal
// qui contient une image JPEG par page), résumé texte pour WhatsApp ou l'e-mail, version imprimable, export CSV.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function imprimerRapport(r) {
  const zone = document.getElementById("rapport");
  const date = new Date();
  const lignes = (l) => l.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
  zone.innerHTML = `
  <header class="r-tete">
    <div><p class="r-app">Faux scans CT : rapport d'analyse</p><h1>${esc(r.titre)}</h1>${r.sousTitre ? `<p class="r-petit">${esc(r.sousTitre)}</p>` : ""}</div>
    <p class="r-date">${date.toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" })}<br>${date.toLocaleTimeString("fr-FR")}</p>
  </header>
  <section class="r-verdict r-${esc(r.classe)}">
    <div><p class="r-etiquette">Verdict</p><p class="r-mot">${esc(r.verdict)}</p></div>
    <div><p class="r-etiquette">Score de falsification</p><p class="r-score">${esc(r.score)}</p><p class="r-petit">${esc(r.definition)}</p></div>
    <div><p class="r-etiquette">Seuils</p><p class="r-petit">À vérifier au-dessus de ${esc(r.tau1)}<br>Alerte au-dessus de ${esc(r.tau2)}</p></div>
  </section>
  ${r.simulation ? `<p class="r-simulation">Simulation : la coupe a été falsifiée volontairement dans l'application (${esc(r.simulation)}) pour tester le détecteur.</p>` : ""}
  <section class="r-grille">
    <div><h2>Fichier analysé</h2><table class="r-table">${lignes(r.fichier)}</table></div>
    <div><h2>Ce que l'analyse a trouvé</h2><ul class="r-constats">${r.constats.map((c) => `<li>${c}</li>`).join("")}</ul></div>
  </section>
  <section class="r-images">
    <figure><img src="${r.imageCoupe}" alt=""><figcaption>Coupe analysée (fenêtre pulmonaire) ; cadres : les trois zones les plus suspectes.</figcaption></figure>
    <figure><img src="${r.imageCarte}" alt=""><figcaption>Carte de falsification : probabilité de chaque zone (rouge = suspect).</figcaption></figure>
  </section>
  <section class="r-graphes">${r.graphes.map((g) => `<figure><h3>${esc(g.titre)}</h3>${g.svg}</figure>`).join("")}</section>
  ${r.serie ? `<section class="r-serie"><h2>Série : ${r.serie.lignes.length} coupes</h2>${r.serie.svg}
    <table class="r-table r-liste"><thead><tr><th>Ordre</th><th>Fichier</th><th>Score</th><th>Verdict</th></tr></thead><tbody>
    ${r.serie.lignes.map((l) => `<tr><td>${l.ordre}</td><td>${esc(l.nom)}</td><td>${esc(l.score)}</td><td>${esc(l.verdict)}</td></tr>`).join("")}</tbody></table></section>` : ""}
  <footer class="r-pied">
    <p>Analyse faite entièrement sur l'appareil de l'utilisateur, sans envoi d'image. Modèles : exécution finale du mémoire de Master de Vinny Jordan Mendoua (ENSPD, Université de Douala), 300 patients LIDC-IDRI.</p>
    <p><b>Outil de recherche, sans validation clinique.</b> Une alerte invite à vérifier l'image auprès de sa source (PACS, scanner) ; elle ne prouve pas une falsification, et un verdict conforme ne garantit pas l'authenticité.</p>
  </footer>`;
  const titre = document.title;
  document.title = "rapport-" + (r.nomFichier || "analyse").replace(/\.[^.]+$/, "");
  const fin = () => { document.title = titre; window.removeEventListener("afterprint", fin); };
  window.addEventListener("afterprint", fin);
  // laisser aux images le temps d'être décodées avant l'impression
  Promise.all([...zone.querySelectorAll("img")].map((i) => (i.decode ? i.decode().catch(() => {}) : null))).then(() => window.print());
}

export function exporterCSV(lignes, nom = "analyse-serie.csv") {
  const cols = Object.keys(lignes[0] || {});
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const texte = "﻿" + [cols.join(";"), ...lignes.map((l) => cols.map((c) => q(l[c])).join(";"))].join("\r\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([texte], { type: "text/csv;charset=utf-8" }));
  a.download = nom; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ============================================================ rapport PDF dessiné dans le navigateur
// A4 à 150 points par pouce : 1240 × 1754 pixels par page.
const PW = 1240, PH = 1754, MG = 96, LG = PW - 2 * MG, BAS = PH - 128;
const C = {
  encre: "#0e1b24", texte: "#2b3c46", gris: "#5a6b75", clair: "#8a9aa4", trait: "#dbe3e8", fin: "#e9eef1",
  carte: "#f5f8fa", marine: "#12293a", ambre: "#9a5d06",
};
const ETATS = {                                   // [couleur du repère, fond, couleur du texte]
  conforme: ["#1f9d72", "#eaf6f1", "#16795a"],
  verifier: ["#e39b22", "#fdf4e3", "#8f5806"],
  alerte: ["#d9472f", "#fcece8", "#b0331e"],
  non: ["#9aaab4", "#f1f4f6", "#5a6b75"],
};
const FAM = { titre: '"Inter Display", Inter, system-ui, sans-serif', ui: "Inter, system-ui, -apple-system, sans-serif", mono: "Annot, ui-monospace, monospace" };
const police = (taille, poids = 400, fam = FAM.ui) => `${poids} ${taille}px ${fam}`;
const PIED = [
  "Analyse faite entièrement sur l'appareil de l'utilisateur, sans envoi d'image à un serveur. Modèles : exécution finale du mémoire de Master de Vinny Jordan Mendoua (ENSPD, Université de Douala), 300 patients LIDC-IDRI.",
  "<b>Outil de recherche, sans validation clinique.</b> Une alerte invite à vérifier l'image auprès de sa source (PACS, scanner) ; elle ne prouve pas une falsification, et un verdict conforme ne garantit pas l'authenticité.",
];

async function policesPretes() {
  if (!document.fonts || !document.fonts.load) return;
  const f = [police(40, 600, '"Inter Display"'), police(20, 400, "Inter"), police(20, 500, "Inter"), police(20, 600, "Inter"), police(16, 400, "Annot")];
  await Promise.race([Promise.all(f.map((x) => document.fonts.load(x).catch(() => null))), new Promise((ok) => setTimeout(ok, 2500))]);
}
function chargerImage(src) {
  return new Promise((ok, ko) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => ko(new Error("image illisible")); i.src = src; });
}
// Un graphique SVG devient une image à la taille voulue (dessin vectoriel net). Renvoie null si le navigateur
// refuse de le copier dans un canvas (certains anciens Safari) : le graphique est alors omis du PDF.
async function imageSVG(svg, largeur) {
  const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  if (!m) return null;
  const l = +m[1], h = +m[2], k = largeur / l, H = Math.round(h * k);
  const code = svg.replace(/^<svg([^>]*?) width="[\d.]+" height="[\d.]+"/, `<svg$1 width="${largeur}" height="${H}"`).replace(/ style="[^"]*"/, ' style="font-family:Inter,Helvetica,Arial,sans-serif"');
  try {
    const img = await chargerImage("data:image/svg+xml;charset=utf-8," + encodeURIComponent(code));
    const c = document.createElement("canvas"); c.width = largeur; c.height = H;
    const x = c.getContext("2d"); x.fillStyle = "#fff"; x.fillRect(0, 0, largeur, H); x.drawImage(img, 0, 0, largeur, H);
    x.getImageData(0, 0, 1, 1);                    // lève une erreur si le canvas est « contaminé »
    return c;
  } catch { return null; }
}

// ------------------------------------------------------------ texte : découpe en lignes, gras <b>…</b>
const decoder = (s) => s.replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");
function segments(html) {
  const out = [], re = /<b>([\s\S]*?)<\/b>/g;
  let i = 0, m;
  while ((m = re.exec(html))) { if (m.index > i) out.push({ t: decoder(html.slice(i, m.index)), gras: false }); out.push({ t: decoder(m[1]), gras: true }); i = re.lastIndex; }
  if (i < html.length) out.push({ t: decoder(html.slice(i)), gras: false });
  return out;
}
// Lignes de mots positionnés : [[{t, gras, x}]]. Un mot plus large que la colonne (empreinte) est coupé.
function couper(ctx, html, largeur, taille, { poids = 400, gras = 600, fam = FAM.ui } = {}) {
  const mots = [];
  for (const s of segments(String(html ?? ""))) {
    for (const p of s.t.split(/(\s+)/)) {
      if (!p) continue;
      if (/^\s+$/.test(p)) { if (mots.length) mots[mots.length - 1].espace = true; continue; }
      mots.push({ t: p, gras: s.gras, espace: false });
    }
  }
  const mesure = (m, t = m.t) => { ctx.font = police(taille, m.gras ? gras : poids, fam); return ctx.measureText(t).width; };
  ctx.font = police(taille, poids, fam);
  const esp = ctx.measureText(" ").width;
  const lignes = [];
  let cour = [], w = 0, prec = null;
  const placer = (m, mw) => {
    const sep = cour.length && prec && prec.espace ? esp : 0;
    if (cour.length && w + sep + mw > largeur) {
      // un signe collé au mot précédent (« 0,972, ») part à la ligne avec lui
      let j = cour.length;
      if (!sep) { j = cour.length - 1; while (j > 0 && !cour[j - 1].espace) j--; }
      const report = j > 0 ? cour.splice(j) : [];
      lignes.push(cour); cour = []; w = 0;
      for (const r of report) { r.x = w; w += r.w; cour.push(r); }
    }
    const x = cour.length ? w + (prec && prec.espace ? esp : 0) : 0;
    cour.push({ ...m, x, w: mw }); w = x + mw; prec = m;
  };
  for (const m of mots) {
    let mw = mesure(m);
    if (mw <= largeur) { placer(m, mw); continue; }
    let reste = m.t;                                 // mot trop long : coupe caractère par caractère
    while (reste) {
      let k = reste.length;
      const dispo = cour.length ? largeur - w - esp : largeur;
      while (k > 1 && mesure(m, reste.slice(0, k)) > dispo) k--;
      if (k === 1 && cour.length && mesure(m, reste.slice(0, 1)) > dispo) { lignes.push(cour); cour = []; w = 0; prec = null; continue; }
      const bout = { ...m, t: reste.slice(0, k), espace: false };
      mw = mesure(bout); placer(bout, mw); reste = reste.slice(k);
      if (reste) { lignes.push(cour); cour = []; w = 0; prec = null; }
    }
    if (cour.length) cour[cour.length - 1].espace = m.espace;
  }
  if (cour.length) lignes.push(cour);
  return lignes;
}
function ecrire(ctx, lignes, x0, y0, taille, interligne, { couleur = C.texte, couleurGras = C.encre, poids = 400, gras = 600, fam = FAM.ui } = {}) {
  let y = y0;
  for (const l of lignes) {
    for (const m of l) { ctx.font = police(taille, m.gras ? gras : poids, fam); ctx.fillStyle = m.gras ? couleurGras : couleur; ctx.fillText(m.t, x0 + m.x, y); }
    y += interligne;
  }
  return y;
}
function rectArrondi(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// ------------------------------------------------------------ pages
class Pages {
  constructor(entete) { this.liste = []; this.entete = entete; this.ajouter(); }
  ajouter() {
    const c = document.createElement("canvas"); c.width = PW; c.height = PH;
    const x = c.getContext("2d");
    x.fillStyle = "#fff"; x.fillRect(0, 0, PW, PH); x.textBaseline = "alphabetic"; x.imageSmoothingQuality = "high";
    this.liste.push(c); this.x = x; this.y = MG;
    if (this.liste.length > 1) {
      x.font = police(16, 500); x.fillStyle = C.gris;
      const t = this.entete.length > 90 ? this.entete.slice(0, 88) + "…" : this.entete;
      x.fillText(t, MG, MG - 26);
      x.fillStyle = C.trait; x.fillRect(MG, MG - 10, LG, 1.5);
      this.y = MG + 34;
    }
    return x;
  }
  besoin(h) { if (this.y + h > BAS) this.ajouter(); return this.x; }
  titre(t, apres = 22) {
    const x = this.besoin(110);
    x.font = police(27, 600, FAM.titre); x.fillStyle = C.encre; x.fillText(t, MG, this.y + 27);
    this.y += 27 + apres;
  }
  finir() {
    const n = this.liste.length;
    this.liste.forEach((c, k) => {
      const x = c.getContext("2d");
      x.fillStyle = C.trait; x.fillRect(MG, PH - 98, LG, 1.5);
      x.font = police(15, 400); x.fillStyle = C.gris; x.textAlign = "left";
      x.fillText("Faux scans CT, outil de recherche sans valeur de diagnostic", MG, PH - 66);
      x.textAlign = "right"; x.fillText(`Page ${k + 1} sur ${n}`, MG + LG, PH - 66); x.textAlign = "left";
    });
    return this.liste;
  }
}

function dateLongue(d) {
  return d.toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" }) + " à " + d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
}

export async function pagesRapport(r) {
  await policesPretes();
  const date = r.date || new Date();
  const P = new Pages(`Rapport d'analyse : ${r.titre}`);
  let x = P.x;
  const [coul, fond, coulTexte] = ETATS[r.classe] || ETATS.non;

  // ---- en-tête
  x.font = police(20, 600); x.fillStyle = C.marine; x.fillText("Faux scans CT", MG, P.y + 20);
  x.font = police(18, 400); x.fillStyle = C.gris; x.textAlign = "right"; x.fillText(dateLongue(date), MG + LG, P.y + 20); x.textAlign = "left";
  P.y += 84;
  x.font = police(52, 600, FAM.titre); x.fillStyle = C.encre; x.fillText("Rapport d'analyse", MG, P.y);
  P.y += 50;
  P.y = ecrire(x, couper(x, r.titre, LG, 26, { poids: 500 }), MG, P.y, 26, 34, { couleur: C.texte, poids: 500 });
  if (r.sousTitre) P.y = ecrire(x, couper(x, r.sousTitre, LG, 20), MG, P.y - 4, 20, 28, { couleur: C.gris });
  P.y += 18;

  // ---- verdict
  {
    const y0 = P.y, ix = MG + 46, cols = [ix, ix + 300, ix + 300 + 400], wDef = 360;
    const def = couper(x, r.definition, wDef, 17);
    const hTexte = Math.max(132, 120 + def.length * 23, 130);
    const yJauge = y0 + 40 + hTexte + 6, hCarte = yJauge - y0 + 70;
    x.fillStyle = fond; rectArrondi(x, MG, y0, LG, hCarte, 20); x.fill();
    x.save(); rectArrondi(x, MG, y0, LG, hCarte, 20); x.clip(); x.fillStyle = coul; x.fillRect(MG, y0, 12, hCarte); x.restore();
    x.font = police(17, 500); x.fillStyle = C.gris;
    x.fillText("Verdict", cols[0], y0 + 50); x.fillText("Score de falsification", cols[1], y0 + 50); x.fillText("Seuils", cols[2], y0 + 50);
    x.font = police(54, 600, FAM.titre); x.fillStyle = coulTexte; x.fillText(r.verdict, cols[0], y0 + 116);
    x.fillStyle = C.encre; x.fillText(r.score, cols[1], y0 + 116);
    ecrire(x, def, cols[1], y0 + 148, 17, 23, { couleur: C.gris });
    x.font = police(19, 400); x.fillStyle = C.texte;
    x.fillText(`À vérifier au-dessus de ${r.tau1}`, cols[2], y0 + 90);
    x.fillText(`Alerte au-dessus de ${r.tau2}`, cols[2], y0 + 120);
    // jauge 0 → 1 avec les trois plages et le score
    const jx = ix, jw = MG + LG - 40 - ix, jy = yJauge, t1 = +String(r.tau1).replace(",", "."), t2 = +String(r.tau2).replace(",", ".");
    const s = Math.min(1, Math.max(0, Number(r.scoreBrut)));
    x.save(); rectArrondi(x, jx, jy, jw, 12, 6); x.clip();
    [[0, t1, "#bfe3d5"], [t1, t2, "#f6deb2"], [t2, 1, "#f2c3b9"]].forEach(([a, b, c]) => { x.fillStyle = c; x.fillRect(jx + a * jw, jy, (b - a) * jw + 1, 12); });
    x.restore();
    if (Number.isFinite(s)) {
      x.beginPath(); x.arc(jx + s * jw, jy + 6, 13, 0, 2 * Math.PI); x.fillStyle = coul; x.fill();
      x.lineWidth = 4; x.strokeStyle = "#fff"; x.stroke();
    }
    x.font = police(15, 400); x.fillStyle = C.gris;
    x.fillText("0 : authentique", jx, jy + 42); x.textAlign = "right"; x.fillText("1 : falsifiée", jx + jw, jy + 42); x.textAlign = "left";
    P.y = y0 + hCarte + 30;
  }
  if (r.simulation) {
    const l = couper(x, `<b>Simulation.</b> La coupe a été falsifiée volontairement dans l'application (${r.simulation}) pour tester le détecteur.`, LG, 19);
    P.y = ecrire(x, l, MG, P.y + 10, 19, 27, { couleur: C.ambre, couleurGras: C.ambre }) + 14;
  }

  // ---- fichier et constats, sur deux colonnes
  {
    const wG = 452, gap = 60, xD = MG + wG + gap, wD = LG - wG - gap;
    const kW = 176, vW = wG - kW - 20;
    const rangs = r.fichier.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => {
      const mono = /SHA-256/.test(k), lh = mono ? 22 : 25;
      const lk = couper(x, k, kW, 16, { poids: 500 }), lv = couper(x, String(v), vW, mono ? 16 : 19, mono ? { fam: FAM.mono } : {});
      return { mono, lh, lk, lv, h: Math.max(lk.length * 21, lv.length * lh) + 18 };
    });
    const hG = 44 + rangs.reduce((h, q) => h + q.h + 1.5, 0);
    const puces = r.constats.map((c) => couper(x, c, wD - 26, 19));
    const hD = 50 + puces.reduce((h, l) => h + l.length * 28 + 14, 0);
    x = P.besoin(60 + Math.max(hG, hD));
    const y0 = P.y;
    x.font = police(25, 600, FAM.titre); x.fillStyle = C.encre;
    x.fillText("Fichier analysé", MG, y0 + 25); x.fillText("Ce que l'analyse a trouvé", xD, y0 + 25);
    let y = y0 + 44;
    for (const q of rangs) {
      ecrire(x, q.lk, MG, y + 26, 16, 21, { couleur: C.gris, poids: 500 });
      ecrire(x, q.lv, MG + kW + 20, y + 26, q.mono ? 16 : 19, q.lh, { couleur: C.encre, fam: q.mono ? FAM.mono : FAM.ui });
      y += q.h; x.fillStyle = C.fin; x.fillRect(MG, y, wG, 1.5); y += 1.5;
    }
    let yD = y0 + 70;
    for (const l of puces) {
      x.beginPath(); x.arc(xD + 6, yD - 7, 4.5, 0, 2 * Math.PI); x.fillStyle = C.clair; x.fill();
      yD = ecrire(x, l, xD + 26, yD, 19, 28) + 14;
    }
    P.y = Math.max(y, yD) + 20;
  }

  // ---- portée de l'analyse : sur la première page s'il y a la place, sinon à la fin
  const portee = () => {
    const ls = PIED.map((t) => couper(x, t, LG - 52, 17));
    const h = ls.reduce((h, l) => h + l.length * 25 + 10, 0) + 30;
    x = P.besoin(h + 30);
    x.fillStyle = C.carte; rectArrondi(x, MG, P.y, LG, h, 16); x.fill();
    let y = P.y + 36;
    for (const l of ls) y = ecrire(x, l, MG + 26, y, 17, 25, { couleur: C.gris, couleurGras: C.texte }) + 10;
    P.y += h + 34;
  };
  const hPortee = PIED.map((t) => couper(x, t, LG - 52, 17)).reduce((h, l) => h + l.length * 25 + 10, 0) + 60;
  const porteeFaite = P.y + hPortee <= BAS;
  if (porteeFaite) portee();

  // ---- images de la coupe
  {
    const [i1, i2] = await Promise.all([chargerImage(r.imageCoupe), chargerImage(r.imageCarte)]);
    const w = (LG - 44) / 2, h = Math.round(w * (i1.height / i1.width));
    const leg = [
      ["Coupe analysée", "Fenêtre pulmonaire. Cadres : les trois zones les plus suspectes, avec leur score."],
      ["Carte de falsification", "Probabilité de falsification de chaque zone (rouge : suspect)."],
    ].map(([t, l]) => [t, couper(x, l, w, 16)]);
    const hLeg = Math.max(...leg.map(([, l]) => l.length)) * 22;
    x = P.besoin(40 + h + 20 + hLeg);
    [i1, i2].forEach((img, k) => {
      const xx = MG + k * (w + 44);
      x.font = police(19, 600); x.fillStyle = C.encre; x.fillText(leg[k][0], xx, P.y + 19);
      x.save(); rectArrondi(x, xx, P.y + 36, w, h, 14); x.clip(); x.fillStyle = "#05090c"; x.fillRect(xx, P.y + 36, w, h);
      x.drawImage(img, xx, P.y + 36, w, h); x.restore();
      ecrire(x, leg[k][1], xx, P.y + 36 + h + 28, 16, 22, { couleur: C.gris });
    });
    P.y += 36 + h + 28 + hLeg + 24;
  }

  // ---- graphiques, deux par ligne
  const gs = [];
  for (const g of r.graphes) { const c = await imageSVG(g.svg, Math.round((LG - 44) / 2)); if (c) gs.push({ titre: g.titre, c }); }
  if (gs.length) {
    P.titre("Graphiques de la coupe", 26);
    for (let k = 0; k < gs.length; k += 2) {
      const ligne = gs.slice(k, k + 2), h = Math.max(...ligne.map((g) => g.c.height));
      x = P.besoin(34 + h + 30);
      ligne.forEach((g, j) => {
        const xx = MG + j * ((LG + 44) / 2);
        x.font = police(19, 600); x.fillStyle = C.encre; x.fillText(g.titre, xx, P.y + 19);
        x.drawImage(g.c, xx, P.y + 34);
      });
      P.y += 34 + h + 30;
    }
  }

  // ---- série
  if (r.serie) {
    const S = r.serie;
    P.titre(`Série : ${S.lignes.length} coupes`, 20);
    x = P.besoin(40);
    if (S.compte) {
      let xx = MG;
      for (const [cle, txt] of [["conforme", "conforme"], ["verifier", "à vérifier"], ["alerte", "alerte"], ["non", "non analysable"]]) {
        const n = S.compte[cle];
        if (!n && cle === "non") continue;
        const t = `${n} ${txt}${n > 1 && cle !== "verifier" ? "s" : ""}`;
        x.beginPath(); x.arc(xx + 7, P.y + 13, 7, 0, 2 * Math.PI); x.fillStyle = ETATS[cle][0]; x.fill();
        x.font = police(19, 500); x.fillStyle = C.texte; x.fillText(t, xx + 22, P.y + 20);
        xx += 22 + x.measureText(t).width + 34;
      }
      P.y += 44;
    }
    const cg = await imageSVG(S.svg, LG);
    if (cg) { x = P.besoin(cg.height + 20); x.drawImage(cg, MG, P.y); P.y += cg.height + 26; }
    // tableau, répété en tête de chaque page
    const avecNom = S.lignes.some((l) => l.nom);
    const colonnes = avecNom
      ? [["Ordre", 0, 90], ["Coupe", 90, 110], ["Fichier", 200, 520], ["Score", 720, 130], ["Verdict", 850, 198]]
      : [["Ordre", 0, 130], ["Coupe", 130, 200], ["Score", 330, 220], ["Verdict", 550, 300]];
    const tete = () => {
      x = P.x;
      x.font = police(16, 600); x.fillStyle = C.gris;
      for (const [t, dx] of colonnes) x.fillText(t, MG + dx, P.y + 18);
      x.fillStyle = C.trait; x.fillRect(MG, P.y + 30, LG, 1.5);
      P.y += 38;
    };
    x = P.besoin(38 + 36 * 3); tete();
    for (const l of S.lignes) {
      if (P.y + 36 > BAS) { P.ajouter(); tete(); }
      x = P.x;
      const yy = P.y + 24, cle = l.classe || "non";
      x.font = police(17, 400); x.fillStyle = C.texte;
      const val = avecNom ? [String(l.ordre), l.coupe ?? "", l.nom || "", l.score, l.verdict] : [String(l.ordre), l.coupe ?? "", l.score, l.verdict];
      val.forEach((v, j) => {
        const [, dx, w] = colonnes[j];
        let t = String(v);
        if (j === val.length - 1) {
          x.beginPath(); x.arc(MG + dx + 6, yy - 6, 6, 0, 2 * Math.PI); x.fillStyle = ETATS[cle][0]; x.fill();
          x.fillStyle = cle === "non" ? C.gris : C.texte; x.fillText(t, MG + dx + 20, yy); return;
        }
        while (t.length > 3 && x.measureText(t).width > w - 16) t = t.slice(0, -2);
        if (t !== String(v)) t = t.slice(0, -1) + "…";
        x.fillStyle = C.texte; x.fillText(t, MG + dx, yy);
      });
      x.fillStyle = C.fin; x.fillRect(MG, P.y + 35, LG, 1);
      P.y += 36;
    }
    P.y += 24;
  }

  if (!porteeFaite) { P.y += 10; portee(); }
  return P.finir();
}

// ------------------------------------------------------------ PDF minimal : une image JPEG pleine page par page
function texteUTF16(s) {
  let h = "FEFF";
  for (const c of String(s)) {
    const n = c.codePointAt(0);
    if (n > 0xffff) { const v = n - 0x10000; h += (0xd800 + (v >> 10)).toString(16).padStart(4, "0") + (0xdc00 + (v & 0x3ff)).toString(16).padStart(4, "0"); }
    else h += n.toString(16).padStart(4, "0");
  }
  return `<${h.toUpperCase()}>`;
}
export async function versPDF(canvases, { titre = "Rapport d'analyse", auteur = "Faux scans CT" } = {}) {
  const jpg = await Promise.all(canvases.map((c) => new Promise((ok, ko) => c.toBlob((b) => (b ? ok(b.arrayBuffer()) : ko(new Error("conversion JPEG impossible"))), "image/jpeg", 0.86))));
  const enc = new TextEncoder(), morceaux = [], offs = [];
  let pos = 0;
  const pousser = (v) => { const u = typeof v === "string" ? enc.encode(v) : new Uint8Array(v); morceaux.push(u); pos += u.length; };
  const objet = (num, corps) => { offs[num] = pos; pousser(`${num} 0 obj\n`); corps(); pousser("\nendobj\n"); };
  pousser("%PDF-1.4\n%âãÏÓ\n");
  const n = canvases.length, page = (k) => 3 + 3 * k, info = 3 + 3 * n;
  objet(1, () => pousser("<< /Type /Catalog /Pages 2 0 R >>"));
  objet(2, () => pousser(`<< /Type /Pages /Kids [${canvases.map((_, k) => `${page(k)} 0 R`).join(" ")}] /Count ${n} >>`));
  const L = 595.28, H = 841.89, flux = `q ${L} 0 0 ${H} 0 0 cm /Im0 Do Q`;
  canvases.forEach((c, k) => {
    const p = page(k);
    objet(p, () => pousser(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${L} ${H}] /Resources << /XObject << /Im0 ${p + 2} 0 R >> >> /Contents ${p + 1} 0 R >>`));
    objet(p + 1, () => pousser(`<< /Length ${flux.length} >>\nstream\n${flux}\nendstream`));
    objet(p + 2, () => {
      pousser(`<< /Type /XObject /Subtype /Image /Width ${c.width} /Height ${c.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpg[k].byteLength} >>\nstream\n`);
      pousser(jpg[k]); pousser("\nendstream");
    });
  });
  const d = new Date(), z = (v) => String(v).padStart(2, "0");
  const dec = -d.getTimezoneOffset(), fuseau = `${dec >= 0 ? "+" : "-"}${z(Math.floor(Math.abs(dec) / 60))}'${z(Math.abs(dec) % 60)}'`;
  const dPdf = `D:${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}${fuseau}`;
  objet(info, () => pousser(`<< /Title ${texteUTF16(titre)} /Author ${texteUTF16(auteur)} /Producer (Faux scans CT) /CreationDate (${dPdf}) >>`));
  const xref = pos;
  pousser(`xref\n0 ${info + 1}\n0000000000 65535 f \n`);
  for (let i = 1; i <= info; i++) pousser(`${String(offs[i]).padStart(10, "0")} 00000 n \n`);
  pousser(`trailer\n<< /Size ${info + 1} /Root 1 0 R /Info ${info} 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(morceaux, { type: "application/pdf" });
}

export async function pdfRapport(r) {
  const pages = await pagesRapport(r);
  const blob = await versPDF(pages, { titre: `Rapport d'analyse : ${r.titre}` });
  return { blob, pages: pages.length };
}

// ------------------------------------------------------------ résumé pour WhatsApp ou l'e-mail
export function texteResume(r, { whatsapp = false, lien = "" } = {}) {
  const g = (t) => (whatsapp ? `*${t}*` : t);
  const date = r.date || new Date();
  const L = [g("Faux scans CT : rapport d'analyse"), r.sousTitre ? `${r.titre}. ${r.sousTitre}.` : `${r.titre}.`, ""];
  if (r.serie && r.serie.compte) {
    const c = r.serie.compte;
    L.push(`Série : ${c.conforme} conforme${c.conforme > 1 ? "s" : ""}, ${c.verifier} à vérifier, ${c.alerte} alerte${c.alerte > 1 ? "s" : ""}` + (c.non ? `, ${c.non} non analysable${c.non > 1 ? "s" : ""}` : "") + ".");
    if (r.serie.aRevoir && r.serie.aRevoir.length) L.push(`Coupes à revoir : ${r.serie.aRevoir.slice(0, 20).join(", ")}${r.serie.aRevoir.length > 20 ? "…" : ""}.`);
    L.push("");
  }
  L.push(`${r.serie ? "Coupe détaillée" : "Verdict"} : ${g(r.verdict)}, score de falsification ${r.score} (vigilance au-dessus de ${r.tau1}, alerte au-dessus de ${r.tau2}).`);
  L.push(`Analyse ${r.analyse}.`);
  if (r.zone) L.push(`Zone la plus suspecte : ${r.zone}.`);
  if (r.simulation) L.push(`Simulation : falsification ajoutée par l'application (${r.simulation}) pour tester le détecteur.`);
  if (r.sha) L.push(`Empreinte SHA-256 ${r.serie ? "de la coupe détaillée" : "du fichier"} : ${r.sha}`);
  L.push(`Analyse du ${dateLongue(date)}.`, "");
  L.push("Le rapport complet est en pièce jointe (PDF).");
  L.push("Outil de recherche, sans valeur de diagnostic : une alerte invite à vérifier l'image auprès de sa source (PACS, scanner).");
  if (lien) L.push(lien);
  return L.join("\n");
}

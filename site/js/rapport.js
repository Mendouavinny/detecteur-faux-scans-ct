// Rapport d'analyse imprimable (PDF par « Imprimer → Enregistrer en PDF ») et export CSV d'une série.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function imprimerRapport(r) {
  const zone = document.getElementById("rapport");
  const date = new Date();
  const lignes = (l) => l.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
  zone.innerHTML = `
  <header class="r-tete">
    <div><p class="r-app">Faux scans CT · rapport d'analyse</p><h1>${esc(r.titre)}</h1></div>
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

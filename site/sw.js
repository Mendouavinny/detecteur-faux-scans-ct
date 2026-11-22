// Service worker : l'application s'ouvre sans connexion une fois chargée.
// Pages, scripts et styles : réseau d'abord (les mises à jour arrivent), copie locale en secours.
// Polices, images, coupes d'exemple, modèle et ONNX Runtime : copie locale d'abord.
// Les réseaux ONNX sont gardés par l'application elle-même (cache « detecteur-reseaux-v1 »).
const VERSION = "faux-scans-ct-v3";
const BASE = [
  "./", "index.html", "css/style.css", "js/app.js", "js/forensique.js", "js/attaques.js", "js/hybride.js",
  "modele/hgb_forensique.json", "modele/fusion.json", "exemples/index.json", "exemples/LIDC-IDRI-0085_coupe086.dcm",
  "manifest.webmanifest", "img/icone.svg", "img/icone-192.png", "img/logo_ud.png", "img/logo_enspd.png",
  "polices/InterDisplay-SemiBold.woff", "polices/InterDisplay-Medium.woff", "polices/Inter-Regular.woff",
  "polices/Inter-Medium.woff", "polices/Inter-SemiBold.woff", "polices/Lora-Variable.woff", "polices/DejaVuSansMono.woff",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(BASE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((k) => Promise.all(k.filter((n) => n.startsWith("faux-scans-ct-") && n !== VERSION).map((n) => caches.delete(n))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const memeSite = url.origin === location.origin;
  const ort = url.hostname === "cdn.jsdelivr.net" && url.pathname.includes("onnxruntime-web");
  if (!memeSite && !ort) return;                       // dépôt des modèles : géré par l'application
  const dynamique = memeSite && (req.mode === "navigate" || /\.(html|js|css|webmanifest)$/.test(url.pathname) || url.pathname.endsWith("/"));
  if (dynamique) {
    e.respondWith(fetch(req).then((r) => {
      if (r.ok) { const copie = r.clone(); caches.open(VERSION).then((c) => c.put(req, copie)); }
      return r;
    }).catch(() => caches.match(req).then((r) => r || caches.match("index.html"))));
    return;
  }
  e.respondWith(caches.match(req).then((r) => r || fetch(req).then((rep) => {
    if (rep.ok) { const copie = rep.clone(); caches.open(VERSION).then((c) => c.put(req, copie)); }
    return rep;
  })));
});

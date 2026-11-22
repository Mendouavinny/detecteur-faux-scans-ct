"""Transfert des fichiers export_web_v7 depuis une session Kaggle vers l'onglet parent.

À lancer dans le carnet Kaggle (session interactive) avec :
    %run /tmp/d/outils_site/transfert_v7.py

Le script affiche la taille et l'empreinte SHA-256 de chaque fichier, puis une
cellule HTML dont le script lit les fichiers par le serveur Jupyter de la session
et les transmet à la page Kaggle qui l'héberge (window.top.postMessage).
Rien n'est envoyé ailleurs.
"""
import hashlib
import json
import os

from IPython.display import HTML, display

D = "/kaggle/working/export_web_v7"
noms = sorted(n for n in os.listdir(D) if os.path.isfile(os.path.join(D, n)))
info = {}
for n in noms:
    with open(os.path.join(D, n), "rb") as f:
        b = f.read()
    info[n] = [len(b), hashlib.sha256(b).hexdigest()]
print(json.dumps(info, indent=1))

JS = r"""
(async () => {
  const el = document.getElementById('xfer-etat');
  const log = (m) => { el.textContent += '\n' + m; };
  const info = __INFO__;
  let base = null;
  try {
    const c = document.getElementById('jupyter-config-data');
    if (c) base = JSON.parse(c.textContent).baseUrl;
  } catch (e) {}
  if (!base) {
    const e = performance.getEntriesByType('resource').map(x => x.name).find(n => n.includes('/proxy/'));
    if (e) { const u = new URL(e); base = u.pathname.slice(0, u.pathname.indexOf('/proxy/') + 7); }
  }
  log('origine ' + location.origin + ' | base ' + (base ? 'trouvee' : 'absente') + ' | cadre ' + (window.top !== window));
  if (!base) return;
  const b64 = (s) => { const t = atob(s); const u = new Uint8Array(t.length); for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return u.buffer; };
  let ok = 0;
  for (const n of Object.keys(info)) {
    let buf = null;
    for (const p of ['files/kaggle/working/export_web_v7/', 'files/export_web_v7/']) {
      try {
        const r = await fetch(base + p + n, { credentials: 'include' });
        if (r.ok) { buf = await r.arrayBuffer(); if (buf.byteLength === info[n][0]) break; log(n + ' taille ' + buf.byteLength); buf = null; }
        else log(n + ' ' + p + ' -> ' + r.status);
      } catch (e) { log(n + ' ' + p + ' erreur ' + e); }
    }
    if (!buf) {
      for (const p of ['api/contents/kaggle/working/export_web_v7/', 'api/contents/export_web_v7/']) {
        try {
          const r = await fetch(base + p + n + '?content=1&format=base64&type=file', { credentials: 'include' });
          if (r.ok) { buf = b64((await r.json()).content); break; }
          else log(n + ' ' + p + ' -> ' + r.status);
        } catch (e) { log(n + ' ' + p + ' erreur ' + e); }
      }
    }
    if (!buf) continue;
    const taille = buf.byteLength;
    window.top.postMessage({ type: 'fichier-v7', nom: n, buf, attendu: info[n] }, 'https://www.kaggle.com', [buf]);
    ok++;
    log(n + ' transmis (' + taille + ' octets)');
  }
  log('fin ' + ok + '/' + Object.keys(info).length);
})();
"""

display(HTML('<pre id="xfer-etat">transfert</pre><script>' + JS.replace("__INFO__", json.dumps(info)) + "</script>"))

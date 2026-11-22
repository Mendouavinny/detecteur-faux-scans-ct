"""Ré-analyse des scores par coupe de l'exécution Kaggle v4 (rangs extraits du journal)."""
import numpy as np, json
from sklearn.metrics import roc_auc_score
d = np.load('v4_scores_rangs.npz'); meta = json.load(open('v4_scores_meta.json'))
ATK, cols = meta['ATK'], meta['cols']
gi, ai, yi = d['gi'], d['ai'], d['yi']; ge, ye = d['ge'], d['ye']
Ri = {c: d['Ri_' + c] for c in cols}; Re = {c: d['Re_' + c] for c in cols}

def boot(y, a, b, g, n=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; D = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() == y[ii].max(): continue
        D.append(roc_auc_score(y[ii], a[ii]) - (roc_auc_score(y[ii], b[ii]) if b is not None else 0))
    D = np.array(D)
    base = roc_auc_score(y, a) - (roc_auc_score(y, b) if b is not None else 0)
    return round(float(base), 3), round(float(np.percentile(D, 2.5)), 3), round(float(np.percentile(D, 97.5)), 3)

if __name__ != "__main__":
    pass
out = {}
# AUC par famille (famille vs authentiques), effectifs
fam = {}
for k, a in enumerate(ATK[1:], 1):
    m = (ai == 0) | (ai == k)
    n_pos = int((ai == k).sum()); n_pat = int(len(np.unique(gi[ai == k])))
    fam[a] = dict(n_coupes=n_pos, n_patients=n_pat, **{c: round(float(roc_auc_score(yi[m], Ri[c][m])), 3) for c in ('forensique', 'cnn', 'hybride')})
out['interne_par_famille'] = fam
out['interne_n_auth'] = int((ai == 0).sum())
# sous-ensemble comparable à CT-GAN : injection + suppression par GAN vs authentiques
m = np.isin(ai, [0, ATK.index('gan_injection'), ATK.index('gan_suppression')])
out['interne_gan_inj_supp'] = dict(n=int(m.sum()), n_pos=int(yi[m].sum()),
    auc={c: round(float(roc_auc_score(yi[m], Ri[c][m])), 3) for c in cols},
    boot_hybride_moins_forensique=boot(yi[m], Ri['hybride'][m], Ri['forensique'][m], gi[m]))
m = np.isin(ai, [0, ATK.index('gan_injection')])
out['interne_gan_inj'] = dict(n=int(m.sum()), n_pos=int(yi[m].sum()),
    auc={c: round(float(roc_auc_score(yi[m], Ri[c][m])), 3) for c in cols},
    boot_hybride_moins_forensique=boot(yi[m], Ri['hybride'][m], Ri['forensique'][m], gi[m]))
out['interne_global_boot_hybride_moins_forensique'] = boot(yi, Ri['hybride'], Ri['forensique'], gi)
out['externe'] = dict(n=int(len(ye)), n_pos=int(ye.sum()), n_scans=int(len(np.unique(ge))),
    auc={c: round(float(roc_auc_score(ye, Re[c])), 3) for c in cols},
    boot_forensique_moins_hybride=boot(ye, Re['forensique'], Re['hybride'], ge))
# AUC seules avec IC
out['ic_auc_interne'] = {c: boot(yi, Ri[c], None, gi) for c in cols}
out['ic_auc_externe'] = {c: boot(ye, Re[c], None, ge) for c in cols}
json.dump(out, open('analyse_v4.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

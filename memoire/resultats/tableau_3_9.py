"""Tableau 3.9 : classement fusion / forensique sur des sous-ensembles comparables (recalculé, scores v4 arrondis)."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
meta = json.load(open('v4_scores_meta.json')); GE, GI, ATK, cols = meta['GE'], meta['GI'], meta['ATK'], meta['cols']
d = np.load('v4_scores_rangs.npz')
gi = np.array([GI[i] for i in d['gi']]); ai, yi = d['ai'], d['yi']; Ri = {c: d['Ri_' + c] for c in cols}
ge = np.array([GE[i] for i in d['ge']]); ye = d['ye']; Re = {c: d['Re_' + c] for c in cols}
T = np.load('v4_externe_types.npy')
an = json.load(open('analyse_ctgan.json'))
def boot(y, a, b, g, n=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; D = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].min() == y[ii].max(): continue
        D.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 3), round(float(np.percentile(D, 2.5)), 3), round(float(np.percentile(D, 97.5)), 3)]
rows = []
def row(name, y, s_for, s_hyb, g, unit):
    rows.append(dict(test=name, pos=int(y.sum()), neg=int((1 - y).sum()), groupes=int(len(set(g))), unite=unit,
                     auc_for=round(float(roc_auc_score(y, s_for)), 3), auc_fus=round(float(roc_auc_score(y, s_hyb)), 3),
                     ecart=boot(y, s_hyb, s_for, g)))
m = np.ones(len(yi), bool); row("Interne, toutes familles", yi[m], Ri['forensique'][m], Ri['hybride'][m], gi[m], "patients")
m = np.isin(ai, [0, ATK.index('gan_injection')]); row("Interne, injection par GAN", yi[m], Ri['forensique'][m], Ri['hybride'][m], gi[m], "patients")
auth = np.isin(T, ['m', 'b'])
m = np.ones(len(ye), bool); row("CT-GAN, injections et suppressions", ye[m], Re['forensique'][m], Re['hybride'][m], ge[m], "scanners")
m = auth | (T == 'I'); row("CT-GAN, injections (FM)", ye[m], Re['forensique'][m], Re['hybride'][m], ge[m], "scanners")
m = auth | (T == 'S'); row("CT-GAN, suppressions (FB)", ye[m], Re['forensique'][m], Re['hybride'][m], ge[m], "scanners")
excl = set(an["candidat_entrainement_ou_validation"])
m = ~np.isin(ge, list(excl)); row("CT-GAN sans les scanners candidats d'entraînement ou de validation", ye[m], Re['forensique'][m], Re['hybride'][m], ge[m], "scanners")
json.dump(rows, open('tableau_3_9.json', 'w'), indent=1, ensure_ascii=False)
for r in rows: print(r)

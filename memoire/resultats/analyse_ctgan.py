"""Test externe CT-GAN : séparation injections / suppressions et analyse de sensibilité à la provenance."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from analyse_v4 import boot
meta = json.load(open('v4_scores_meta.json')); GE, cols = meta['GE'], meta['cols']
d = np.load('v4_scores_rangs.npz'); ge, ye = d['ge'], d['ye']; Re = {c: d['Re_' + c] for c in cols}
gi, ai, yi = d['gi'], d['ai'], d['yi']; Ri = {c: d['Ri_' + c] for c in cols}
T = np.load('v4_externe_types.npy'); scan = np.array([GE[i] for i in ge])
P = json.load(open('provenance_ctgan_resume.json'))
test_ids = {"0019","0030","0062","0064","0065","0067","0078","0080","0102","0114","0116","0119","0140","0144","0145","0166","0190","0275","0330","0344","0357","0367","0368","0377","0411","0426","0447","0453","0463","0468","0533","0543","0550","0575","0603","0625","0641","0642","0677","0684","0784","0796","0928","0940","0967"}
cand = {}
for tok in P['candidats_dans_300'].split():
    u, n, lst = tok.split(':'); cand[u] = (int(n), lst.split(','))
uniq_trainval = {u for u, (n, l) in cand.items() if n == 1 and l[0] not in test_ids}
any_trainval = {u for u, (n, l) in cand.items() if any(x not in test_ids for x in l)}
uniq_test = {u for u, (n, l) in cand.items() if n == 1 and l[0] in test_ids}
out = {"repartition_types": {k: int((T == k).sum()) for k in "ISmb"},
       "scanners_externes": int(len(set(scan))),
       "empreinte_unique_entrainement_ou_validation": sorted(uniq_trainval & set(scan)),
       "empreinte_unique_test": sorted(uniq_test & set(scan)),
       "candidat_entrainement_ou_validation": sorted(any_trainval & set(scan))}
def block(mask, name):
    y = ye[mask]; g = scan[mask]
    r = dict(n=int(mask.sum()), n_pos=int(y.sum()), scanners=int(len(set(g))),
             auc={c: round(float(roc_auc_score(y, Re[c][mask])), 3) for c in ('forensique', 'cnn', 'hybride', 'sans_diffusion')},
             forensique_moins_fusion=boot(y, Re['forensique'][mask], Re['hybride'][mask], g))
    out[name] = r
auth = np.isin(T, ['m', 'b'])
block(np.ones(len(T), bool), "externe_tout")
block(auth | (T == 'I'), "externe_injections_FM")
block(auth | (T == 'S'), "externe_suppressions_FB")
block(~np.isin(scan, list(uniq_trainval)), "externe_sans_empreinte_unique_entrainement_validation")
block(~np.isin(scan, list(any_trainval)), "externe_sans_aucun_candidat_entrainement_validation")
m = ~np.isin(scan, list(any_trainval))
block(m & (auth | (T == 'I')), "externe_injections_sans_candidat")
json.dump(out, open('analyse_ctgan.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

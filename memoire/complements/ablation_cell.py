# --- Ablation : apport de la branche diffusion (mêmes patients, même protocole) ---
# Exécutée après le carnet complet : réutilise BO / EXB (sorties des branches), vkeys (validation), tkeys (test).
import json as _json
import numpy as _np
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
from sklearn.metrics import roc_auc_score as _auc

try:
    _LX2 = _np.vstack([_np.c_[BO[k]["p_for"], BO[k]["p_cnn"]] for k in vkeys])
    _loc2 = _LR(max_iter=2000, class_weight="balanced").fit(_LX2, L_y)

    def _fused2(bo):
        return _loc2.predict_proba(_np.c_[bo["p_for"], bo["p_cnn"]])[:, 1]

    def _meta2_feat(bo):
        return _np.r_[bo["gfeat"], agg(_fused2(bo)), agg(bo["p_for"]), agg(bo["p_cnn"])]

    _meta2 = _HGB(max_iter=300, learning_rate=0.05, random_state=0)
    _meta2.fit(_np.nan_to_num(_np.array([_meta2_feat(BO[k]) for k in vkeys])),
               _np.array([int(DATA[k]["attack"] != "authentique") for k in vkeys]))

    def _score2(bo):
        return float(_meta2.predict_proba(_np.nan_to_num(_meta2_feat(bo))[None])[0, 1])

    def _boot(y, a, b, g, n=2000, seed=0):
        rng = _np.random.default_rng(seed)
        ug = _np.unique(g)
        idx = {u: _np.where(g == u)[0] for u in ug}
        D = []
        for _ in range(n):
            ii = _np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            if y[ii].min() == y[ii].max():
                continue
            D.append(_auc(y[ii], a[ii]) - _auc(y[ii], b[ii]))
        D = _np.array(D)
        return dict(diff=float(_auc(y, a) - _auc(y, b)), lo=float(_np.percentile(D, 2.5)),
                    hi=float(_np.percentile(D, 97.5)), p_le0=float((D <= 0).mean()), n_boot=int(len(D)),
                    n_groupes=int(len(ug)))

    ABL = {}
    # test interne
    _g = _np.array([str(k[0]) for k in tkeys])
    _s = {b: _np.asarray(SC[b], float) for b in SC}
    _s["sans_diffusion"] = _np.array([_score2(BO[k]) for k in tkeys])
    ABL["interne"] = {b: dict(auc=float(_auc(yt, v)), tpr1=float(tpr_at(yt, v))) for b, v in _s.items()}
    ABL["interne"]["n"] = int(len(yt)); ABL["interne"]["patients"] = int(len(set(_g)))
    ABL["interne"]["falsifiees"] = int(yt.sum())
    ABL["boot_interne_hybride_vs_sans_diffusion"] = _boot(yt, _s["hybride"], _s["sans_diffusion"], _g)
    ABL["boot_interne_hybride_vs_forensique"] = _boot(yt, _s["hybride"], _s["forensique"], _g)
    ABL["boot_interne_sans_diffusion_vs_forensique"] = _boot(yt, _s["sans_diffusion"], _s["forensique"], _g)
    _pred2 = (_s["sans_diffusion"] >= 0.5).astype(int)
    from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
    ABL["interne"]["sans_diffusion"].update(acc=float(_acc(yt, _pred2)), f1=float(_f1(yt, _pred2)))
    ABL["interne"]["hybride"].update(acc=float(_acc(yt, (_s["hybride"] >= 0.5).astype(int))),
                                     f1=float(_f1(yt, (_s["hybride"] >= 0.5).astype(int))))
    _atk = _np.array([DATA[k]["attack"] for k in tkeys])
    ABL["auc_par_attaque"] = {}
    for a in sorted(set(_atk) - {"authentique"}):
        m = (_atk == a) | (_atk == "authentique")
        ABL["auc_par_attaque"][a] = {b: float(_auc(yt[m], _s[b][m])) for b in ("hybride", "sans_diffusion")}
    SCORES = dict(interne=dict(pid=_g.tolist(), y=yt.tolist(), attack=_atk.tolist(),
                               **{b: _np.round(v, 5).tolist() for b, v in _s.items()}))
    # test externe CT-GAN
    try:
        _ek = list(EXB.keys())
        _ey = _np.array([ex_y[k] for k in _ek])
        _eg = _np.array([str(k[1]) for k in _ek])
        _es = {b: _np.array([slice_scores(EXB[k])[b] for k in _ek], float) for b in ("forensique", "cnn", "diffusion", "hybride")}
        _es["sans_diffusion"] = _np.array([_score2(EXB[k]) for k in _ek])
        ABL["externe"] = {b: dict(auc=float(_auc(_ey, v)), tpr1=float(tpr_at(_ey, v))) for b, v in _es.items()}
        ABL["externe"]["n"] = int(len(_ey)); ABL["externe"]["scanners"] = int(len(set(_eg)))
        ABL["boot_externe_hybride_vs_sans_diffusion"] = _boot(_ey, _es["hybride"], _es["sans_diffusion"], _eg)
        ABL["boot_externe_forensique_vs_hybride"] = _boot(_ey, _es["forensique"], _es["hybride"], _eg)
        SCORES["externe"] = dict(scan=_eg.tolist(), y=_ey.tolist(), **{b: _np.round(v, 5).tolist() for b, v in _es.items()})
    except Exception as _e:
        ABL["externe_erreur"] = repr(_e)
    _json.dump(ABL, open(OUT / "ablation_diffusion.json", "w"), indent=1)
    _json.dump(SCORES, open(OUT / "scores_par_coupe.json", "w"))
    print("ABLATION_JSON_DEBUT")
    print(_json.dumps(ABL))
    print("ABLATION_JSON_FIN")
    for _part in ("interne", "externe"):
        if _part in SCORES:
            print("SCORES_" + _part.upper() + "_DEBUT")
            print(_json.dumps(SCORES[_part]))
            print("SCORES_" + _part.upper() + "_FIN")
except Exception as _e:
    import traceback
    traceback.print_exc()
    print("ABLATION_ERREUR", repr(_e))

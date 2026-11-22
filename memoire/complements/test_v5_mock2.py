exec(open("test_v5_mock.py").read().replace('exec(compile(src, "v5", "exec"))', 'V5_SEEDS=[]\nexec(compile(src, "v5", "exec"))'))
print("----- tests directs E/F")
H = np.random.default_rng(1).random((31, 31)).astype(np.float32); H[5:10, 5:10] = 0.9
f = z_glcm(H); print("glcm", f.shape, np.isfinite(f).all())
cnn_probs = lambda model, img, coords: np.random.default_rng(0).random(len(coords))
class FC:
    def predict_proba(self, X): p = 1/(1+np.exp(-X[:, 0])); return np.c_[1-p, p]
EXF = {k: _cpu_job(k) for k in _ek}
R = eval_partition({k[0]: v["split"] for k, v in DATA.items()}, FC(), None, 42)
print(json.dumps(R)[:1500])

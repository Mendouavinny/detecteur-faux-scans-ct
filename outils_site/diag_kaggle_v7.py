"""Diagnostic de l'export v7 (Kaggle) : écarts ONNX / PyTorch du CNN, différences de features.py."""
import difflib, glob, json, sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
OUTD = Path("/kaggle/working/export_web_v7")
SRC = Path(sorted(glob.glob("/kaggle/input/**/out/modeles_sklearn.joblib", recursive=True))[0]).parent
DEPOT = Path("/kaggle/working/depot")
r = json.loads((OUTD / "reference.json").read_text())
print("écart logit CNN ONNX/PyTorch sur les coupes :", {k[10:23]: round(v["ecart_cnn_onnx_torch"], 5) for k, v in r.items()})
a = (SRC.parent / "features.py").read_text().splitlines(); b = (DEPOT / "features.py").read_text().splitlines()
print("\n".join(difflib.unified_diff(a, b, "features_v7.py", "features_depot.py", lineterm="", n=1)))
sys.path.insert(0, str(SRC.parent)); import features as FZ  # noqa
sys.path.insert(1, str(DEPOT)); import reseaux_torch as RT  # noqa
cnn = RT.ForensicUNet(); cnn.load_state_dict(torch.load(SRC / "cnn_best.pt", map_location="cpu")); cnn.eval()
s = ort.InferenceSession(str(OUTD / "cnn.onnx"), providers=["CPUExecutionProvider"])
for nom, x in [("aléatoire N(0,1)", torch.randn(11, 1, 64, 64)), ("échelle HU/1000", torch.randn(11, 1, 64, 64) * 0.3 - 0.7)]:
    with torch.no_grad():
        lo, se = cnn(x)
    lo2, se2 = s.run(None, {"x": x.numpy()})
    print(f"{nom}: logit max {lo.abs().max():.2f} écart {np.abs(lo.numpy() - lo2).max():.2e} ; seg max {se.abs().max():.2f} écart {np.abs(se.numpy() - se2).max():.2e} ; "
          f"écart des probabilités {np.abs(1 / (1 + np.exp(-lo.numpy())) - 1 / (1 + np.exp(-lo2))).max():.2e}")

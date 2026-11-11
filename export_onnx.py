"""Exporte les trois réseaux PyTorch (poids .pt de la 4e exécution) au format ONNX et vérifie l'égalité des sorties.

Usage : python export_onnx.py [dossier_modeles] [dossier_sortie]
Les fichiers cnn.onnx, ddpm.onnx et gan.onnx permettent d'exécuter l'application sans PyTorch (onnxruntime).
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import reseaux_torch as RT  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI / "modeles"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC
DST.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


def charger(cls, nom):
    m = cls(); m.load_state_dict(torch.load(SRC / nom, map_location="cpu")); return m.eval()


class DDPMEps(torch.nn.Module):     # t entier (N,) → bruit prédit
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, t):
        return self.m(x, t)


RES = {
    "cnn.onnx": (charger(RT.ForensicUNet, "cnn_best.pt"), (torch.randn(8, 1, 64, 64),), ["x"], ["logit", "seg"]),
    "ddpm.onnx": (DDPMEps(charger(RT.DiffUNet, "ddpm_attaquant.pt")),
                  (torch.randn(8, 1, 64, 64), torch.full((8,), 150, dtype=torch.long)), ["x", "t"], ["eps"]),
    "gan.onnx": (charger(RT.UNetG, "gan_G.pt"), (torch.randn(1, 3, 64, 64),), ["x"], ["y"]),
}
for nom, (m, ex, ins, outs) in RES.items():
    axes = {k: {0: "n"} for k in ins + outs}
    torch.onnx.export(m, ex, str(DST / nom), input_names=ins, output_names=outs, dynamic_axes=axes,
                      opset_version=17, dynamo=False)
    # contrôle : même entrée (autre taille de lot) → mêmes sorties
    ex2 = tuple(torch.cat([e, e[:3]]) for e in ex)
    with torch.no_grad():
        ref = m(*ex2)
    ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    s = ort.InferenceSession(str(DST / nom), providers=["CPUExecutionProvider"])
    got = s.run(None, {k: e.numpy() for k, e in zip(ins, ex2)})
    err = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    amp = max(float(np.abs(a).max()) for a in ref)
    print(f"{nom}: {(DST / nom).stat().st_size / 1e6:.1f} Mo, écart max ONNX/PyTorch = {err:.2e} "
          f"(amplitude des sorties {amp:.1f}, écart relatif {err / amp:.1e})")
    assert err / amp < 1e-3, nom   # arrondis flottants ; la fidélité de bout en bout est contrôlée par comparer_moteurs.py

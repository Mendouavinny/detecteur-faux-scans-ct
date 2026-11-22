# --- M3Dsynth : extraction ciblée de coupes falsifiées (exécution CPU, lecture en flux des trois archives) ---
# Les archives (153 à 183 Go) ne peuvent pas être téléchargées : elles sont lues en flux et seules les coupes choisies sont écrites.
import os, sys, json, time, subprocess
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc"], check=False)
import configparser
configparser.SafeConfigParser = getattr(configparser, "SafeConfigParser", configparser.ConfigParser)
import numpy as np, pandas as pd
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
(Path.home() / ".pylidcrc").write_text("[dicom]\npath = /tmp/lidc\nwarn = True\n")
import pylidc as pl
T0 = time.time(); W = Path("/kaggle/working"); OUTD = W / "m3d"; OUTD.mkdir(parents=True, exist_ok=True)
N_INJ, N_REM = 80, 60
GH = "https://raw.githubusercontent.com/grip-unina/M3Dsynth/main/data/"
data = pd.read_csv(GH + "data.csv").merge(pd.read_csv(GH + "sets.csv"))
scans = pl.query(pl.Scan).all(); patients = sorted({s.patient_id for s in scans})
rng = np.random.default_rng(42); sel = sorted(rng.choice(patients, size=300, replace=False).tolist())
have = {s.patient_id for s in scans}
sel = [p for p in sel if p in have]
rng = np.random.default_rng(42); perm = rng.permutation(sel); n = len(perm); ntr, nva = int(0.7 * n), int(0.15 * n)
EXCL = set(perm[:ntr + nva])                     # patients d'entraînement et de validation de la partition principale
te = data[(data["set"] == "test") & ~data.orig_id.isin(EXCL)]
full = te[te["mod"] != "real"].groupby("img_id")["mod"].nunique()
pick = []
for ty, nmax in (("inj", N_INJ), ("rem", N_REM)):
    ids = sorted(i for i in full[full == 3].index if i.startswith(ty + "_"))
    if len(ids) < nmax:
        ids = sorted(i for i in full.index if i.startswith(ty + "_"))
    pick += list(np.random.default_rng(0).choice(ids, min(nmax, len(ids)), replace=False))
SEL = te[te.img_id.isin(pick)].copy()
SEL.to_csv(W / "m3d_selection.csv", index=False)
print("SELECTION", SEL.groupby(["mod", "ty"]).size().to_dict(), "patients", SEL.orig_id.nunique(), "exclus", len(EXCL), flush=True)

EXTRACT = r'''
import sys, tarfile, time
from pathlib import Path
mod, lst, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
want = set(Path(lst).read_text().split()); got = 0; t0 = time.time(); seen = 0
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tf:
    for m in tf:
        seen += 1
        if m.isfile() and m.name in want:
            p = out / m.name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tf.extractfile(m).read()); got += 1
            if got == len(want):
                break
        if seen % 200000 == 0:
            print(f"{mod}: {seen} membres lus, {got}/{len(want)} coupes, {(time.time() - t0) / 60:.0f} min", flush=True)
print(f"{mod}: FIN {got}/{len(want)} coupes, {seen} membres, {(time.time() - t0) / 60:.1f} min", flush=True)
'''
(W / "extract_stream.py").write_text(EXTRACT)
procs = {}
for mod in ("pix2pix", "cycle", "diffusion"):
    sub = SEL[SEL["mod"] == mod]
    names = [f"{mod}/scan/{r.img_id}/slide{int(r.coord_z):04d}.tiff" for r in sub.itertuples()]
    names += [f"{mod}/label/{r.img_id}/slide{int(r.coord_z):04d}.tiff" for r in sub.itertuples()]
    (W / f"liste_{mod}.txt").write_text("\n".join(names) + "\n")
    cmd = (f'curl -s --retry 5 "https://www.grip.unina.it/download/prog/M3Dsynth/{mod}.tgz" | '
           f'{sys.executable} {W}/extract_stream.py {mod} {W}/liste_{mod}.txt {OUTD} > {W}/journal_{mod}.txt 2>&1')
    procs[mod] = subprocess.Popen(cmd, shell=True)
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for mod in procs:
        j = (W / f"journal_{mod}.txt").read_text().strip().splitlines()
        print(f"[{(time.time() - T0) / 60:5.0f} min] {mod} : {j[-1] if j else '...'}", flush=True)
RES = {}
for mod in procs:
    j = (W / f"journal_{mod}.txt").read_text().strip().splitlines()
    RES[mod] = dict(code=procs[mod].returncode, fin=j[-1] if j else None,
                    coupes=len(list((OUTD / mod / "scan").glob("*/*.tiff"))) if (OUTD / mod / "scan").exists() else 0)
print("M3DX_DEBUT"); print(json.dumps(RES)); print("M3DX_FIN")

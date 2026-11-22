# --- Identité des scanners CT-GAN et sondes M3Dsynth / TruFor (exécution CPU) ---
import os, sys, json, glob, zipfile, io, re, time, subprocess, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc", "pydicom"], check=False)
import configparser
configparser.SafeConfigParser = getattr(configparser, "SafeConfigParser", configparser.ConfigParser)
import numpy as np, pandas as pd, pydicom
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
(Path.home() / ".pylidcrc").write_text("[dicom]\npath = /tmp/lidc\nwarn = True\n")
import pylidc as pl
T0 = time.time(); R = {}
def ch(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

scans = pl.query(pl.Scan).all()
patients = sorted({s.patient_id for s in scans})
rng = np.random.default_rng(42); sel = sorted(rng.choice(patients, size=300, replace=False).tolist())
scan_of = {}
for s in scans:
    if s.patient_id in sel and s.patient_id not in scan_of:
        scan_of[s.patient_id] = s
sel = [p for p in sel if p in scan_of]
rng = np.random.default_rng(42); perm = rng.permutation(sel); n = len(perm); ntr = int(0.7 * n); nva = int(0.15 * n)
SPLIT = {p: "train" for p in perm[:ntr]}; SPLIT.update({p: "val" for p in perm[ntr:ntr + nva]}); SPLIT.update({p: "test" for p in perm[ntr + nva:]})
R["split"] = pd.Series(SPLIT).value_counts().to_dict()
try:
    R["zvals_hors_dicom"] = int(len(np.asarray(scans[0].slice_zvals, float)))
except Exception as e:
    R["zvals_hors_dicom"] = repr(e)[:100]

def hu(ds):
    return np.round(ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))).astype(np.int32)

# --- 1. CT-GAN : positions de toutes les coupes, pixels des coupes étiquetées et de la coupe médiane ---
root = Path("/tmp/ctgan"); root.mkdir(exist_ok=True); zp = root / "u.zip"
os.system(f'wget -q -O "{zp}" "https://archive.ics.uci.edu/static/public/520/deepfakes+medical+image+tamper+detection.zip"')
with zipfile.ZipFile(zp) as z:
    z.extractall(root)
zp.unlink()
inner = sorted(root.glob("**/*.zip"))
for zf in inner:
    with zipfile.ZipFile(zf) as z:
        for nme in z.namelist():
            if nme.lower().endswith(".csv"):
                z.extract(nme, zf.parent)
want = {}
for lf in sorted(glob.glob(str(root / "**" / "labels_exp*.csv"), recursive=True)):
    lab = pd.read_csv(lf); lab.columns = [c.strip().lower() for c in lab.columns]
    for _, r in lab.iterrows():
        want.setdefault(str(r["uuid"]), set()).add(int(r["slice"]))
CTG = {}
for zf in inner:
    with zipfile.ZipFile(zf) as z:
        by = {}
        for nme in z.namelist():
            p = Path(nme).parts
            if not nme.endswith("/") and len(p) >= 2 and re.fullmatch(r"\d+", p[-2]):
                by.setdefault(p[-2], []).append(nme)
        for u, files in by.items():
            hd = []
            for f in files:
                try:
                    ds = pydicom.dcmread(io.BytesIO(z.read(f)), stop_before_pixels=True)
                    hd.append((float(ds.ImagePositionPatient[2]), f, [round(float(v), 2) for v in ds.ImagePositionPatient[:2]]))
                except Exception:
                    pass
            hd.sort()
            if not hd:
                continue
            idx = sorted({len(hd) // 2} | {i for i in want.get(u, ()) if 0 <= i < len(hd)})
            px = {i: hu(pydicom.dcmread(io.BytesIO(z.read(hd[i][1])))) for i in idx}
            CTG[u] = dict(n=len(hd), zt=tuple(round(h[0], 2) for h in hd), org=hd[0][2], px=px, zip=zf.name)
ch(f"CT-GAN lu : {len(CTG)} scanners")
R["ctgan_scanners"] = len(CTG)

# --- 2. Les 300 patients LIDC : positions de toutes les coupes, comparaison aux scanners CT-GAN ---
NBIA = "https://services.cancerimagingarchive.net/nbia-api"; _TOK = {}
def tok():
    if "v" not in _TOK:
        d = urllib.parse.urlencode({"username": "nbia_guest", "password": "", "client_id": "NBIA", "grant_type": "password"}).encode()
        with urllib.request.urlopen(urllib.request.Request(NBIA + "/oauth/token", data=d), timeout=60) as r:
            _TOK["v"] = json.load(r)["access_token"]
    return _TOK["v"]
def fetch(pid):
    uid, last = scan_of[pid].series_instance_uid, None
    for a in range(4):
        for url, auth in ((NBIA + "/services/v1/getImage?SeriesInstanceUID=" + uid, False), (NBIA + "/services/v2/getImage?SeriesInstanceUID=" + uid, True)):
            try:
                rq = urllib.request.Request(url, headers={"User-Agent": "lidc-tampering-research"})
                if auth:
                    rq.add_header("Authorization", "Bearer " + tok())
                with urllib.request.urlopen(rq, timeout=300) as r:
                    return r.read()
            except Exception as e:
                last = e
        time.sleep(5 * (a + 1))
    raise RuntimeError(str(last)[:80])
byn = {}
for u, c in CTG.items():
    byn.setdefault(c["n"], []).append(u)
LID, CMP, FAIL = {}, [], []
pool = ThreadPoolExecutor(6); futs = {}; done = set()
for i, pid in enumerate(sel):
    for nx in sel[i:i + 7]:
        if nx not in futs and nx not in done:
            futs[nx] = pool.submit(fetch, nx)
    done.add(pid)
    try:
        data = futs.pop(pid).result()
    except Exception as e:
        FAIL.append(pid); continue
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        hd = []
        for m in z.namelist():
            if m.lower().endswith(".dcm"):
                ds = pydicom.dcmread(io.BytesIO(z.read(m)), stop_before_pixels=True)
                hd.append((float(ds.ImagePositionPatient[2]), m, [round(float(v), 2) for v in ds.ImagePositionPatient[:2]]))
        hd.sort()
        zt = tuple(round(h[0], 2) for h in hd)
        LID[pid] = dict(n=len(hd), z0=zt[0] if zt else None, z1=zt[-1] if zt else None, org=hd[0][2] if hd else None)
        for u in byn.get(len(hd), []):
            c = CTG[u]; res = dict(uuid=u, pid=pid, split=SPLIT[pid], zmatch=c["zt"] == zt, orgmatch=c["org"] == LID[pid]["org"])
            eq = {}
            for k, a in c["px"].items():
                b = hu(pydicom.dcmread(io.BytesIO(z.read(hd[k][1]))))
                if a.shape == b.shape:
                    eq[k] = round(float((a == b).mean()), 4)
            res["eq"] = eq
            if res["zmatch"] or (eq and max(eq.values()) > 0.2):
                CMP.append(res)
    del data
    if i % 25 == 0:
        ch(f"{i}/{len(sel)} patients LIDC, {len(CMP)} correspondances")
pool.shutdown()
R["lidc_lus"] = len(LID); R["lidc_echecs"] = FAIL
best = {}
for r in CMP:
    s = (r["zmatch"], max(r["eq"].values()) if r["eq"] else 0)
    if r["uuid"] not in best or s > best[r["uuid"]][0]:
        best[r["uuid"]] = (s, r)
IDENT = {u: v[1] for u, v in best.items()}
R["ctgan_identifies"] = len(IDENT)
R["ctgan_identifies_par_partition"] = pd.Series([v["split"] for v in IDENT.values()]).value_counts().to_dict() if IDENT else {}
ch("comparaison terminée")

# --- 3. Sondes : archives M3Dsynth, poids TruFor, recouvrement M3Dsynth / nos partitions ---
PR = {}
for nm, url in (("cycle", "https://www.grip.unina.it/download/prog/M3Dsynth/cycle.tgz"), ("pix2pix", "https://www.grip.unina.it/download/prog/M3Dsynth/pix2pix.tgz"),
                ("diffusion", "https://www.grip.unina.it/download/prog/M3Dsynth/diffusion.tgz"), ("trufor", "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip")):
    o = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    PR[nm] = [l.strip() for l in o.splitlines() if l.upper().startswith("HTTP/") or l.lower().startswith("content-length")]
t = time.time()
subprocess.run(["curl", "-s", "--max-time", "120", "-r", "0-199999999", "-o", "/tmp/part.tgz", "https://www.grip.unina.it/download/prog/M3Dsynth/pix2pix.tgz"])
sz = os.path.getsize("/tmp/part.tgz") if os.path.exists("/tmp/part.tgz") else 0
PR["debit_Mo_s"] = round(sz / 1e6 / max(time.time() - t, 1e-3), 1); PR["octets_partiels"] = sz
PR["membres"] = subprocess.run("tar -tzf /tmp/part.tgz 2>/dev/null | head -12", shell=True, capture_output=True, text=True).stdout.split("\n")
try:
    st = pd.read_csv("https://raw.githubusercontent.com/grip-unina/M3Dsynth/main/data/sets.csv")
    st["nous"] = st.orig_id.map(SPLIT).fillna("hors_300")
    PR["recouvrement"] = st.groupby(["set", "nous"]).size().unstack(fill_value=0).to_dict()
except Exception as e:
    PR["recouvrement"] = repr(e)[:100]
R["sondes"] = PR
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(dict(R=R, IDENT=IDENT, CMP=CMP, LID=LID), open("/kaggle/working/identite_ctgan.json", "w"), default=str)
print("RESUME_DEBUT"); print(json.dumps(R, ensure_ascii=False, default=str)); print("RESUME_FIN")
print("IDENT_DEBUT"); print(json.dumps({u: [v["pid"], v["split"], int(v["zmatch"]), int(v["orgmatch"]), max(v["eq"].values()) if v["eq"] else None] for u, v in IDENT.items()})); print("IDENT_FIN")
ch("fin")

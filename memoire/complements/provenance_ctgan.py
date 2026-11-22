# --- Provenance des scanners CT-GAN et type de falsification par coupe (exécution CPU) ---
# 1) types (FM/FB/TM/TB) des 147 coupes du test externe, dans l'ordre exact du carnet principal ;
# 2) en-têtes DICOM des scanners CT-GAN comparés aux métadonnées LIDC-IDRI (pylidc) et aux 300 patients retenus.
import os, sys, json, glob, zipfile, subprocess, io, re
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylidc", "pydicom"], check=False)
import configparser
configparser.SafeConfigParser = getattr(configparser, "SafeConfigParser", configparser.ConfigParser)
import numpy as np, pandas as pd, pydicom
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
(Path.home() / ".pylidcrc").write_text("[dicom]\npath = /tmp\nwarn = True\n")
import pylidc as pl

R = {}
# --- LIDC : métadonnées et sélection des 300 patients (même code et même graine que le carnet principal) ---
scans = pl.query(pl.Scan).all()
patients = sorted({s.patient_id for s in scans})
rng = np.random.default_rng(42)
sel = sorted(rng.choice(patients, size=300, replace=False).tolist())
scan_of = {}
for s in scans:
    if s.patient_id in sel and s.patient_id not in scan_of:
        scan_of[s.patient_id] = s
sel = [p for p in sel if p in scan_of]
rng = np.random.default_rng(42)
perm = rng.permutation(sel); n = len(perm); n_tr = int(0.70 * n); n_va = int(0.15 * n)
SPLIT = {p: "train" for p in perm[:n_tr]}
SPLIT.update({p: "val" for p in perm[n_tr:n_tr + n_va]})
SPLIT.update({p: "test" for p in perm[n_tr + n_va:]})
lidc = []
for s in scans:
    nf = len([f for f in (s.sorted_dicom_file_names or "").split(",") if f.strip()])
    lidc.append(dict(pid=s.patient_id, study=s.study_instance_uid, series=s.series_instance_uid,
                     ps=round(float(s.pixel_spacing), 4), st=round(float(s.slice_thickness), 3), nf=nf,
                     retenu=s.patient_id in scan_of and scan_of[s.patient_id].id == s.id))
L = pd.DataFrame(lidc)
R["lidc_scans"] = len(L); R["lidc_patients"] = int(L.pid.nunique()); R["retenus"] = len(sel)
R["test_patients_v4"] = sorted([p for p, v in SPLIT.items() if v == "test"])

# --- CT-GAN : téléchargement de l'archive UCI (id 520) ---
root = Path("/tmp/ctgan"); root.mkdir(exist_ok=True)
zp = root / "uci520.zip"
os.system(f'wget -q -O "{zp}" "https://archive.ics.uci.edu/static/public/520/deepfakes+medical+image+tamper+detection.zip"')
R["zip_octets"] = zp.stat().st_size if zp.exists() else 0
with zipfile.ZipFile(zp) as z:
    z.extractall(root)
zp.unlink()
inner = sorted(root.glob("**/*.zip"))
R["zips_internes"] = [p.name for p in inner]
# labels : extraits ; en-têtes DICOM : lus dans les zips internes sans tout extraire
heads = {}
for zf in inner:
    with zipfile.ZipFile(zf) as z:
        names = z.namelist()
        for nme in names:
            if nme.lower().endswith(".csv"):
                z.extract(nme, zf.parent)
        by_folder = {}
        for nme in names:
            if nme.endswith("/"):
                continue
            parts = Path(nme).parts
            if len(parts) >= 2:
                by_folder.setdefault(parts[-2], []).append(nme)
        for uuid, files in by_folder.items():
            if not re.fullmatch(r"\d+", uuid):
                continue
            info = None
            for f in sorted(files)[:5]:
                try:
                    ds = pydicom.dcmread(io.BytesIO(z.read(f)), stop_before_pixels=True)
                except Exception:
                    continue
                if info is None:
                    info = dict(zip=zf.name, PatientID=str(getattr(ds, "PatientID", "")),
                                Study=str(getattr(ds, "StudyInstanceUID", "")), Series=str(getattr(ds, "SeriesInstanceUID", "")),
                                ps=round(float(ds.PixelSpacing[0]), 4) if hasattr(ds, "PixelSpacing") else None,
                                st=round(float(getattr(ds, "SliceThickness", 0) or 0), 3),
                                Manufacturer=str(getattr(ds, "Manufacturer", "")), Date=str(getattr(ds, "StudyDate", "")))
                break
            if info is not None:
                info["n"] = len(files)
                heads.setdefault(uuid, []).append(info)
R["scanners_ctgan_lus"] = len(heads)

# --- Ordre exact des coupes du test externe (mêmes règles que le carnet principal) ---
labs = sorted(set(glob.glob(str(root / "**" / "labels_exp*.csv"), recursive=True)))
R["fichiers_labels"] = [Path(l).name for l in labs]
rows = []
for lf in labs:
    lab = pd.read_csv(lf); lab.columns = [c.strip().lower() for c in lab.columns]
    base = Path(lf).parent
    for uuid, grp in lab.groupby("uuid"):
        folder = next(iter(glob.glob(str(base / "**" / str(uuid)), recursive=True)), None)
        h = heads.get(str(uuid))
        nsl = h[0]["n"] if h else None
        if folder is None and h is None:
            continue
        for j, r in grp.reset_index().iterrows():
            zz = int(r["slice"])
            if nsl is not None and not 0 <= zz < nsl:
                continue
            rows.append(dict(lf=Path(lf).name, uuid=str(uuid), slice=zz, type=str(r["type"]).strip().upper()))
R["n_coupes_externes"] = len(rows)
R["types"] = pd.Series([r["type"] for r in rows]).value_counts().to_dict()
R["sequence_types"] = "".join({"FM": "I", "FB": "S", "TM": "m", "TB": "b"}.get(r["type"], "?") for r in rows)
R["sequence_uuids"] = [r["uuid"] for r in rows]

# --- Appariement des scanners CT-GAN avec LIDC-IDRI ---
series_set = dict(zip(L.series, L.pid)); study_set = dict(zip(L.study, L.pid))
match = {}
for uuid, hs in heads.items():
    h = hs[0]
    m = dict(uid=series_set.get(h["Series"]) or study_set.get(h["Study"]),
             pid_tag=h["PatientID"] if h["PatientID"].startswith("LIDC") else None)
    fp = L[(L.ps == h["ps"]) & (L.st == h["st"]) & (L.nf == h["n"])]
    m["empreinte"] = sorted(fp.pid.unique().tolist())[:6]
    m["n_empreinte"] = int(fp.pid.nunique())
    cands = {c for c in [m["uid"], m["pid_tag"]] if c}
    m["dans_300"] = sorted(c for c in cands if c in SPLIT)
    m["partition"] = sorted({SPLIT[c] for c in cands if c in SPLIT})
    m["empreinte_dans_300"] = sorted(p for p in m["empreinte"] if p in SPLIT)
    match[uuid] = m
R["uid_retrouves"] = sum(1 for m in match.values() if m["uid"])
R["patientid_lidc"] = sum(1 for m in match.values() if m["pid_tag"])
R["scanners_dans_300_par_uid_ou_id"] = sorted(u for u, m in match.items() if m["dans_300"])
R["scanners_empreinte_unique"] = sum(1 for m in match.values() if m["n_empreinte"] == 1)
R["scanners_empreinte_unique_dans_300"] = sorted(u for u, m in match.items() if m["n_empreinte"] == 1 and m["empreinte_dans_300"])
R["exemples_entetes"] = {u: hs[0] for u, hs in list(heads.items())[:3]}
R["prefixes_uid"] = pd.Series([hs[0]["Series"][:28] for hs in heads.values()]).value_counts().head(5).to_dict()
R["patientid_exemples"] = pd.Series([hs[0]["PatientID"] for hs in heads.values()]).head(8).tolist()
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(dict(R=R, match=match, rows=rows), open("/kaggle/working/provenance_ctgan.json", "w"), indent=1)
print("PROVENANCE_DEBUT")
print(json.dumps(R, ensure_ascii=False))
print("PROVENANCE_FIN")
print("MATCH_DEBUT")
print(json.dumps(match, ensure_ascii=False))
print("MATCH_FIN")

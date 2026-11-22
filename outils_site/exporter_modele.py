"""Exporte le classifieur forensique (modeles/modeles_fusion.joblib) pour l'application web (site/modele/)
et calcule les scores de référence Python des coupes d'exemple, comparés ensuite par verifier_parite.mjs.

    python outils_site/exporter_modele.py
    node outils_site/verifier_parite.mjs
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import joblib

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import detecteur as D  # noqa: E402

import __main__  # noqa: E402
__main__.PositiveLogit = D.PositiveLogit


def lire_dicom(chemin):
    """Lecture minimale (DICOM non compressé, VR implicite ou explicite), indépendante de pydicom."""
    b = Path(chemin).read_bytes()

    def val(g, e):
        cle = struct.pack('<HH', g, e)
        i = b.rfind(cle) if (g, e) == (0x7fe0, 0x10) else b.find(cle, 132)
        vr = b[i + 4:i + 6]
        if vr.isalpha() and vr.isupper():
            if vr in (b'OB', b'OW', b'SQ', b'UN', b'UT', b'OF'):
                n = struct.unpack_from('<I', b, i + 8)[0]
                return b[i + 12:i + 12 + n]
            n = struct.unpack_from('<H', b, i + 6)[0]
            return b[i + 8:i + 8 + n]
        n = struct.unpack_from('<I', b, i + 4)[0]
        return b[i + 8:i + 8 + n]

    rows, cols = struct.unpack('<H', val(0x28, 0x10))[0], struct.unpack('<H', val(0x28, 0x11))[0]
    signe = struct.unpack('<H', val(0x28, 0x103))[0]
    pente = float(val(0x28, 0x1053).decode().strip() or 1)
    origine = float(val(0x28, 0x1052).decode().strip() or 0)
    px = np.frombuffer(val(0x7fe0, 0x10), dtype='<i2' if signe else '<u2').reshape(rows, cols)
    return (px.astype(np.float32) * pente + origine).astype(np.int16)


def main():
    clf = joblib.load(RACINE / 'modeles' / 'modeles_fusion.joblib')['forensic_clf']
    arbres = []
    for it in clf._predictors:
        nd = it[0].nodes
        assert not nd['is_categorical'].any()
        arbres.append([nd['feature_idx'].tolist(), [float(x) for x in nd['num_threshold']], nd['left'].tolist(),
                       nd['right'].tolist(), nd['is_leaf'].tolist(), [float(x) for x in nd['value']],
                       nd['missing_go_to_left'].tolist()])
    sortie = RACINE / 'site' / 'modele' / 'hgb_forensique.json'
    sortie.write_text(json.dumps(dict(baseline=float(clf._baseline_prediction.ravel()[0]), n_features=27, trees=arbres),
                                 separators=(',', ':')))
    ref = {}
    for f in sorted((RACINE / 'site' / 'exemples').glob('*.dcm')):
        img = lire_dicom(f)
        coords, feats = D.slice_patch_features(img, D.lung_mask_slice(img))
        p = clf.predict_proba(feats)[:, 1]
        ref[f.name] = dict(n=len(coords), p=p.tolist(), score=float(np.mean(np.sort(p)[::-1][:3])))
    (RACINE / 'outils_site' / 'reference_python.json').write_text(json.dumps(ref))
    print(f'{len(arbres)} arbres exportés vers {sortie.relative_to(RACINE)} ; {len(ref)} coupes de référence')


if __name__ == '__main__':
    main()

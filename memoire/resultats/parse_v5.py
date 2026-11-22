"""Extrait les blocs JSON (V5_<TAG>_DEBUT ... V5_<TAG>_FIN, ABLATION_JSON, SCORES_*) d'un journal Kaggle copié en texte."""
import json, re, sys
from pathlib import Path


def blocks(text):
    out = {}
    lines = [re.sub(r"^\s*\d+(\.\d+)?s\s+\d+\s+", "", l) for l in text.splitlines()]   # retire « 123.4s  56 » du visualiseur
    i = 0
    while i < len(lines):
        m = re.match(r"^(V5_[A-Z0-9_]+|ABLATION_JSON|SCORES_INTERNE|SCORES_EXTERNE|RESUME|IDENT)_DEBUT$", lines[i].strip())
        if m:
            tag = m.group(1); buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != f"{tag}_FIN":
                buf.append(lines[i].strip()); i += 1
            try:
                out[tag] = json.loads("".join(buf).replace("NaN", "null"))
            except Exception as e:
                out[tag] = {"_erreur": repr(e), "_brut": "".join(buf)[:500]}
        i += 1
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    B = blocks(Path(src).read_text(encoding="utf-8"))
    json.dump(B, open(dst, "w"), ensure_ascii=False, indent=1)
    print({k: (list(v)[:6] if isinstance(v, dict) else type(v).__name__) for k, v in B.items()})

import json, ast, re, sys
sys.path.insert(0, '.')
import cells_a, cells_b, cells_c
CELLS = cells_a.CELLS
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
bad = 0
for i, (t, src) in enumerate(CELLS):
    if t == "code":
        lines = src.split("\n")
        if lines[0].startswith("%%writefile"):
            body = "\n".join(lines[1:])
        else:
            body = "\n".join(l for l in lines if not l.lstrip().startswith(("!", "%")))
        try:
            ast.parse(body)
        except SyntaxError as e:
            bad += 1; print("ERREUR cellule", i, e)
    cell = {"cell_type": t, "metadata": {}, "source": [l + "\n" for l in src.split("\n")]}
    cell["source"][-1] = cell["source"][-1].rstrip("\n")
    cell["id"] = f"c{i:03d}"
    if t == "code":
        cell["outputs"] = []; cell["execution_count"] = None
    nb["cells"].append(cell)
json.dump(nb, open("detection_faux_scans_LIDC.ipynb", "w"), ensure_ascii=False, indent=1)
print(len(CELLS), "cellules, erreurs de syntaxe :", bad)

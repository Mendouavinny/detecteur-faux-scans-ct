"""Génère le jeu de données de la preuve de concept (SimCT-Tamper)."""
import numpy as np
from multiprocessing import Pool
import simct as S

N_PAT, N_AUTH, N_PER_ATT = 36, 15, 3
N_GEN_PAT = 6
KERNELS = ["ramp", "shepp-logan", "cosine", "hann"]
CLASSES = ["authentique", "injection", "suppression", "copier-deplacer", "bruit", "generatif"]


def scanner(pid):
    r = np.random.default_rng(1000 + pid)
    return float(r.uniform(2e4, 1.2e5)), KERNELS[pid % len(KERNELS)]


def acquire_one(args):
    pid, j, with_nod = args
    rng = np.random.default_rng(pid * 1000 + j)
    I0, ker = scanner(pid)
    hu, lungs, nods = S.make_phantom(rng, with_nodule=with_nod)
    img = S.acquire(hu, rng, I0, ker)
    return pid, j, img, lungs, nods


def main():
    # 1) Patients « attaquant » pour apprendre le générateur PPCA
    jobs = [(100 + p, j, True) for p in range(N_GEN_PAT) for j in range(12)]
    # 2) Patients de l'étude
    plan = []
    for p in range(N_PAT):
        labels = [0] * N_AUTH + sum([[c] * N_PER_ATT for c in range(1, 6)], [])
        for j, c in enumerate(labels):
            rng = np.random.default_rng(p * 77 + j)
            with_nod = True if c == 2 else (rng.random() < 0.5)
            plan.append((p, j, c, with_nod))
    jobs += [(p, j, wn) for p, j, c, wn in plan]
    with Pool(2) as pool:
        res = pool.map(acquire_one, jobs, chunksize=8)
    gen_res = [r for r in res if r[0] >= 100]
    stu = {(r[0], r[1]): r for r in res if r[0] < 100}

    # Générateur
    patches = []
    for _, _, img, lungs, nods in gen_res:
        for cy, cx, r in nods:
            h = 14
            pt = img[cy - h:cy + h, cx - h:cx + h].astype(np.float32)
            if pt.shape == (28, 28):
                patches.append(pt)
                patches.append(pt[:, ::-1]); patches.append(pt[::-1]); patches.append(pt.T)
    gen = S.PPCAGenerator(k=24).fit(np.array(patches))
    print("patchs PPCA:", len(patches))

    X, M, Y, P, L = [], [], [], [], []
    for p, j, c, _ in plan:
        _, _, img, lungs, nods = stu[(p, j)]
        rng = np.random.default_rng(10_000 + p * 100 + j)
        if c == 0:
            out, m = img, np.zeros(img.shape, bool)
        elif c == 1:
            out, m = S.tamper_injection(img, lungs, rng)
        elif c == 2:
            out, m = S.tamper_removal(img, lungs, nods, rng)
        elif c == 3:
            out, m = S.tamper_copymove(img, lungs, rng)
        elif c == 4:
            out, m = S.tamper_noise(img, rng)
        else:
            out, m = S.tamper_generative(img, lungs, gen, rng)
        X.append(out); M.append(m); Y.append(c); P.append(p); L.append(lungs)
    np.savez_compressed("simct_tamper.npz", X=np.array(X), M=np.packbits(np.array(M), axis=-1),
                        Y=np.array(Y), P=np.array(P), L=np.packbits(np.array(L), axis=-1))
    print("ok", len(X), np.bincount(Y))


if __name__ == "__main__":
    main()

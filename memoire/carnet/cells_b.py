from cells_a import md, code, CELLS

md(r"""
### 3 bis. Cache des coupes candidates et banques de patchs (un seul passage sur les DICOM)

Lecture de chaque patient une seule fois : coupes candidates (HU int16), masques pulmonaires, nodules présents dans la coupe.
Les scanners sont téléchargés par lots de 8 en parallèle pendant que le lot précédent est lu, puis supprimés du disque.
Les patchs servant à entraîner les **outils de l'attaquant** (GAN, DDPM, classifieur cible de FGSM) proviennent **uniquement des patients d'entraînement**.
""")
code(r"""
from concurrent.futures import ThreadPoolExecutor
P = CFG["PATCH"]
CACHE = {}          # (pid, z) -> dict(img, lung, nods)
NOD_PATCHES, LUNG_PATCHES = [], []
rng = np.random.default_rng(CFG["SEED"] + 1)
t0 = time.time(); FAILED = []
_dl = ThreadPoolExecutor(8)
_fut = {pid: _dl.submit(ensure_dicom, pid) for pid in patients[:8]}
for i, pid in enumerate(patients):
    for nxt in patients[i + 1:i + 9]:                      # pré-téléchargement des 8 suivants
        if nxt not in _fut: _fut[nxt] = _dl.submit(ensure_dicom, nxt)
    try:
        _fut.pop(pid).result()
        vol, nods, ps, st = load_patient(pid)
    except Exception as e:
        print("échec", pid, e); FAILED.append(pid); drop_dicom(pid); continue
    drop_dicom(pid)
    for z in candidate_slices(vol, nods, CFG["SLICES_PER_PATIENT"], rng):
        img = vol[:, :, z].copy()
        lung = lung_mask_slice(img)
        if lung.sum() < 4000:
            continue
        nz = [(r, c, rad) for (r, c, zz, rad) in nods if abs(zz - z) <= 1 and 40 < r < 472 and 40 < c < 472]
        CACHE[(pid, z)] = dict(img=img, lung=lung, nods=nz)
        if SPLIT[pid] == "train":
            for (r, c, rad) in nz:
                NOD_PATCHES.append(img[r - P // 2:r + P // 2, c - P // 2:c + P // 2])
            er_idx = np.argwhere(ndi.binary_erosion(lung, iterations=P // 2))
            for _ in range(6 if len(er_idx) else 0):
                y, x = er_idx[rng.integers(len(er_idx))]
                if all(abs(y - r) > P and abs(x - c) > P for (r, c, _) in nz):
                    pt = img[y - P // 2:y + P // 2, x - P // 2:x + P // 2]
                    if pt.shape == (P, P):
                        LUNG_PATCHES.append(pt)
    if i % 20 == 0:
        print(f"{i}/{len(patients)} patients, {len(CACHE)} coupes, {time.time() - t0:.0f}s", flush=True)
_dl.shutdown()
patients = [p for p in patients if p not in FAILED]
NOD_PATCHES = np.array([p for p in NOD_PATCHES if p.shape == (P, P)], np.int16)
LUNG_PATCHES = np.array(LUNG_PATCHES, np.int16)
# une coupe réelle avec nodule, conservée pour les figures du §12
try:
    np.save("real_slice.npy", CACHE[next(k for k, c in CACHE.items() if c["nods"])]["img"])
except StopIteration:
    pass
print("patients lus :", len(patients), "| échecs :", len(FAILED))
print("coupes en cache :", len(CACHE), "| patchs nodule :", NOD_PATCHES.shape, "| patchs poumon :", LUNG_PATCHES.shape)
chrono("cache des coupes construit")
""")

md(r"""
### 3 ter. Classifieur de nodules (cible de l'attaque FGSM)
""")
code(r"""
def augment_d4(x):
    k = np.random.randint(4); x = np.rot90(x, k, axes=(-2, -1))
    return np.flip(x, -1) if np.random.rand() < 0.5 else x

Xn = np.concatenate([NOD_PATCHES, LUNG_PATCHES[:len(NOD_PATCHES) * 3]]).astype(np.float32)
yn = np.concatenate([np.ones(len(NOD_PATCHES)), np.zeros(min(len(LUNG_PATCHES), len(NOD_PATCHES) * 3))]).astype(np.float32)
nodcls = NoduleClassifier().to(DEVICE)
opt = torch.optim.Adam(nodcls.parameters(), 1e-4)
for ep in range(CFG["NODCLS_EPOCHS"]):
    perm = np.random.permutation(len(Xn)); tot = 0
    for b in range(0, len(perm), 64):
        idx = perm[b:b + 64]
        xb = torch.tensor(hu_to_net(np.stack([augment_d4(Xn[j]).copy() for j in idx]))[:, None], device=DEVICE)
        loss = F.binary_cross_entropy_with_logits(nodcls(xb), torch.tensor(yn[idx], device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    print(f"époque {ep}: perte {tot / len(perm):.4f}")
""")

md(r"""
## 4. GAN d'inpainting conditionnel (pix2pix) — injection et suppression de nodules

Entrée du générateur : patch $64\times64$ dont le carré central $32\times32$ est masqué, le masque, et une carte de condition (1 = générer un nodule, 0 = générer du parenchyme sain). Perte : $\mathcal{L}_{cGAN} + 100\,\lVert y - G(x_m)\rVert_1$ (éq. 2.5).
""")
code(r"""
WIN_LO, WIN_HI = -1100.0, 300.0
def to_gan(x):   return (np.clip((x - WIN_LO) / (WIN_HI - WIN_LO), 0, 1) * 2 - 1).astype(np.float32)
def from_gan(y): return (y + 1) / 2 * (WIN_HI - WIN_LO) + WIN_LO

CENTER = np.zeros((P, P), np.float32); CENTER[P // 4:3 * P // 4, P // 4:3 * P // 4] = 1

def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))

class UNetG(nn.Module):
    def __init__(self, cin=3, b=48):
        super().__init__()
        self.e1, self.e2, self.e3, self.bt = cbr(cin, b), cbr(b, 2 * b), cbr(2 * b, 4 * b), cbr(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), cbr(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), cbr(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), cbr(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); bt = self.bt(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(bt), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return torch.tanh(self.out(d1))

class PatchD(nn.Module):
    def __init__(self, cin=4, b=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(b, 2 * b, 4, 2, 1), nn.BatchNorm2d(2 * b), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(2 * b, 4 * b, 4, 2, 1), nn.BatchNorm2d(4 * b), nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(4 * b, 1, 3, 1, 1))
    def forward(self, x): return self.net(x)

def gan_inputs(y, cond):
    # y : (B,1,P,P) dans [-1,1] ; cond : (B,) 0/1
    m = torch.tensor(CENTER, device=y.device)[None, None].expand_as(y)
    c = cond.view(-1, 1, 1, 1).float().expand_as(y)
    return torch.cat([y * (1 - m), m, c], 1), m

G, D = UNetG().to(DEVICE), PatchD().to(DEVICE)
oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
Xg = np.concatenate([NOD_PATCHES, LUNG_PATCHES]); cg = np.concatenate([np.ones(len(NOD_PATCHES)), np.zeros(len(LUNG_PATCHES))])
w = np.where(cg == 1, len(cg) / (2 * cg.sum()), len(cg) / (2 * (1 - cg).sum())); w /= w.sum()   # rééquilibrage
bce = nn.BCEWithLogitsLoss()
steps = max(1, len(Xg) // 64)
for ep in range(CFG["GAN_EPOCHS"]):
    for _ in range(steps):
        idx = np.random.choice(len(Xg), 64, p=w)
        y = torch.tensor(to_gan(np.stack([augment_d4(Xg[j]).copy() for j in idx]))[:, None], device=DEVICE)
        cond = torch.tensor(cg[idx], device=DEVICE)
        x, m = gan_inputs(y, cond)
        fake = G(x); comp = y * (1 - m) + fake * m
        # discriminateur
        dr, df = D(torch.cat([x, y], 1)), D(torch.cat([x, comp.detach()], 1))
        lD = 0.5 * (bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df)))
        oD.zero_grad(); lD.backward(); oD.step()
        # générateur
        df = D(torch.cat([x, comp], 1))
        lG = bce(df, torch.ones_like(df)) + 100 * F.l1_loss(fake * m, y * m)
        oG.zero_grad(); lG.backward(); oG.step()
    print(f"GAN époque {ep}: L_D={lD.item():.3f} L_G={lG.item():.3f}")
torch.save(G.state_dict(), OUT / "gan_G.pt")
""")
code(r"""
@torch.no_grad()
def gan_fill(patch_hu, cond):
    G.eval()
    y = torch.tensor(to_gan(patch_hu)[None, None], device=DEVICE)
    x, m = gan_inputs(y, torch.tensor([cond], device=DEVICE))
    out = (y * (1 - m) + G(x) * m)[0, 0].cpu().numpy()
    return from_gan(out)

def paste_center(img, lung, new_patch, cy, cx, rng):
    # Recolle la zone centrale générée (40x40) avec fenêtre gaussienne + bruit de camouflage.
    c = new_patch[P // 2 - 20:P // 2 + 20, P // 2 - 20:P // 2 + 20]
    return blend_patch(img, c, cy, cx, mad_sigma(img, lung), rng)

def atk_gan_inject(img, lung, rng):
    cy, cx = random_lung_point(lung, P // 2 + 2, rng)
    return paste_center(img, lung, gan_fill(img[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2], 1), cy, cx, rng)

def atk_gan_remove(img, lung, nod, rng):
    cy, cx, _ = nod
    return paste_center(img, lung, gan_fill(img[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2], 0), cy, cx, rng)

# Contrôle visuel
k = min(4, len(NOD_PATCHES))
fig, ax = plt.subplots(2, k, figsize=(3 * k, 6))
for j in range(k):
    ax[0, j].imshow(to_gan(NOD_PATCHES[j]), cmap="gray"); ax[0, j].set_title("réel (nodule)")
    ax[1, j].imshow(to_gan(gan_fill(NOD_PATCHES[j], 0)), cmap="gray"); ax[1, j].set_title("GAN : suppression")
for a in ax.ravel(): a.axis("off")
plt.show()
""")

md(r"""
## 5. Modèle de diffusion (DDPM) et inpainting de type RePaint

DDPM à $T=1000$ pas (planning linéaire de $\beta_t$), perte $\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$ (éq. 1.7), entraîné **uniquement sur des patchs pulmonaires authentiques**.
**Attaquant et défenseur disposent de deux modèles distincts** (`SEPARATE_DEFENDER_DDPM=True`), entraînés sur deux moitiés disjointes des patients d'entraînement : le défenseur ne connaît donc pas le générateur de l'attaquant, ce qui évite de surestimer la branche diffusion.
""")
code(r"""
SEPARATE_DEFENDER_DDPM = True
T = CFG["T_DIFF"]
betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

class TEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, 4 * d))
    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))

class ResB(nn.Module):
    def __init__(self, i, o, td=512):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, i), nn.Conv2d(i, o, 3, padding=1)
        self.t = nn.Linear(td, o)
        self.n2, self.c2 = nn.GroupNorm(8, o), nn.Conv2d(o, o, 3, padding=1)
        self.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        return self.c2(F.silu(self.n2(h))) + self.sk(x)

class DiffUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = TEmb(128); self.inc = nn.Conv2d(1, 64, 3, padding=1)
        self.d1, self.d2, self.d3, self.mid = ResB(64, 64), ResB(64, 128), ResB(128, 256), ResB(256, 256)
        self.u3, self.u2, self.u1 = ResB(512, 128), ResB(256, 64), ResB(128, 64)
        self.no, self.out = nn.GroupNorm(8, 64), nn.Conv2d(64, 1, 3, padding=1)
    def forward(self, x, t):
        te = self.te(t); x0 = self.inc(x)
        h1 = self.d1(x0, te); h2 = self.d2(F.avg_pool2d(h1, 2), te); h3 = self.d3(F.avg_pool2d(h2, 2), te)
        m = self.mid(F.avg_pool2d(h3, 2), te)
        u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), h3], 1), te)
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(F.silu(self.no(u)))

def train_ddpm(patches, epochs, tag):
    net = DiffUNet().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), 2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    X = to_gan(patches)
    for ep in range(epochs):
        perm = np.random.permutation(len(X)); tot = 0
        for b in range(0, len(perm) - 63, 64):
            x0 = torch.tensor(np.stack([augment_d4(X[j]).copy() for j in perm[b:b + 64]])[:, None], device=DEVICE)
            t = torch.randint(0, T, (len(x0),), device=DEVICE); eps = torch.randn_like(x0)
            xt = abar[t].sqrt()[:, None, None, None] * x0 + (1 - abar[t]).sqrt()[:, None, None, None] * eps
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = F.mse_loss(net(xt, t), eps)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        print(f"DDPM[{tag}] époque {ep}: perte {tot / max(1, len(perm) // 64):.4f}")
    torch.save(net.state_dict(), OUT / f"ddpm_{tag}.pt")
    return net.eval()

half = len(LUNG_PATCHES) // 2       # les patchs sont rangés patient par patient : les deux moitiés sont disjointes en patients (à un patient près)
ddpm_att = train_ddpm(LUNG_PATCHES[:half] if SEPARATE_DEFENDER_DDPM else LUNG_PATCHES, CFG["DDPM_EPOCHS"], "attaquant")
ddpm_def = train_ddpm(LUNG_PATCHES[half:], CFG["DDPM_EPOCHS"], "defenseur") if SEPARATE_DEFENDER_DDPM else ddpm_att
chrono("DDPM entraînés")
""")
code(r"""
@torch.no_grad()
def repaint(net, patch_hu, mask):
    # Inpainting RePaint (sans rééchantillonnage) : la région connue est réinjectée bruitée à chaque pas.
    x0 = torch.tensor(to_gan(patch_hu)[None, None], device=DEVICE)
    m = torch.tensor(mask[None, None].astype(np.float32), device=DEVICE)    # 1 = à générer
    x = torch.randn_like(x0)
    for t in reversed(range(T)):
        tt = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        eps = net(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x_unk = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        ab_prev = abar[t - 1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        x_kn = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * torch.randn_like(x0) if t > 0 else x0
        x = m * x_unk + (1 - m) * x_kn
    return from_gan(x[0, 0].clamp(-1, 1).cpu().numpy())

def atk_diffusion(img, lung, rng, nod=None):
    if nod is not None:
        cy, cx, _ = nod
    else:
        cy, cx = random_lung_point(lung, P // 2 + 2, rng)
    new = repaint(ddpm_att, img[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2], CENTER)
    return paste_center(img, lung, new, cy, cx, rng)
""")

md(r"""
## 5 bis. Construction du jeu falsifié

Chaque coupe candidate reçoit soit aucune attaque (authentique, probabilité `P_AUTH`), soit **une** attaque tirée parmi celles compatibles (les suppressions exigent un nodule dans la coupe). Le masque de vérité terrain est conservé.
""")
code(r"""
ATTACKS = ["authentique", "copier_deplacer", "inpainting", "bruit", "fgsm", "gan_injection", "gan_suppression", "diffusion"]
NEEDS_NODULE = {"inpainting", "gan_suppression"}
GLOBAL_ATTACKS = {"bruit", "fgsm"}

def apply_attack(name, img, lung, nods, rng):
    if name == "authentique":     return img, np.zeros(img.shape, bool)
    if name == "copier_deplacer": return atk_copy_move(img, lung, rng)
    if name == "inpainting":      return atk_inpaint(img, lung, nods[0], rng)
    if name == "bruit":           return atk_noise(img, rng)
    if name == "fgsm":            return atk_fgsm(img, lung, nodcls)
    if name == "gan_injection":   return atk_gan_inject(img, lung, rng)
    if name == "gan_suppression": return atk_gan_remove(img, lung, nods[0], rng)
    if name == "diffusion":       return atk_diffusion(img, lung, rng, nods[0] if nods and rng.random() < 0.5 else None)
    raise ValueError(name)

rng = np.random.default_rng(CFG["SEED"] + 2)
DATA = {}   # (pid, z) -> dict(img, mask, lung, attack, split)
rows = []; ATK_FAIL = {}
for (pid, z), c in CACHE.items():
    if rng.random() < CFG["P_AUTH"]:
        atk = "authentique"
    else:
        pool = [a for a in ATTACKS[1:] if (a not in NEEDS_NODULE) or c["nods"]]
        atk = pool[rng.integers(len(pool))]
    try:
        out, m = apply_attack(atk, c["img"], c["lung"], c["nods"], rng)
    except Exception as e:     # bord d'image, etc. -> coupe laissée authentique
        ATK_FAIL[atk] = ATK_FAIL.get(atk, 0) + 1
        atk, out, m = "authentique", c["img"], np.zeros(c["img"].shape, bool)
    DATA[(pid, z)] = dict(img=out, mask=m, lung=c["lung"], attack=atk, split=SPLIT[pid])
    rows.append(dict(pid=pid, z=z, split=SPLIT[pid], attack=atk, label=int(atk != "authentique"), area=float(m.mean())))
MANIFEST = pd.DataFrame(rows)
MANIFEST.to_csv(OUT / "manifest.csv", index=False)
print(pd.crosstab(MANIFEST.attack, MANIFEST.split))
print("attaques échouées (coupes laissées authentiques) :", ATK_FAIL)
CACHE.clear()      # libère la mémoire : DATA contient désormais toutes les coupes
chrono("jeu falsifié construit")
""")

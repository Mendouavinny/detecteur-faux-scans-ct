from common import *

W, H = 6.3, 3.40
fig, ax = canvas(W, H)
img, msk, lung = ct_images()

yc = 2.22
thumb(ax, img, 0.05, yc - 0.36, 0.72, 0.72)
text(ax, 0.41, yc - 0.42, "Coupe DICOM", fs=5.5, va="top", color=SOFT)
pp = box(ax, 0.92, yc - 0.52, 1.10, 1.04, "Prétraitement",
         "HU = $a\\cdot v+b$\n(pente et ordonnée\nDICOM)\nécrêtage commun\n[−1024 ; 3071] HU\nmasque pulmonaire", kind="own", fs=5.2)
arrow(ax, (0.77, yc), (pp[0], yc), ms=5)
pt = box(ax, 2.17, yc - 0.33, 0.80, 0.66, "Patchs", "64×64, pas 32\n≥ 50 % de\npoumon", kind="own", fs=5.2)
arrow(ax, (pp[0] + pp[2], yc), (pt[0], yc), ms=5)

br = [("Branche forensique", "résidus, 27 descripteurs, HGB\nfigure 2.7", "$p_{for}$", yc + 0.74),
      ("Branche CNN", "Bayar + SRM, EfficientNet-B0,\nU-Net ; figure 2.8", "$p_{cnn}$", yc),
      ("Branche diffusion", "erreur de reconstruction\ndu DDPM ; figure 2.9", "$z$, $|z|$", yc - 0.74)]
B = []
for t, b, o, y in br:
    bb = box(ax, 3.17, y - 0.27, 1.42, 0.54, t, b, kind="plain", fs=5.2)
    poly_arrow(ax, [(pt[0] + pt[2], yc), (3.07, yc), (3.07, y), (bb[0], y)] if y != yc else [(pt[0] + pt[2], yc), (bb[0], y)], ms=5)
    text(ax, bb[0] + bb[2] + 0.05, y + 0.04, o, fs=5.4, ha="left", va="bottom", color="#3A6EA5")
    B.append(bb)
fu = box(ax, 5.05, yc - 0.62, 1.20, 1.24, "Fusion local → global",
         "régression locale\nagrégation $A(\\cdot)$\n+ 40 descripteurs\nglobaux\nméta-classifieur HGB\nfigures 2.10 et 2.11", kind="own", fs=5.1)
for (t, b, o, y), bb in zip(br, B):
    poly_arrow(ax, [(bb[0] + bb[2], y), (4.94, y), (4.94, yc + (y - yc) * 0.5), (fu[0], yc + (y - yc) * 0.5)]
               if y != yc else [(bb[0] + bb[2], y), (fu[0], y)], ms=5)

# sorties
y3 = 0.55
sc = box(ax, 5.05, y3, 1.20, 0.46, "Score $s(x)$", "signalée si $s(x)\\geq\\tau_1$", kind="data", fs=5.2)
arrow(ax, (fu[0] + fu[2] / 2, fu[1]), (fu[0] + fu[2] / 2, y3 + 0.46), ms=5)
cf = box(ax, 3.45, y3, 1.38, 0.46, "Carte de fusion", "renvoyée par le prototype", kind="data", fs=5.2)
poly_arrow(ax, [(fu[0] + 0.15, fu[1]), (fu[0] + 0.15, y3 + 0.58), (cf[0] + cf[2] / 2, y3 + 0.58), (cf[0] + cf[2] / 2, y3 + 0.46)], ms=5)
cu = box(ax, 1.85, y3, 1.38, 0.46, "Carte U-Net", "issue de $\\hat m$ (branche CNN)", kind="data", fs=5.2)
ms = box(ax, 0.05, y3, 1.58, 0.46, "Module de sécurité", "prototype, section 2.2.8", kind="plain", fs=5.2, ls=(0, (3, 2)))
text(ax, 0.05, 0.30, "Données : LIDC-IDRI, 300 patients (210 entraînement, 45 validation, 45 test) ; tests externes CT-GAN et M3Dsynth sans réapprentissage",
     fs=5.0, ha="left", color=SOFT)
legend(ax, 0.05, 0.03, items=("own", "plain", "out"))
save(fig, "figures/arch_vue.png")

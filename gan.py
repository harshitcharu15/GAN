# cgan_celebA_blond.py
import os, math, random, itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
CHANNELS = 3
BATCH = 128
EPOCHS = 20
NZ = 128                 # noise dim
NCOND = 1               # single binary attribute Blond_Hair
G_LR = 2e-4
D_LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
N_CRITIC = 5            # WGAN-GP: train critic more often
LAMBDA_GP = 10.0
SAMPLE_EVERY = 500
OUTDIR = Path("samples"); OUTDIR.mkdir(exist_ok=True, parents=True)
SEED = 42
torch.manual_seed(SEED); random.seed(SEED)

# -----------------------------
# Data: CelebA + Blond_Hair
# -----------------------------
transform = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

ds = datasets.CelebA(root="data", split="train", target_type="attr", transform=transform, download=True)
# Find Blond_Hair index robustly
attr_names = ds.attr_names
blond_idx = attr_names.index("Blond_Hair")
def blond_label(attr_row):
    # CelebA stores attributes as {-1, +1}; convert to {0,1}
    return 1 if attr_row[blond_idx].item() == 1 else 0

class CelebABlond(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, attrs = self.base[i]
        y = blond_label(attrs)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

train_ds = CelebABlond(ds)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# -----------------------------
# Models (DCGAN-style)
# Conditioning rule:
#   - G: concat z with y (vector)
#   - D: concat image with y as a 1xHxW mask (channel-wise)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, nz, ncond):
        super().__init__()
        in_z = nz + ncond
        ngf = 64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_z, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z, y):
        # z: (B, nz); y: (B, 1)
        zy = torch.cat([z, y], dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, nz+1, 1, 1)
        return self.net(zy)

class Critic(nn.Module):
    def __init__(self, ncond):
        super().__init__()
        n_in = CHANNELS + ncond
        ndf = 64
        self.net = nn.Sequential(
            nn.Conv2d(n_in, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*2, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*4, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.InstanceNorm2d(ndf*8, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x, y):
        # x: (B,3,H,W); y: (B,1)
        B, _, H, W = x.shape
        y_img = y.view(B,1,1,1).expand(B,1,H,W)    # label mask channel
        xy = torch.cat([x, y_img], dim=1)
        score = self.net(xy).view(-1)
        return score

G = Generator(NZ, NCOND).to(DEVICE)
D = Critic(NCOND).to(DEVICE)
optG = torch.optim.Adam(G.parameters(), lr=G_LR, betas=(BETA1, BETA2))
optD = torch.optim.Adam(D.parameters(), lr=D_LR, betas=(BETA1, BETA2))

# Fixed noise for monitoring
FIXED = 16
fixed_z = torch.randn(FIXED, NZ, device=DEVICE)
fixed_y0 = torch.zeros(FIXED, 1, device=DEVICE)
fixed_y1 = torch.ones(FIXED, 1, device=DEVICE)

# -----------------------------
# WGAN-GP utilities
# -----------------------------
def gradient_penalty(D, real, fake, y):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=real.device)
    xhat = eps*real + (1-eps)*fake
    xhat.requires_grad_(True)
    d_hat = D(xhat, y)
    grads = torch.autograd.grad(outputs=d_hat, inputs=xhat,
                                grad_outputs=torch.ones_like(d_hat),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((grads.view(B, -1).norm(2, dim=1) - 1)**2).mean()
    return gp

# -----------------------------
# TRAIN STEP (key asked change)
#  - match input formats for G and D:
#    * G takes (z, y_vector)
#    * D takes (image, y_mask_channel)
# -----------------------------
global_step = 0
for epoch in range(1, EPOCHS+1):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # --- train Critic N_CRITIC times
        for _ in range(N_CRITIC):
            z = torch.randn(x.size(0), NZ, device=DEVICE)
            with torch.no_grad():
                fake = G(z, y)
            d_real = D(x, y).mean()
            d_fake = D(fake, y).mean()
            gp = gradient_penalty(D, x, fake.detach(), y)
            d_loss = (d_fake - d_real) + LAMBDA_GP*gp

            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

        # --- train Generator once
        z = torch.randn(x.size(0), NZ, device=DEVICE)
        fake = G(z, y)
        g_loss = -D(fake, y).mean()

        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()

        # --- log & samples
        if global_step % SAMPLE_EVERY == 0:
            with torch.no_grad():
                grid0 = G(fixed_z, fixed_y0).cpu()
                grid1 = G(fixed_z, fixed_y1).cpu()
                grid = torch.cat([grid0, grid1], dim=0)
                utils.save_image(grid, OUTDIR / f"epoch{epoch:03d}_step{global_step:06d}.png",
                                 nrow=FIXED, normalize=True, value_range=(-1, 1))
            print(f"[ep {epoch:02d} | step {global_step:06d}] "
                  f"D_loss: {d_loss.item():.3f} (gp {gp.item():.3f}) | G_loss: {g_loss.item():.3f}")

        global_step += 1

# final samples
with torch.no_grad():
    z = torch.randn(64, NZ, device=DEVICE)
    y0 = torch.zeros(64, 1, device=DEVICE)
    y1 = torch.ones(64, 1, device=DEVICE)
    out0 = G(z, y0).cpu(); out1 = G(z, y1).cpu()
    utils.save_image(out0, OUTDIR / "final_not_blond.png", nrow=8, normalize=True, value_range=(-1,1))
    utils.save_image(out1, OUTDIR / "final_blond.png", nrow=8, normalize=True, value_range=(-1,1))

print("Done. Samples saved to:", OUTDIR.resolve())

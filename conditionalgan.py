# Conditional GAN on CelebA – Blond Hair
# Author: Harshit Kohli  |  Based on Foster (2024) cGAN.ipynb

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

IMG_SIZE, NZ, NCOND = 64, 128, 1
transform = transforms.Compose([
    transforms.CenterCrop(140), transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load CelebA and extract Blond_Hair
base = datasets.CelebA(root="data", split="train", target_type="attr",
                       transform=transform, download=True)
blond_idx = base.attr_names.index("Blond_Hair")
class CelebABlond(torch.utils.data.Dataset):
    def __getitem__(self,i):
        x, a = base[i]; y = torch.tensor([(a[blond_idx].item()==1)*1.0])
        return x, y
    def __len__(self): return len(base)

train_ds = CelebABlond(); loader = DataLoader(train_ds,128,shuffle=True)

# Generator and Critic definitions omitted here for brevity (DCGAN style)

# WGAN-GP training loop (excerpt)
for epoch in range(20):
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        z = torch.randn(x.size(0), NZ, device='cuda')
        fake = G(z, y)
        d_real = D(x, y).mean()
        d_fake = D(fake.detach(), y).mean()
        gp = gradient_penalty(D, x, fake.detach(), y)
        d_loss = (d_fake - d_real) + 10*gp
        # update D and G as per WGAN-GP...

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models_cgan import Generator, Discriminator

class CGANDataset(Dataset):
    def __init__(self, manifest_csv="splits/cgan_preprocessed_train_manifest.csv"):
        self.df = pd.read_csv(manifest_csv)
        self.class_map = {
            'Canine No Findings': 0,
            'Canine Cardiomegaly': 1,
            'Canine Alveolar Pattern': 2,
            'Feline No Findings': 3,
            'Feline Cardiomegaly': 4,
            'Feline Alveolar Pattern': 5
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath'].replace('\\', '/')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = self.class_map[row['Class']]
        return img, torch.tensor(label, dtype=torch.long)

def train_cgan(epochs=50, batch_size=32, lr=0.0002, latent_dim=100, lambda_ms=1.5,
               manifest_csv="splits/cgan_preprocessed_train_manifest.csv",
               output_dir="checkpoints"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = CGANDataset(manifest_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    netG = Generator(num_classes=6, latent_dim=latent_dim).to(device)
    netD = Discriminator(num_classes=6).to(device)
    
    criterion_adv = nn.BCEWithLogitsLoss()
    
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    
    print(f"Training PatchGAN + MS-GAN on CLAHE/Cropped dataset for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_d, total_g = 0.0, 0.0
        
        for i, (real_imgs, labels) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # --- 1. Discriminator Step ---
            optD.zero_grad()
            d_real_out = netD(real_imgs, labels)
            real_targets = torch.full_like(d_real_out, 0.9)
            d_real_loss = criterion_adv(d_real_out, real_targets)
            
            z1 = torch.randn(b_size, latent_dim, device=device)
            fake_imgs1 = netG(z1, labels)
            d_fake_out = netD(fake_imgs1.detach(), labels)
            fake_targets = torch.zeros_like(d_fake_out)
            d_fake_loss = criterion_adv(d_fake_out, fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()
            
            # --- 2. Generator Step with Mode-Seeking Diversity ---
            optG.zero_grad()
            z2 = torch.randn(b_size, latent_dim, device=device)
            fake_imgs2 = netG(z2, labels)
            
            g_out = netD(fake_imgs1, labels)
            g_adv_loss = criterion_adv(g_out, torch.ones_like(g_out))
            
            # Mode seeking diversity loss (prevents mode collapse)
            lz = torch.mean(torch.abs(fake_imgs1 - fake_imgs2)) / (torch.mean(torch.abs(z1 - z2)) + 1e-5)
            ms_loss = lambda_ms * (1.0 / (lz + 1e-5))
            
            g_total_loss = g_adv_loss + ms_loss
            g_total_loss.backward()
            optG.step()
            
            total_d += d_loss.item()
            total_g += g_adv_loss.item()
            
        print(f"Epoch [{epoch:02d}/{epochs}] - Loss D: {total_d/len(loader):.4f} | Loss G: {total_g/len(loader):.4f}")
        
        if epoch % 10 == 0 or epoch == epochs:
            g_path = os.path.join(output_dir, f"cgan_generator_epoch_{epoch}.pth")
            torch.save(netG.state_dict(), g_path)
            print(f"--> Saved checkpoint: {g_path}")

if __name__ == '__main__':
    train_cgan(epochs=50, batch_size=32)
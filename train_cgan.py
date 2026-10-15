import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models_cgan import Generator, Discriminator

class CGANDataset(Dataset):
    def __init__(self, manifest_csv):
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
        img_path = row['filepath']
        
        if not os.path.exists(img_path):
            alt_path = img_path.replace('\\', '/')
            if os.path.exists(alt_path):
                img_path = alt_path

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1] for Tanh
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        label = self.class_map[row['Class']]
        return img, torch.tensor(label, dtype=torch.long)


def train_cgan(epochs=50, batch_size=32, lr=0.0002, latent_dim=100, lambda_ms=1.5, output_dir="checkpoints"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CGANDataset("splits/model1_train_base_manifest.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    netG = Generator(num_classes=6, latent_dim=latent_dim).to(device)
    netD = Discriminator(num_classes=6).to(device)
    
    # PatchGAN loss with label smoothing
    criterion = nn.BCEWithLogitsLoss()
    
    # Standard DCGAN Adam hyperparameters
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))  # slower D for balance
    
    print(f"Starting Mode-Seeking cGAN training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_ms_loss = 0.0
        
        for i, (real_imgs, labels) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # --- 1. Train Discriminator ---
            optD.zero_grad()
            d_real_out = netD(real_imgs, labels)
            real_targets = torch.full_like(d_real_out, 0.9)  # One-sided label smoothing
            d_real_loss = criterion(d_real_out, real_targets)
            
            z1 = torch.randn(b_size, latent_dim, device=device)
            fake_imgs1 = netG(z1, labels)
            d_fake_out = netD(fake_imgs1.detach(), labels)
            fake_targets = torch.zeros_like(d_fake_out)
            d_fake_loss = criterion(d_fake_out, fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()
            
            # --- 2. Train Generator with Mode-Seeking Loss ---
            optG.zero_grad()
            
            # Generate second set of images with different random noise to evaluate diversity
            z2 = torch.randn(b_size, latent_dim, device=device)
            fake_imgs2 = netG(z2, labels)
            
            g_out = netD(fake_imgs1, labels)
            g_adv_loss = criterion(g_out, torch.ones_like(g_out))
            
            # Mode-Seeking Diversity Penalty: prevents z1 and z2 from collapsing to the same image
            lz = torch.mean(torch.abs(fake_imgs1 - fake_imgs2)) / (torch.mean(torch.abs(z1 - z2)) + 1e-5)
            eps_ms = 1.0 / (lz + 1e-5)
            ms_loss = lambda_ms * eps_ms
            
            g_total_loss = g_adv_loss + ms_loss
            g_total_loss.backward()
            optG.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_adv_loss.item()
            total_ms_loss += ms_loss.item()
            
        avg_d = total_d_loss / len(loader)
        avg_g = total_g_loss / len(loader)
        avg_ms = total_ms_loss / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs}] - Loss D: {avg_d:.4f} | Loss G: {avg_g:.4f} | MS Diversity Penalty: {avg_ms:.4f}")
        
        if epoch % 10 == 0 or epoch == epochs:
            g_path = os.path.join(output_dir, f"cgan_generator_epoch_{epoch}.pth")
            d_path = os.path.join(output_dir, f"cgan_discriminator_epoch_{epoch}.pth")
            torch.save(netG.state_dict(), g_path)
            torch.save(netD.state_dict(), d_path)
            print(f"--> Saved checkpoint: {g_path}")

if __name__ == '__main__':
    train_cgan(epochs=50, batch_size=32, lr=0.0002)
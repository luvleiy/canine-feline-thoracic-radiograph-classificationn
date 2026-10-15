import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models_cgan import Generator, Discriminator

# 1. Custom Dataset scaled to [-1, 1] for Tanh Generator outputs
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
        
        # Cross-platform fallback for pathing
        if not os.path.exists(img_path):
            alt_path = img_path.replace('\\', '/')
            if os.path.exists(alt_path):
                img_path = alt_path

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        label = self.class_map[row['Class']]
        return img, torch.tensor(label, dtype=torch.long)

# 2. Main Training Procedure
def train_cgan(epochs=50, batch_size=32, lr=0.0002, latent_dim=100, output_dir="checkpoints"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CGANDataset("splits/model1_train_base_manifest.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    netG = Generator(num_classes=6, latent_dim=latent_dim).to(device)
    netD = Discriminator(num_classes=6).to(device)
    
    # BCEWithLogitsLoss is numerically stable and pairs with our spectral-norm Discriminator logits
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizers with standard DCGAN beta values
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    for epoch in range(1, epochs + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        
        for i, (real_imgs, labels) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # Label smoothing: real labels = 0.9, fake labels = 0.0
            real_targets = torch.full((b_size, 1), 0.9, device=device)
            fake_targets = torch.zeros(b_size, 1, device=device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            d_real_out = netD(real_imgs, labels)
            d_real_loss = criterion(d_real_out, real_targets)
            
            z = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = netG(z, labels)
            d_fake_out = netD(fake_imgs.detach(), labels)
            d_fake_loss = criterion(d_fake_out, fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            # Generator wants discriminator to think fake images are real (target: 1.0)
            gen_targets = torch.ones(b_size, 1, device=device)
            g_out = netD(fake_imgs, labels)
            g_loss = criterion(g_out, gen_targets)
            g_loss.backward()
            optG.step()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
        avg_d_loss = total_d_loss / len(loader)
        avg_g_loss = total_g_loss / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs}] - Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")
        
        # Save checkpoints periodically
        if epoch % 10 == 0 or epoch == epochs:
            g_path = os.path.join(output_dir, f"cgan_generator_epoch_{epoch}.pth")
            d_path = os.path.join(output_dir, f"cgan_discriminator_epoch_{epoch}.pth")
            torch.save(netG.state_dict(), g_path)
            torch.save(netD.state_dict(), d_path)
            print(f"--> Checkpoint saved: {g_path}")

if __name__ == '__main__':
    train_cgan(epochs=50, batch_size=64, lr=0.0002)
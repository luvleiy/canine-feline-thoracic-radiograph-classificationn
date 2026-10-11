import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models_cgan import Generator, Discriminator

# 1. Custom Dataset specifically scaled to [-1, 1] for Tanh Generator outputs
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
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = torch.tensor(img).permute(2, 0, 1)
        
        label = self.class_map[row['Class']]
        return img, torch.tensor(label, dtype=torch.long)

# 2. Main Training Procedure
def train_cgan(epochs=50, batch_size=32, lr=0.0002, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = CGANDataset("splits/model1_train_base_manifest.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    netG = Generator(num_classes=6, latent_dim=latent_dim).to(device)
    netD = Discriminator(num_classes=6).to(device)
    
    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        for i, (real_imgs, labels) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            real_targets = torch.ones(b_size, 1, device=device)
            fake_targets = torch.zeros(b_size, 1, device=device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            d_real_loss = criterion(netD(real_imgs, labels), real_targets)
            
            z = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = netG(z, labels)
            d_fake_loss = criterion(netD(fake_imgs.detach(), labels), fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            g_loss = criterion(netD(fake_imgs, labels), real_targets)
            g_loss.backward()
            optG.step()
            
        print(f"Epoch [{epoch}/{epochs}] - Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
        
        # Save checkpoints periodically
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(netG.state_dict(), f"checkpoints/cgan_generator_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"checkpoints/cgan_discriminator_epoch_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch}.")

if __name__ == '__main__':
    train_cgan(epochs=1, batch_size=32)
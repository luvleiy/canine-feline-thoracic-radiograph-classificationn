import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class Generator(nn.Module):
    def __init__(self, num_classes=6, latent_dim=100, embed_dim=64, feature_maps=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        in_dim = latent_dim + embed_dim
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, feature_maps * 8 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 8 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.stage1 = ResUpsampleBlock(feature_maps * 8, feature_maps * 4)  # 7 -> 14
        self.stage2 = ResUpsampleBlock(feature_maps * 4, feature_maps * 2)  # 14 -> 28
        self.stage3 = ResUpsampleBlock(feature_maps * 2, feature_maps)      # 28 -> 56
        self.stage4 = ResUpsampleBlock(feature_maps, feature_maps // 2)     # 56 -> 112
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                    # 112 -> 224
            nn.Conv2d(feature_maps // 2, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 7, 7)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.final(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, num_classes=6, feature_maps=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 224 * 224)
        
        # 3 (image) + 1 (label) = 4 channels
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(4, feature_maps, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(feature_maps * 8, 1, kernel_size=3, stride=1, padding=1))
        )

    def forward(self, img, labels):
        b, _, h, w = img.shape
        c = self.label_embedding(labels).view(b, 1, h, w)
        x = torch.cat([img, c], dim=1)
        return self.net(x)

Discriminator = PatchDiscriminator
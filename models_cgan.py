import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class UpsampleBlock(nn.Module):
    """
    Nearest-neighbor upsample followed by Conv2d completely eliminates
    transposed convolution checkerboard artifacts.
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, num_classes=6, latent_dim=100, embed_dim=50, feature_maps=64):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        in_dim = latent_dim + embed_dim
        
        # Initial projection: latent vector -> 7x7 spatial map
        self.fc = nn.Sequential(
            nn.Linear(in_dim, feature_maps * 8 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsampling stages: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.stage1 = UpsampleBlock(feature_maps * 8, feature_maps * 4)  # 7 -> 14
        self.stage2 = UpsampleBlock(feature_maps * 4, feature_maps * 2)  # 14 -> 28
        self.stage3 = UpsampleBlock(feature_maps * 2, feature_maps)      # 28 -> 56
        self.stage4 = UpsampleBlock(feature_maps, feature_maps // 2)     # 56 -> 112
        
        # Final block to output resolution 224x224 (3 channels, normalized [-1, 1])
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                 # 112 -> 224
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


class Discriminator(nn.Module):
    def __init__(self, num_classes=6, feature_maps=32):
        super(Discriminator, self).__init__()
        # Project label into spatial channel map (1 x 224 x 224)
        self.label_embedding = nn.Embedding(num_classes, 224 * 224)
        
        # Input channels: 3 (image) + 1 (label map) = 4 channels
        # Spectral normalization controls the Lipschitz constant and stabilizes training
        self.net = nn.Sequential(
            # 224x224 -> 112x112
            spectral_norm(nn.Conv2d(4, feature_maps, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            spectral_norm(nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1 scalar
            spectral_norm(nn.Conv2d(feature_maps * 8, 1, kernel_size=7, stride=1, padding=0)),
            nn.Flatten()
        )

    def forward(self, img, labels):
        b, _, h, w = img.shape
        c = self.label_embedding(labels).view(b, 1, h, w)
        x = torch.cat([img, c], dim=1)
        return self.net(x)
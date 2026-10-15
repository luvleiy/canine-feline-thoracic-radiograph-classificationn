import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class AddCoords(nn.Module):
    """Appends explicit (x, y) spatial coordinates to give the network global orientation."""
    def forward(self, x):
        b, _, h, w = x.shape
        xx_channel = torch.arange(w, dtype=torch.float32, device=x.device).repeat(1, 1, h, 1)
        yy_channel = torch.arange(h, dtype=torch.float32, device=x.device).repeat(1, 1, w, 1).permute(0, 1, 3, 2)
        
        xx_channel = (xx_channel / (w - 1)) * 2.0 - 1.0
        yy_channel = (yy_channel / (h - 1)) * 2.0 - 1.0
        
        xx_channel = xx_channel.repeat(b, 1, 1, 1)
        yy_channel = yy_channel.repeat(b, 1, 1, 1)
        return torch.cat([x, xx_channel, yy_channel], dim=1)


class SelfAttention(nn.Module):
    """Self-Attention block to maintain long-range anatomical consistency (spine, ribs, heart)."""
    def __init__(self, in_dim):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.key   = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        proj_query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key   = self.key(x).view(b, -1, h * w)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(b, -1, h * w)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


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
        self.attn   = SelfAttention(feature_maps * 2)                       # Attention at 28x28
        self.stage3 = ResUpsampleBlock(feature_maps * 2, feature_maps)      # 28 -> 56
        self.stage4 = ResUpsampleBlock(feature_maps, feature_maps // 2)     # 56 -> 112
        
        # Output with CoordConv to bind edges to spatial coordinates
        self.add_coords = AddCoords()
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                    # 112 -> 224
            nn.Conv2d((feature_maps // 2) + 2, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 7, 7)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.attn(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.add_coords(x)
        return self.final(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, num_classes=6, feature_maps=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 224 * 224)
        self.add_coords = AddCoords()
        
        # 3 (image) + 1 (label) + 2 (coords) = 6 channels
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(6, feature_maps, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn = SelfAttention(feature_maps * 2)
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.head = spectral_norm(nn.Conv2d(feature_maps * 8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, img, labels, return_features=False):
        b, _, h, w = img.shape
        c = self.label_embedding(labels).view(b, 1, h, w)
        x = torch.cat([img, c], dim=1)
        x = self.add_coords(x)
        
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f2 = self.attn(f2)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = self.head(f4)
        
        if return_features:
            return out, [f2, f3, f4]
        return out

Discriminator = PatchDiscriminator
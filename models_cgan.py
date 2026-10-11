import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=6, latent_dim=100, embed_dim=50, feature_maps=64):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Total input dimension: latent vector z + class embedding
        in_dim = latent_dim + embed_dim
        
        # Initial projection to feature map of size 7x7
        self.fc = nn.Sequential(
            nn.Linear(in_dim, feature_maps * 8 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsampling stages: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.net = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(feature_maps // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=6, feature_maps=32):
        super(Discriminator, self).__init__()
        # Project label into spatial channel map (1 x 224 x 224)
        self.label_embedding = nn.Embedding(num_classes, 224 * 224)
        
        # Input channels: 3 (image) + 1 (label map) = 4 channels
        self.net = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(4, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1 scalar probability
            nn.Conv2d(feature_maps * 16, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        b, _, h, w = img.shape
        c = self.label_embedding(labels).view(b, 1, h, w)
        x = torch.cat([img, c], dim=1)
        return self.net(x).view(b, -1)
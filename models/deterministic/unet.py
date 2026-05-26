# models/unet.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Contracting path
        self.d_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)  # padding='same' remplacé par 1
        self.d_pool1 = nn.MaxPool2d(2)
        self.d_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.d_pool2 = nn.MaxPool2d(2)
        self.d_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.d_pool3 = nn.MaxPool2d(2)
        self.d_conv4 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Expanding path
        self.u_upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.u_conv1 = nn.Conv2d(128, 64, 3, padding=1)   # concat avec x3 (64)
        
        self.u_upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1)
        )
        self.u_conv2 = nn.Conv2d(64, 32, 3, padding=1)    # 32 (up) + 32 (x2)
        
        self.u_upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1)
        )
        self.u_conv3 = nn.Conv2d(32, 16, 3, padding=1)    # 16 (up) + 16 (x1)
        
        self.u_conv4 = nn.Conv2d(16, out_channels, 3, padding=1)
        self.final_activation = nn.Tanh()   # pour sortie dans [-1, 1]

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.d_conv1(x))
        p1 = self.d_pool1(x1)
        x2 = torch.relu(self.d_conv2(p1))
        p2 = self.d_pool2(x2)
        x3 = torch.relu(self.d_conv3(p2))
        p3 = self.d_pool3(x3)
        x4 = torch.relu(self.d_conv4(p3))
        
        # Decoder
        u1 = torch.relu(self.u_upconv1(x4))
        u1 = torch.cat([u1, x3], dim=1)
        u1 = torch.relu(self.u_conv1(u1))
        
        u2 = torch.relu(self.u_upconv2(u1))
        u2 = torch.cat([u2, x2], dim=1)
        u2 = torch.relu(self.u_conv2(u2))
        
        u3 = torch.relu(self.u_upconv3(u2))
        u3 = torch.cat([u3, x1], dim=1)
        u3 = torch.relu(self.u_conv3(u3))
        
        out = self.u_conv4(u3)
        out = self.final_activation(out)
        return out
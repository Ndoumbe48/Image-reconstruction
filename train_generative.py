"""
train_generative.py
Entraînement d'un VAE-GAN pour la restauration d'images sur LSDIR.
Version adaptée pour la lecture directe depuis S3 (Onyxia).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import os
import argparse
import time
from tqdm import tqdm
from pathlib import Path

# Imports locaux
from dataset import LSDIRDataset   # version avec support S3
from models.vae_gan import VAE_GAN, weights_init

# Récupération des identifiants S3 (variables d'environnement Onyxia)
S3_ENDPOINT = os.environ.get('AWS_S3_ENDPOINT')
S3_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
S3_SECRET = os.environ.get('AWS_SECRET_ACCESS_KEY')

# --- Fonctions métriques (inchangées) ---
def calculate_psnr(img1, img2):
    """Calcul du PSNR entre deux tenseurs d'images normalisés [-1,1]"""
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """SSIM simplifié avec fallback"""
    try:
        from pytorch_ssim import SSIM
        ssim = SSIM()
        return ssim(img1, img2)
    except ImportError:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1 = torch.var(img1)
        sigma2 = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_val

class VAEGANTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Transformations (identique)
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # DataLoaders (avec S3)
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Modèle VAE-GAN (3 canaux, taille 64x64 par défaut)
        self.model = VAE_GAN(in_channels=3, latent_dim=args.latent_dim).to(self.device)
        self.model.apply(weights_init)
        
        # Optimiseurs
        self.optimizer_E = optim.Adam(self.model.encoder.parameters(), lr=args.lr)
        self.optimizer_Dec = optim.Adam(self.model.decoder.parameters(), lr=args.lr)
        self.optimizer_Dis = optim.Adam(self.model.discriminator.parameters(), lr=args.lr * args.alpha)
        
        self.bce_loss = nn.BCELoss()
        self.writer = SummaryWriter(args.log_dir)
        self.best_val_psnr = 0.0
        self.current_epoch = 0
        
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_dataloaders(self):
        """Crée les DataLoaders pour train/val à partir de S3"""
        # Utilise les chemins S4 passés en arguments (par ex. s3://bucket/...)
        dataset = LSDIRDataset(
            hr_root=self.args.hr_dir,
            lr_root=self.args.lr_dir,
            s3_endpoint=S3_ENDPOINT,
            s3_key=S3_KEY,
            s3_secret=S3_SECRET,
            transform=self.transform,
            image_size=self.args.image_size
        )
        
        # Split 90/10
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(42))
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                  shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        
        print(f"📂 Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
        return train_loader, val_loader
    
    # --- Les méthodes train_one_epoch, validate, log_images, save_checkpoint, train sont inchangées ---
    # (recopier exactement les mêmes fonctions que dans ton script original, je les omets pour la lisibilité)
    
    # ...

def main():
    parser = argparse.ArgumentParser(description='Train VAE-GAN on LSDIR (S3)')
    
    # Chemins S3 (obligatoires, par ex. "s3://mon-bucket/LSDIR/HR/train" et "s3://.../X2/train")
    parser.add_argument('--hr_dir', type=str, required=True,
                        help='S3 path to HR images (clean)')
    parser.add_argument('--lr_dir', type=str, required=True,
                        help='S3 path to LR/X2 images (degraded)')
    
    # Hyperparamètres (par défaut adaptés à 64x64)
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Discriminator LR factor')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL divergence weight')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--adv_weight', type=float, default=1.0,
                        help='Adversarial loss weight')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_images_every', type=int, default=5,
                        help='Log images every N epochs')
    
    args = parser.parse_args()
    
    # Vérification que les variables S3 sont bien définies
    if not S3_ENDPOINT or not S3_KEY or not S3_SECRET:
        print("⚠️  Variables S3 non trouvées. Assurez-vous d'être dans un service Onyxia avec accès au stockage.")
    else:
        print("✅ Identifiants S3 récupérés depuis l'environnement.")
    
    trainer = VAEGANTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
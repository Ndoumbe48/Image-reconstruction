# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM

class Losses(nn.Module):
    def __init__(self, beta=1.0, perceptual_weight=0.1, adv_weight=1.0):
        super().__init__()
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        self.bce = nn.BCELoss()          # pour les probabilités sigmoïdes
        self.mse = nn.MSELoss()
        self.ssim_module = SSIM()

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / mu.size(0)

    def l1_loss(self, recon, target):
        return F.l1_loss(recon, target)

    def perceptual_loss(self, recon, target, lpips_fn):
        return lpips_fn(recon, target).mean()

    # Perte simple pour le modèle déterministe (U-Net, autoencodeur)
    def deterministic_loss(self, recon, target, loss_type='l1'):
        if loss_type == 'l1':
            return self.l1_loss(recon, target)
        elif loss_type == 'mse':
            return self.mse(recon, target)
        elif loss_type == 'ssim':
            return 1 - self.ssim_module(recon, target).mean()
        else:
            # combinaison L1 + SSIM
            return self.l1_loss(recon, target) + (1 - self.ssim_module(recon, target).mean())

    # Perte pour le générateur VAE-GAN
    def vae_gan_generator_loss(self, recon, target, mu, logvar, fake_validity, lpips_fn):
        recon_loss = self.l1_loss(recon, target) + (1 - self.ssim_module(recon, target).mean())
        kl_loss = self.kl_divergence(mu, logvar)
        perc_loss = self.perceptual_loss(recon, target, lpips_fn)
        adv_loss = self.bce(fake_validity, torch.ones_like(fake_validity))
        total = recon_loss + self.beta * kl_loss + self.perceptual_weight * perc_loss + self.adv_weight * adv_loss
        return total, recon_loss, kl_loss, perc_loss, adv_loss

    # Perte pour le discriminateur du VAE-GAN
    def discriminator_loss(self, real_validity, fake_validity):
        real_loss = self.bce(real_validity, torch.ones_like(real_validity))
        fake_loss = self.bce(fake_validity, torch.zeros_like(fake_validity))
        return (real_loss + fake_loss) / 2
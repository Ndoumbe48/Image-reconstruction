# metrics.py
import torch
import torch.nn.functional as F
from piqa import SSIM, PSNR

class ImageMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.ssim = SSIM().to(device)
        self.psnr = PSNR().to(device)

    def psnr(self, img1, img2):
        return self.psnr(img1, img2)

    def ssim(self, img1, img2):
        return self.ssim(img1, img2)

    def rmse(self, img1, img2):
        return torch.sqrt(F.mse_loss(img1, img2))

    def lpips(self, img1, img2, lpips_fn):
        return lpips_fn(img1, img2).mean()
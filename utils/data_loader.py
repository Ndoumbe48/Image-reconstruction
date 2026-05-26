import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import s3fs  # <-- Nouvel import

class LSDIRDataset(Dataset):
    def __init__(self, hr_root, lr_root, transform=None, image_size=128, s3_endpoint=None, s3_key=None, s3_secret=None):
        """
        hr_root : chemin S3 du dossier HR (ex: 's3://mon-bucket/LSDIR/HR/train')
        lr_root : chemin S3 du dossier LR (ex: 's3://mon-bucket/LSDIR/X2/train')
        s3_endpoint : l'URL de l'API S3 (ex: 'https://minio.lab.sspcloud.fr')
        s3_key, s3_secret : les identifiants de connexion S3
        """
        self.fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': s3_endpoint},
            key=s3_key,
            secret=s3_secret
        )
        self.hr_paths = []
        self.lr_paths = []

        # Récupération de la liste des fichiers HR depuis S3
        hr_files = [f for f in self.fs.glob(os.path.join(hr_root, '**', '*.[jJ][pP][gG]'))] + \
                   [f for f in self.fs.glob(os.path.join(hr_root, '**', '*.[pP][nN][gG]'))]
        
        for hr_path in hr_files:
            rel_path = os.path.relpath(hr_path, hr_root)
            lr_path = os.path.join(lr_root, rel_path)
            if self.fs.exists(lr_path):
                self.hr_paths.append(hr_path)
                self.lr_paths.append(lr_path)
        print(f"Found {len(self.hr_paths)} image pairs")
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # Ouverture des fichiers directement depuis S3 via s3fs
        with self.fs.open(self.hr_paths[idx], 'rb') as f:
            hr_img = Image.open(f).convert('RGB')
        with self.fs.open(self.lr_paths[idx], 'rb') as f:
            lr_img = Image.open(f).convert('RGB')
        
        hr_img = hr_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        lr_img = lr_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        return lr_img, hr_img
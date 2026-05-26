import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os

# Imports locaux
from dataset import LSDIRDataset   # la version avec support S3
from models.deterministic import UNet

# --- Récupération des identifiants S3 (variables d'environnement Onyxia) ---
S3_ENDPOINT = os.environ.get('AWS_S3_ENDPOINT')
S3_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
S3_SECRET = os.environ.get('AWS_SECRET_ACCESS_KEY')

# --- Chemins vers vos dossiers sur S3 (à adapter) ---
HR_S3_PATH = "s3://votre-bucket/LSDIR/HR/train"   # dossier contenant les 4000 HR
LR_S3_PATH = "s3://votre-bucket/LSDIR/X2/train"   # dossier contenant les 4000 X2 correspondants

# --- Paramètres d'entraînement ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
epochs = 20
learning_rate = 1e-4
image_size = 64   # doit correspondre à l'architecture du modèle (64x64)

# --- Transformations (normalisation dans [-1, 1]) ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Dataset complet (train + val) ---
full_dataset = LSDIRDataset(
    hr_root=HR_S3_PATH,
    lr_root=LR_S3_PATH,
    s3_endpoint=S3_ENDPOINT,
    s3_key=S3_KEY,
    s3_secret=S3_SECRET,
    transform=transform,
    image_size=image_size
)

# Séparation train/val (90% / 10%)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- Modèle U-Net (avec sortie Tanh) ---
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter("logs_deterministic")

# --- Boucle d'entraînement ---
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for degraded, clean in train_loader:
        degraded, clean = degraded.to(device), clean.to(device)
        recon = model(degraded)
        loss = criterion(recon, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for degraded, clean in val_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            recon = model(degraded)
            val_loss += criterion(recon, clean).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    # Sauvegarde périodique
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

writer.close()
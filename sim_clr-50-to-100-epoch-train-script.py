import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================
CATEGORIES = ["bottle", "screw", "metal_nut", "capsule", "cable"]
DATA_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
CHECKPOINT_PATH = "/content/drive/MyDrive/BIG5/BIG5.pth"
ADDITIONAL_EPOCHS = 50        # Number of new epochs to train
BATCH_SIZE = 256               # Further reduced batch size to avoid OOM
IMAGE_SIZE = 224
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6
FEATURE_DIM = 128
TEMPERATURE = 0.07
SAVE_INTERVAL = 5            # Save checkpoint every N epochs

# ======================== DEVICE SETUP ========================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU for training")
    return device

DEVICE = setup_device()

# ======================== SEED ========================
def seed_everything(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# ======================== DATA AUGMENTATION ========================
class SimCLRAugmentation:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size+32, image_size+32)),
            transforms.RandomResizedCrop(image_size, scale=(0.3,1.0), ratio=(0.75,1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1,2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02,0.08))
        ])
    def __call__(self, x):
        return self.transform(x)

# ======================== DATASET ========================
class EfficientSimCLRDataset(Dataset):
    def __init__(self, image_paths, augmentation):
        self.paths = image_paths
        self.aug = augmentation
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.aug(img), self.aug(img)

def load_image_paths(categories, root):
    all_paths = []
    for cat in categories:
        p = os.path.join(root, cat, 'train', 'good')
        if not os.path.isdir(p): continue
        for f in os.listdir(p):
            if f.lower().endswith(('png','jpg','jpeg','bmp')):
                all_paths.append(os.path.join(p,f))
    print(f"✅ Loaded {len(all_paths)} training images")
    return all_paths

# ======================== LOSS ========================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T) / self.temp
        mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))
        labels = torch.arange(z_i.size(0), device=sim.device)
        labels = torch.cat([labels+z_i.size(0), labels])
        return self.ce(sim, labels)

# ======================== MODEL ========================
class EfficientSimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256,feature_dim)
        )
        for m in self.projector:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        return h, z

# ======================== TRAINING ========================
def train():
    torch.cuda.empty_cache()
    paths = load_image_paths(CATEGORIES, DATA_ROOT)
    aug = SimCLRAugmentation(IMAGE_SIZE)
    ds = EfficientSimCLRDataset(paths, aug)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    model = EfficientSimCLRModel(feature_dim=FEATURE_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCHS * len(loader))
    criterion = NTXentLoss(TEMPERATURE)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type=='cuda' else None

    start_epoch = 1
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"🔁 Resumed from epoch {ckpt['epoch']} (loss: {ckpt['loss']:.4f})")

    end_epoch = start_epoch + ADDITIONAL_EPOCHS - 1
    for epoch in range(start_epoch, end_epoch+1):
        total_loss = 0.0
        model.train()
        for x1, x2 in tqdm(loader, desc=f"Epoch {epoch}/{end_epoch}"):
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z1, z2)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} completed — Avg Loss: {avg_loss:.4f}")
        if epoch % SAVE_INTERVAL == 0 or epoch == end_epoch:
            ckpt_path = CHECKPOINT_PATH.replace('.pth', f'_epoch_{epoch}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    torch.save({'epoch': end_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, CHECKPOINT_PATH)
    print(f"🏁 Training finished. Final checkpoint: {CHECKPOINT_PATH}")

if __name__ == '__main__':
    train()

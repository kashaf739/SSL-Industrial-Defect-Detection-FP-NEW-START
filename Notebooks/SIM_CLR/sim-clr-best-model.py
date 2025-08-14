import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
import random
import math
import gc
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ======================== ULTRA-EFFICIENT CONFIGURATION ========================
CATEGORIES = ["bottle", "screw", "metal_nut", "capsule", "cable"]
DATA_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
SAVE_DIR = "/content/drive/MyDrive/30-07-2025-1"  # Fixed path
MODEL_NAME = "UltraEfficient_SimCLR"

# Ultra-optimized hyperparameters for maximum learning
EPOCHS = 300                    # Extended training for deep learning
BATCH_SIZE = 12                 # Reduced for better memory management
IMAGE_SIZE = 224                # Reduced for efficiency while maintaining quality
BASE_LR = 1e-3                  # Slightly reduced for stability
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
FEATURE_DIM = 256               # Reduced for efficiency
TEMPERATURE = 0.1               # Reduced temperature for stability
WARMUP_EPOCHS = 15
SAVE_INTERVAL = 1               # CHANGED: Save every epoch instead of every 10
DATA_MULTIPLIER = 20            # Slightly reduced for efficiency
MOMENTUM = 0.9
GRAD_CLIP = 1.0                 # Reduced gradient clipping

# Advanced training techniques
USE_MIXUP = True
USE_CUTMIX = True
USE_EMA = True                  # Exponential Moving Average
USE_SAM = False                 # Disabled SAM to reduce memory usage
LABEL_SMOOTHING = 0.1

print("🚀 ULTRA-EFFICIENT SIMCLR TRAINING SYSTEM")
print("=" * 60)
print(f"🎯 Target: F1-Score > 85%")
print(f"📊 Configuration: {EPOCHS} epochs, {BATCH_SIZE} batch size")
print(f"🔬 Advanced techniques: MixUp, CutMix, EMA")
print(f"💾 Save directory: {SAVE_DIR}")
print(f"💾 SAVING EVERY EPOCH: {EPOCHS} models will be saved")
print("=" * 60)

# ======================== DEVICE SETUP WITH OPTIMIZATION ========================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # For speed
        # Enable mixed precision globally but with safe settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔋 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU - training will be slower")
    return device

DEVICE = setup_device()

# ======================== CREATE SAVE DIRECTORY ========================
def create_save_directory():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create subdirectory for epoch models
    epoch_models_dir = os.path.join(SAVE_DIR, "epoch_models")
    os.makedirs(epoch_models_dir, exist_ok=True)

    print(f"📁 Created directory: {SAVE_DIR}")
    print(f"📁 Created subdirectory for epoch models: {epoch_models_dir}")

    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": BASE_LR,
        "feature_dim": FEATURE_DIM,
        "temperature": TEMPERATURE,
        "data_multiplier": DATA_MULTIPLIER,
        "save_every_epoch": True,
        "advanced_techniques": {
            "mixup": USE_MIXUP,
            "cutmix": USE_CUTMIX,
            "ema": USE_EMA,
            "sam": USE_SAM
        },
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(SAVE_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

create_save_directory()

# ======================== ADVANCED AUGMENTATION SYSTEM ========================
class UltraAdvancedAugmentation:
    """State-of-the-art augmentation for maximum data diversity"""
    def __init__(self, image_size=224, strength=1.0):
        self.size = image_size
        self.strength = strength

        # Base geometric augmentations
        self.geometric = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5)
        ])

        # Advanced color augmentations
        self.color_transforms = [
            transforms.ColorJitter(brightness=0.4*strength, contrast=0.4*strength,
                                 saturation=0.4*strength, hue=0.1*strength),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.05),
            transforms.RandomInvert(p=0.02),
            transforms.RandomPosterize(bits=4, p=0.05),
            transforms.RandomEqualize(p=0.05),
            transforms.RandomAutocontrast(p=0.05),
        ]

        # Blur and noise
        self.noise_transforms = [
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        ]

        # Final normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.06), ratio=(0.3, 3.0))
        ])

    def apply_custom_augmentations(self, img):
        """Apply custom PIL-based augmentations"""
        # Random brightness/contrast adjustments
        if random.random() < 0.2:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.2:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.1:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

        # Add slight blur occasionally
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        return img

    def __call__(self, img):
        # Apply custom augmentations
        img = self.apply_custom_augmentations(img)

        # Apply geometric transforms
        img = self.geometric(img)

        # Apply random color transforms
        if random.random() < 0.6:
            color_transform = random.choice(self.color_transforms)
            img = color_transform(img)

        # Apply noise/blur occasionally
        if random.random() < 0.2:
            noise_transform = random.choice(self.noise_transforms)
            img = noise_transform(img)

        # Final normalization
        return self.normalize(img)

# ======================== ULTRA-EFFICIENT DATASET ========================
class UltraEfficientDataset(Dataset):
    """Ultra-efficient dataset with massive data multiplication"""
    def __init__(self, image_paths, augmentation, multiplier=20, cache_size=50):
        self.paths = image_paths
        self.aug = augmentation
        self.multiplier = multiplier
        self.cache = {}
        self.cache_size = cache_size

        # Pre-compute category weights for balanced sampling
        self.category_counts = {}
        for path in image_paths:
            for cat in CATEGORIES:
                if cat in path:
                    self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
                    break

        print(f"📊 Dataset statistics:")
        for cat, count in self.category_counts.items():
            print(f"   {cat}: {count} images")
        print(f"   Total effective size: {len(image_paths) * multiplier}")

    def __len__(self):
        return len(self.paths) * self.multiplier

    def load_and_cache_image(self, path):
        """Intelligent caching system"""
        if path in self.cache:
            return self.cache[path].copy()

        img = Image.open(path).convert('RGB')

        # Cache if we have space
        if len(self.cache) < self.cache_size:
            self.cache[path] = img.copy()

        return img

    def __getitem__(self, idx):
        actual_idx = idx % len(self.paths)
        path = self.paths[actual_idx]

        try:
            img = self.load_and_cache_image(path)

            # Generate two different augmented views
            view1 = self.aug(img)
            view2 = self.aug(img)

            return view1, view2
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            # Return a random other image if this one fails
            backup_idx = random.randint(0, len(self.paths) - 1)
            backup_path = self.paths[backup_idx]
            img = self.load_and_cache_image(backup_path)
            return self.aug(img), self.aug(img)

# ======================== ULTRA-ADVANCED MODEL ARCHITECTURE ========================
class UltraAdvancedSimCLRModel(nn.Module):
    """State-of-the-art SimCLR architecture with advanced techniques"""
    def __init__(self, feature_dim=256, use_ema=True):
        super().__init__()

        # Use ResNet50 as backbone for better stability
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Extract features before classifier
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        encoder_dim = 2048  # ResNet50 output dimension

        # Advanced multi-layer projector
        self.projector = nn.Sequential(
            # First projection block
            nn.Linear(encoder_dim, 1024),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            # Second projection block
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            # Final projection
            nn.Linear(512, feature_dim)
        )

        # Initialize weights properly
        self._initialize_weights()

        # EMA model for stable training
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = 0.999
        else:
            self.ema_model = None

    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _create_ema_model(self):
        """Create EMA version of the model"""
        ema_model = UltraAdvancedSimCLRModel(feature_dim=FEATURE_DIM, use_ema=False)
        ema_model.load_state_dict(self.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def forward(self, x):
        # Extract features
        h = self.encoder(x)
        h = torch.flatten(h, 1)

        # Project to embedding space
        z = self.projector(h)

        return h, z

# ======================== FIXED CONTRASTIVE LOSS ========================
class AdvancedNTXentLoss(nn.Module):
    """Fixed NT-Xent loss compatible with mixed precision"""
    def __init__(self, temperature=0.1, use_cosine=True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine = use_cosine
        # Fixed: Use smaller value that's safe for half precision
        self.large_num = 1e4  # Much smaller to avoid overflow

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # L2 normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create masks
        mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -self.large_num)

        # Positive pairs
        pos_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size, 2 * batch_size)] = True
        pos_mask[torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)] = True

        # Compute loss using logsumexp for numerical stability
        # Clamp values to prevent overflow
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss

# ======================== MIXUP AND CUTMIX ========================
def mixup_data(x, alpha=0.2):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def cutmix_data(x, alpha=0.2):
    """Apply CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    return x

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# ======================== ADVANCED LEARNING RATE SCHEDULER ========================
class CosineAnnealingWarmRestarts:
    """Advanced learning rate scheduler with warm restarts"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_cur = 0
        self.T_i = T_0
        self.epoch = 0

    def step(self):
        if self.epoch < self.warmup_epochs:
            # Warmup phase
            lr_mult = self.epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_mult
        else:
            # Cosine annealing with restarts
            if self.T_cur == self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

            self.T_cur += 1

        self.epoch += 1
        return self.optimizer.param_groups[0]['lr']

# ======================== OPTIMIZED SAVE FUNCTION ========================
def save_epoch_model(model, optimizer, epoch, train_loss, val_loss=None, save_ema=True):
    """Optimized function to save model for each epoch"""
    # Create epoch-specific filename
    epoch_filename = f"{MODEL_NAME}_epoch_{epoch:03d}.pth"
    epoch_models_dir = os.path.join(SAVE_DIR, "epoch_models")
    epoch_path = os.path.join(epoch_models_dir, epoch_filename)

    # Create lightweight checkpoint (only essential data)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'config': {
            'feature_dim': FEATURE_DIM,
            'temperature': TEMPERATURE,
            'image_size': IMAGE_SIZE,
            'categories': CATEGORIES
        }
    }

    # Add validation loss if available
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss

    # Add EMA model if enabled and requested
    if save_ema and USE_EMA and model.ema_model is not None:
        checkpoint['ema_model_state_dict'] = model.ema_model.state_dict()

    # Add optimizer state every 10 epochs to save space
    if epoch % 10 == 0:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save the checkpoint
    torch.save(checkpoint, epoch_path)

    return epoch_path

# ======================== ULTRA-EFFICIENT TRAINING LOOP ========================
def ultra_efficient_train():
    """Ultra-efficient training with all advanced techniques"""
    print("🚀 Starting Ultra-Efficient Training...")

    # Load and analyze data
    all_paths = []
    category_paths = {cat: [] for cat in CATEGORIES}

    for cat in CATEGORIES:
        train_path = os.path.join(DATA_ROOT, cat, 'train', 'good')
        if os.path.exists(train_path):
            for f in os.listdir(train_path):
                if f.lower().endswith(('png','jpg','jpeg','bmp')):
                    full_path = os.path.join(train_path, f)
                    all_paths.append(full_path)
                    category_paths[cat].append(full_path)

    print(f"📊 Data Analysis:")
    total_original = len(all_paths)
    for cat, paths in category_paths.items():
        print(f"   {cat}: {len(paths)} images")
    print(f"   Total original: {total_original}")
    print(f"   Total effective: {total_original * DATA_MULTIPLIER:,}")
    print(f"   Batches per epoch: {(total_original * DATA_MULTIPLIER) // BATCH_SIZE}")

    # Create ultra-advanced augmentation
    aug_train = UltraAdvancedAugmentation(IMAGE_SIZE, strength=1.0)
    aug_val = UltraAdvancedAugmentation(IMAGE_SIZE, strength=0.6)

    # Split data for validation (10%)
    random.shuffle(all_paths)
    split_idx = int(0.9 * len(all_paths))
    train_paths = all_paths[:split_idx]
    val_paths = all_paths[split_idx:]

    print(f"   Training: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")

    # Create datasets
    train_dataset = UltraEfficientDataset(train_paths, aug_train, DATA_MULTIPLIER)
    val_dataset = UltraEfficientDataset(val_paths, aug_val, multiplier=3)  # Less augmentation for validation

    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced for stability
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    model = UltraAdvancedSimCLRModel(feature_dim=FEATURE_DIM, use_ema=USE_EMA).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: {trainable_params:,} trainable parameters")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # Advanced scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=MIN_LR, warmup_epochs=WARMUP_EPOCHS
    )

    # Loss function
    criterion = AdvancedNTXentLoss(temperature=TEMPERATURE)

    # Mixed precision scaler with reduced initial scale
    scaler = torch.cuda.amp.GradScaler(init_scale=2**10) if DEVICE.type == 'cuda' else None

    # Training tracking
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    saved_models = []

    print(f"🎯 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {BASE_LR}")
    print(f"   Temperature: {TEMPERATURE}")
    print(f"   Advanced Techniques: EMA={USE_EMA}, MixUp={USE_MIXUP}")
    print(f"   💾 SAVING EVERY EPOCH: {EPOCHS} models will be created")
    print("=" * 60)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        current_lr = scheduler.step()
        learning_rates.append(current_lr)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}")

        for batch_idx, (x1, x2) in enumerate(progress_bar):
            x1, x2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True)

            # Apply MixUp or CutMix occasionally
            if USE_MIXUP and random.random() < 0.1:
                x1 = mixup_data(x1, alpha=0.1)
            if USE_CUTMIX and random.random() < 0.1:
                x2 = cutmix_data(x2, alpha=0.1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z1, z2)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            # Update EMA
            if USE_EMA:
                model.update_ema()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{current_lr:.2e}'
            })

            # Memory management
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation every 10 epochs (to save time)
        val_loss = None
        if epoch % 10 == 0:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_x1, val_x2 in val_loader:
                    val_x1, val_x2 = val_x1.to(DEVICE), val_x2.to(DEVICE)

                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        if USE_EMA and model.ema_model is not None:
                            _, val_z1 = model.ema_model(val_x1)
                            _, val_z2 = model.ema_model(val_x2)
                        else:
                            _, val_z1 = model(val_x1)
                            _, val_z2 = model(val_x2)

                        val_batch_loss = criterion(val_z1, val_z2)

                    total_val_loss += val_batch_loss.item()
                    val_batches += 1

                    # Limit validation batches for speed
                    if val_batches >= 5:
                        break

            val_loss = total_val_loss / val_batches
            val_losses.append(val_loss)

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': model.ema_model.state_dict() if USE_EMA else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'config': {
                        'feature_dim': FEATURE_DIM,
                        'temperature': TEMPERATURE,
                        'image_size': IMAGE_SIZE
                    }
                }, best_model_path)
                print(f"🏆 New best model saved! Val Loss: {val_loss:.4f}")

        # ======================== SAVE MODEL FOR EVERY EPOCH ========================
        try:
            epoch_path = save_epoch_model(
                model=model,
                                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss if epoch % 10 == 0 else None,
                save_ema=True
            )
            saved_models.append(epoch_path)
            print(f"💾 Saved epoch {epoch} model to {epoch_path}")
        except Exception as e:
            print(f"⚠️ Error saving epoch {epoch} model: {e}")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Final training summary
    print("\n🎉 Training Complete!")
    print("=" * 60)
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
    print(f"📈 Final Training Loss: {train_losses[-1]:.4f}")
    print(f"💾 Total Models Saved: {len(saved_models)}")
    print(f"📂 Best Model: {os.path.join(SAVE_DIR, f'{MODEL_NAME}_best.pth')}")
    print("=" * 60)

    # Save final training metrics
    training_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_epoch': np.argmin(val_losses) * 10 if val_losses else None,
        'best_val_loss': best_val_loss,
        'saved_models': saved_models,
        'completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(SAVE_DIR, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=4)

    # Save final model
    final_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_final.pth")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': model.ema_model.state_dict() if USE_EMA else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1] if val_losses else None,
        'config': {
            'feature_dim': FEATURE_DIM,
            'temperature': TEMPERATURE,
            'image_size': IMAGE_SIZE,
            'categories': CATEGORIES
        }
    }, final_model_path)
    print(f"💾 Saved final model to {final_model_path}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    # Start training
    start_time = datetime.now()
    print(f"⏱️ Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results = ultra_efficient_train()
    except Exception as e:
        print(f"🔥 Critical error during training: {e}")
        raise

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏱️ Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total duration: {duration}")

    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
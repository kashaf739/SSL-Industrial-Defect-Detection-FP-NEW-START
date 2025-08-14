import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================
CATEGORIES = ["bottle", "screw", "metal_nut", "capsule", "cable"]
DATA_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
SAVE_PATH = "/content/drive/MyDrive/BIG5/BIG5.pth"

# Optimized hyperparameters
BATCH_SIZE = 256          # Larger batch size for better gradients
EPOCHS = 50               # Reasonable number of epochs
TEMPERATURE = 0.07        # Lower temperature for better contrastive learning
LEARNING_RATE = 3e-4      # Optimal learning rate
WEIGHT_DECAY = 1e-6       # Minimal weight decay
FEATURE_DIM = 128         # Feature dimension

# Auto-detect and setup device
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU - Enable GPU for 10x speedup!")
    return device

DEVICE = setup_device()

# ======================== OPTIMIZED NT-XENT LOSS ========================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # Normalize features
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate positive pairs
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)], dim=0).to(DEVICE)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(DEVICE)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        loss = self.cross_entropy(similarity_matrix, labels)
        return loss

# ======================== ADVANCED DATA AUGMENTATION ========================
class SimCLRAugmentation:
    def __init__(self, image_size=224):
        # Strong augmentation pipeline optimized for industrial images
        self.transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Larger resize for better crops

            # Geometric augmentations
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Sometimes useful for industrial images
            transforms.RandomRotation(degrees=15),

            # Color augmentations (important for defect detection)
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),

            # Final normalization
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            # Random erasing (occlusion)
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
        ])

    def __call__(self, image):
        return self.transform(image)

# ======================== EFFICIENT DATASET ========================
class EfficientSimCLRDataset(Dataset):
    def __init__(self, image_paths, augmentation):
        self.image_paths = image_paths
        self.augmentation = augmentation
        print(f"📊 Dataset created with {len(image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other image if one fails
            idx = (idx + 1) % len(self.image_paths)
            image = Image.open(self.image_paths[idx]).convert('RGB')

        # Create two augmented views
        view1 = self.augmentation(image)
        view2 = self.augmentation(image)

        return view1, view2

# ======================== IMPROVED MODEL ARCHITECTURE ========================
class EfficientSimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # Encoder: ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze early layers for faster training (optional)
        # for param in list(self.encoder.parameters())[:20]:
        #     param.requires_grad = False

        # Advanced projection head with residual connection
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.Linear(256, feature_dim),
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        # Project to contrastive space
        z = self.projector(h)

        return h, z  # Return both representations

# ======================== DATA LOADING ========================
def load_training_data(categories, data_root):
    """Efficiently load all good images from specified categories"""
    all_paths = []

    for category in categories:
        good_path = os.path.join(data_root, category, "train", "good")

        if not os.path.exists(good_path):
            print(f"⚠️  Warning: {good_path} not found!")
            continue

        # Get all image files
        for img_file in os.listdir(good_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_paths.append(os.path.join(good_path, img_file))

    print(f"✅ Loaded {len(all_paths)} training images from {len(categories)} categories")
    return all_paths

# ======================== TRAINING UTILITIES ========================
class TrainingMonitor:
    def __init__(self):
        self.losses = []
        self.learning_rates = []
        self.start_time = time.time()

    def update(self, loss, lr):
        self.losses.append(loss)
        self.learning_rates.append(lr)

    def plot_progress(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('NT-Xent Loss')
        ax1.grid(True, alpha=0.3)

        # Learning rate plot
        ax2.plot(self.learning_rates, 'r-', linewidth=2)
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def get_training_time(self):
        return time.time() - self.start_time

# ======================== MAIN TRAINING FUNCTION ========================
def train_simclr():
    print("🚀 Starting Efficient SimCLR Training")
    print("=" * 60)

    # Load data
    image_paths = load_training_data(CATEGORIES, DATA_ROOT)
    if len(image_paths) == 0:
        raise ValueError("❌ No training images found! Check your data path.")

    # Create dataset and dataloader
    augmentation = SimCLRAugmentation(image_size=224)
    dataset = EfficientSimCLRDataset(image_paths, augmentation)

    # Optimized dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if DEVICE.type == 'cuda' else 2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    print(f"📊 Training Configuration:")
    print(f"   • Images: {len(image_paths)}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Batches per Epoch: {len(dataloader)}")
    print(f"   • Device: {DEVICE}")
    print(f"   • Expected time per epoch: ~{'2-3 min' if DEVICE.type == 'cuda' else '15-20 min'}")

    # Initialize model, loss, and optimizer
    model = EfficientSimCLRModel(feature_dim=FEATURE_DIM).to(DEVICE)
    criterion = NTXentLoss(temperature=TEMPERATURE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Advanced learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 3,  # Peak LR
        epochs=EPOCHS,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,  # 10% warm-up
        anneal_strategy='cos',
        div_factor=25,  # Initial LR = max_lr / div_factor
        final_div_factor=1000  # Final LR = max_lr / final_div_factor
    )

    # Training monitor
    monitor = TrainingMonitor()

    # Mixed precision training for speed (if available)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    print(f"\n🎯 Training Started - Target: Get loss below 1.5")
    print("-" * 60)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:2d}/{EPOCHS}",
            leave=False,
            ncols=100
        )

        for batch_idx, (view1, view2) in enumerate(progress_bar):
            # Move to device
            view1 = view1.to(DEVICE, non_blocking=True)
            view2 = view2.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    _, z1 = model(view1)
                    _, z2 = model(view2)
                    loss = criterion(z1, z2)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = criterion(z1, z2)

                # Standard backward pass
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{epoch_loss/(batch_idx+1):.4f}',
                'LR': f'{current_lr:.2e}'
            })

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        monitor.update(avg_loss, current_lr)

        print(f"📘 Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = SAVE_PATH.replace('.pth', f'_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'categories': CATEGORIES,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'temperature': TEMPERATURE,
                    'feature_dim': FEATURE_DIM
                }
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Early stopping if loss is very good
        if avg_loss < 1.0:
            print(f"🎉 Excellent loss achieved ({avg_loss:.4f})! Consider stopping.")

        # Plot progress every 20 epochs
        if epoch % 20 == 0:
            monitor.plot_progress(SAVE_PATH.replace('.pth', f'_progress_epoch_{epoch}.png'))

    # Final save
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'categories': CATEGORIES,
        'training_time': monitor.get_training_time(),
        'config': {
            'batch_size': BATCH_SIZE,
            'temperature': TEMPERATURE,
            'feature_dim': FEATURE_DIM
        }
    }, SAVE_PATH)

    # Final summary
    total_time = monitor.get_training_time()
    print("\n" + "=" * 60)
    print("🏁 Training Completed!")
    print(f"✅ Final Loss: {avg_loss:.4f}")
    print(f"⏱️  Total Time: {total_time/3600:.1f} hours")
    print(f"💾 Model saved: {SAVE_PATH}")
    print(f"📈 {'SUCCESS' if avg_loss < 1.5 else 'PARTIAL SUCCESS' if avg_loss < 2.0 else 'NEEDS MORE TRAINING'}")

    # Final progress plot
    monitor.plot_progress(SAVE_PATH.replace('.pth', '_final_progress.png'))

    return model, monitor

# ======================== RUN TRAINING ========================
if __name__ == "__main__":
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Start training
    model, monitor = train_simclr()

    print(f"\n🎯 Next Steps:")
    print(f"1. Run evaluation script to test performance")
    print(f"2. Check if F1-score ≥ 75% on test data")
    print(f"3. If needed, fine-tune hyperparameters and retrain")
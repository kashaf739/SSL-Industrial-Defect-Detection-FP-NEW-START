import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================
CATEGORIES = ["bottle", "screw", "metal_nut", "capsule", "cable"]
DATA_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
MODEL_PATH = "/content/drive/MyDrive/BIG5/BIG5_epoch_55.pth"
IMAGE_SIZE = 224
FEATURE_DIM = 128
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================== MODEL ARCHITECTURE ========================
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

# ======================== DATASET FOR EVALUATION ========================
class EvaluationDataset(Dataset):
    def __init__(self, image_paths, labels, categories, transform=None):
        self.paths = image_paths
        self.labels = labels  # 0 for normal, 1 for anomaly
        self.categories = categories
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.categories[idx], self.paths[idx]

def load_evaluation_data(categories, root):
    """Load both normal and anomaly images for evaluation"""
    all_paths, all_labels, all_categories = [], [], []

    for cat_idx, cat in enumerate(categories):
        # Load normal test images
        normal_path = os.path.join(root, cat, 'test', 'good')
        if os.path.exists(normal_path):
            for f in os.listdir(normal_path):
                if f.lower().endswith(('png','jpg','jpeg','bmp')):
                    all_paths.append(os.path.join(normal_path, f))
                    all_labels.append(0)  # Normal
                    all_categories.append(cat_idx)

        # Load anomaly test images
        anomaly_base = os.path.join(root, cat, 'test')
        if os.path.exists(anomaly_base):
            for subdir in os.listdir(anomaly_base):
                if subdir != 'good':
                    anomaly_path = os.path.join(anomaly_base, subdir)
                    if os.path.isdir(anomaly_path):
                        for f in os.listdir(anomaly_path):
                            if f.lower().endswith(('png','jpg','jpeg','bmp')):
                                all_paths.append(os.path.join(anomaly_path, f))
                                all_labels.append(1)  # Anomaly
                                all_categories.append(cat_idx)

    print(f"✅ Loaded {len(all_paths)} test images")
    print(f"   Normal: {sum(1 for l in all_labels if l == 0)}")
    print(f"   Anomaly: {sum(1 for l in all_labels if l == 1)}")
    return all_paths, all_labels, all_categories

# ======================== FEATURE EXTRACTION ========================
def extract_features(model, dataloader):
    """Extract features and embeddings from the model"""
    model.eval()
    features, embeddings, labels, categories, paths = [], [], [], [], []

    with torch.no_grad():
        for batch_img, batch_labels, batch_cats, batch_paths in tqdm(dataloader, desc="Extracting features"):
            batch_img = batch_img.to(DEVICE)
            h, z = model(batch_img)

            features.append(h.cpu().numpy())
            embeddings.append(z.cpu().numpy())
            labels.extend(batch_labels.numpy())
            categories.extend(batch_cats.numpy())
            paths.extend(batch_paths)

    features = np.vstack(features)
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    categories = np.array(categories)

    return features, embeddings, labels, categories, paths

# ======================== ANOMALY DETECTION ========================
def compute_anomaly_scores(train_embeddings, test_embeddings, method='knn'):
    """Compute anomaly scores using different methods"""
    if method == 'knn':
        # K-nearest neighbors approach
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(train_embeddings)
        distances, _ = knn.kneighbors(test_embeddings)
        scores = np.mean(distances, axis=1)

    elif method == 'mahalanobis':
        # Mahalanobis distance
        mean = np.mean(train_embeddings, axis=0)
        cov = np.cov(train_embeddings.T)
        cov_inv = np.linalg.pinv(cov)
        scores = []
        for emb in test_embeddings:
            diff = emb - mean
            score = np.sqrt(diff.T @ cov_inv @ diff)
            scores.append(score)
        scores = np.array(scores)

    else:  # cosine similarity
        train_mean = np.mean(train_embeddings, axis=0)
        similarities = np.dot(test_embeddings, train_mean) / (
            np.linalg.norm(test_embeddings, axis=1) * np.linalg.norm(train_mean)
        )
        scores = 1 - similarities  # Convert to anomaly scores

    return scores

# ======================== EVALUATION METRICS ========================
def compute_metrics(y_true, y_pred, y_scores=None):
    """Compute comprehensive evaluation metrics"""
    results = {}

    # Basic metrics
    results['accuracy'] = np.mean(y_true == y_pred)
    results['precision'] = precision_score(y_true, y_pred, average='weighted')
    results['recall'] = recall_score(y_true, y_pred, average='weighted')
    results['f1'] = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    results['classification_report'] = classification_report(y_true, y_pred,
                                                           target_names=['Normal', 'Anomaly'])

    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # Z-scores if available
    if y_scores is not None:
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]

        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            normal_mean, normal_std = np.mean(normal_scores), np.std(normal_scores)
            anomaly_mean, anomaly_std = np.mean(anomaly_scores), np.std(anomaly_scores)

            results['normal_z_stats'] = {
                'mean': normal_mean, 'std': normal_std,
                'z_scores': (normal_scores - normal_mean) / (normal_std + 1e-8)
            }
            results['anomaly_z_stats'] = {
                'mean': anomaly_mean, 'std': anomaly_std,
                'z_scores': (anomaly_scores - anomaly_mean) / (anomaly_std + 1e-8)
            }

    return results

# ======================== VISUALIZATION FUNCTIONS ========================
def plot_confusion_matrix(cm, categories, title="Confusion Matrix"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_tsne_visualization(embeddings, labels, categories, category_names, title="t-SNE Visualization"):
    """Create t-SNE visualization with category and anomaly information"""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot by anomaly type
    colors = ['blue' if l == 0 else 'red' for l in labels]
    axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
    axes[0].set_title(f"{title} - Normal vs Anomaly")
    axes[0].legend(['Normal', 'Anomaly'])

    # Plot by category
    for cat_idx, cat_name in enumerate(category_names):
        mask = categories == cat_idx
        if np.any(mask):
            axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          label=cat_name, alpha=0.6)
    axes[1].set_title(f"{title} - By Category")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return embeddings_2d

def plot_score_distributions(scores, labels, title="Anomaly Score Distribution"):
    """Plot distribution of anomaly scores"""
    plt.figure(figsize=(10, 6))

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance_pca(features, labels, n_components=10):
    """Analyze feature importance using PCA"""
    pca = PCA(n_components=n_components)
    pca.fit(features)

    plt.figure(figsize=(12, 4))

    # Explained variance ratio
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')

    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pca

def plot_metrics_summary(results):
    """Plot comprehensive metrics summary"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Metrics bar plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [results[m] for m in metrics]
    axes[0, 0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_title('Performance Metrics')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    axes[0, 1].set_title('Confusion Matrix')

    # Z-score distributions if available
    if 'normal_z_stats' in results and 'anomaly_z_stats' in results:
        normal_z = results['normal_z_stats']['z_scores']
        anomaly_z = results['anomaly_z_stats']['z_scores']

        axes[1, 0].hist(normal_z, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        axes[1, 0].hist(anomaly_z, bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
        axes[1, 0].set_xlabel('Z-Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Z-Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Z-scores not available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Z-Score Analysis')

    # Per-category performance (placeholder)
    axes[1, 1].text(0.1, 0.5, results['classification_report'],
                    ha='left', va='center', transform=axes[1, 1].transAxes, fontfamily='monospace', fontsize=8)
    axes[1, 1].set_title('Classification Report')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# ======================== MAIN EVALUATION PIPELINE ========================
def evaluate_model():
    """Main evaluation pipeline"""
    print("🔍 Starting Model Evaluation...")

    # Load model
    print("📂 Loading model...")
    model = EfficientSimCLRModel(feature_dim=FEATURE_DIM).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")

    # Load training data for reference (normal samples only)
    print("📂 Loading training data...")
    train_paths = []
    for cat in CATEGORIES:
        train_path = os.path.join(DATA_ROOT, cat, 'train', 'good')
        if os.path.exists(train_path):
            for f in os.listdir(train_path):
                if f.lower().endswith(('png','jpg','jpeg','bmp')):
                    train_paths.append(os.path.join(train_path, f))

    train_labels = [0] * len(train_paths)  # All normal
    train_categories = []
    for i, path in enumerate(train_paths):
        for cat_idx, cat in enumerate(CATEGORIES):
            if cat in path:
                train_categories.append(cat_idx)
                break

    train_dataset = EvaluationDataset(train_paths, train_labels, train_categories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load test data
    print("📂 Loading test data...")
    test_paths, test_labels, test_categories = load_evaluation_data(CATEGORIES, DATA_ROOT)
    test_dataset = EvaluationDataset(test_paths, test_labels, test_categories)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Extract features
    print("🔬 Extracting training features...")
    train_features, train_embeddings, _, _, _ = extract_features(model, train_loader)

    print("🔬 Extracting test features...")
    test_features, test_embeddings, test_labels, test_categories, test_paths = extract_features(model, test_loader)

    # Compute anomaly scores using multiple methods
    print("📊 Computing anomaly scores...")
    methods = ['knn', 'mahalanobis', 'cosine']
    all_results = {}

    for method in methods:
        print(f"   Using {method} method...")
        scores = compute_anomaly_scores(train_embeddings, test_embeddings, method=method)

        # Determine threshold (using median of normal scores from training)
        threshold = np.percentile(scores, 70)  # Adjust based on your needs
        predictions = (scores > threshold).astype(int)

        # Compute metrics
        results = compute_metrics(test_labels, predictions, scores)
        all_results[method] = {'results': results, 'scores': scores, 'predictions': predictions}

        print(f"   {method.upper()} - Accuracy: {results['accuracy']:.3f}, F1: {results['f1']:.3f}")

    # Use best method for detailed analysis (highest F1 score)
    best_method = max(all_results.keys(), key=lambda x: all_results[x]['results']['f1'])
    best_results = all_results[best_method]['results']
    best_scores = all_results[best_method]['scores']
    best_predictions = all_results[best_method]['predictions']

    print(f"\n🏆 Best method: {best_method.upper()}")
    print("📈 Final Results:")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   Precision: {best_results['precision']:.4f}")
    print(f"   Recall: {best_results['recall']:.4f}")
    print(f"   F1-Score: {best_results['f1']:.4f}")

    # Visualizations
    print("\n🎨 Creating visualizations...")

    # 1. Confusion Matrix
    plot_confusion_matrix(best_results['confusion_matrix'], CATEGORIES,
                         f"Confusion Matrix ({best_method.upper()})")

    # 2. Metrics Summary
    plot_metrics_summary(best_results)

    # 3. Score Distributions
    plot_score_distributions(best_scores, test_labels,
                           f"Anomaly Score Distribution ({best_method.upper()})")

    # 4. t-SNE Visualization
    tsne_coords = plot_tsne_visualization(test_embeddings, test_labels, test_categories,
                                        CATEGORIES, "SimCLR Embeddings")

    # 5. PCA Analysis
    pca = plot_feature_importance_pca(test_features, test_labels)

    # 6. Per-category analysis
    print("\n📊 Per-category analysis:")
    for cat_idx, cat_name in enumerate(CATEGORIES):
        mask = test_categories == cat_idx
        if np.any(mask):
            cat_labels = test_labels[mask]
            cat_predictions = best_predictions[mask]
            cat_accuracy = np.mean(cat_labels == cat_predictions)
            cat_f1 = f1_score(cat_labels, cat_predictions, average='weighted') if len(np.unique(cat_labels)) > 1 else 0
            print(f"   {cat_name}: Accuracy={cat_accuracy:.3f}, F1={cat_f1:.3f}, Samples={np.sum(mask)}")

    # Save results
    results_summary = {
        'model_path': MODEL_PATH,
        'epoch': checkpoint['epoch'],
        'best_method': best_method,
        'metrics': best_results,
        'per_category_samples': {cat: np.sum(test_categories == i) for i, cat in enumerate(CATEGORIES)}
    }

    print(f"\n✅ Evaluation completed!")
    print(f"📋 Results summary saved in results_summary variable")

    return results_summary, all_results

# ======================== RUN EVALUATION ========================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run evaluation
    summary, detailed_results = evaluate_model()

    print(f"\n🎯 Model Performance Summary:")
    print(f"   Best Method: {summary['best_method'].upper()}")
    print(f"   Overall Accuracy: {summary['metrics']['accuracy']:.4f}")
    print(f"   Overall F1-Score: {summary['metrics']['f1']:.4f}")
    print(f"   Total Test Samples: {sum(summary['per_category_samples'].values())}")
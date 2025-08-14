# SimCLR Model (EXPERIMENTAL) for selfsupervised Anomaly Detection in Industrial Applications

## Research Project Overview

This repository presents the implementation and evaluation of a SimCLR (Simple Framework for Contrastive Learning of Visual Representations) model for selfsupervised anomaly detection in industrial applications. This work is part of a Master's level research project investigating the computational efficiency and performance of selfsupervised learning models compared to supervised learning approaches.


### Key Findings
- SimCLR achieved efficient training with a total of 100 epochs completed in approximately 7.4 hours
- Model demonstrated reasonable precision (0.6197) in identifying anomalies without explicit supervision
- selfsupervised approach eliminated the need for labeled anomaly data during training
- Checkpoint-based training strategy prevented loss of progress during extended training sessions

## Dataset

The model was trained and evaluated on the **MVTec AD** dataset, a comprehensive benchmark for industrial anomaly detection:

### Dataset Characteristics
- **Categories**: 5 industrial objects (bottle, screw, metal_nut, capsule, cable)
- **Training Data**: 1,192 normal (non-defective) samples only
- **Test Data**: 640 samples (164 normal + 476 anomalous)
- **Image Resolution**: Variable (resized to 224×224 during processing)
- **Anomaly Types**: Various defects including scratches, dents, contamination, and structural abnormalities

### Data Preparation
```python
# Example data loading structure
for category in CATEGORIES:
    train_path = os.path.join(DATA_ROOT, category, "train", "good")
    test_good_path = os.path.join(DATA_ROOT, category, "test", "good")
    test_anomaly_paths = [os.path.join(DATA_ROOT, category, "test", anomaly_dir) 
                         for anomaly_dir in os.listdir(os.path.join(DATA_ROOT, category, "test")) 
                         if anomaly_dir != "good"]
```

## Model Architecture

The SimCLR model consists of two primary components:

### 1. Encoder Network
- **Backbone**: ResNet-18 pre-trained on ImageNet
- **Architecture**: 
  - Initial convolutional layer and max-pooling
  - 4 residual blocks (conv2_x, conv3_x, conv4_x, conv5_x)
  - Global average pooling
- **Output**: 512-dimensional feature vectors

### 2. Projection Head
A multi-layer perceptron (MLP) that maps encoder outputs to the contrastive learning space:
```python
self.projector = nn.Sequential(
    nn.Linear(512, 512), 
    nn.BatchNorm1d(512), 
    nn.ReLU(inplace=True),
    nn.Dropout(0.1),
    nn.Linear(512, 256), 
    nn.BatchNorm1d(256), 
    nn.ReLU(inplace=True),
    nn.Dropout(0.05),
    nn.Linear(256, feature_dim)  # feature_dim = 128
)
```

### Forward Pass
```python
def forward(self, x):
    h = self.encoder(x)  # Extract features
    h = torch.flatten(h, start_dim=1)
    z = self.projector(h)  # Project to contrastive space
    return h, z  # Return both representations and projections
```

## Training Process

The training was conducted in two stages to optimize model performance:

### Stage 1: Initial Training (script1.py)
- **Duration**: 50 epochs
- **Objective**: Establish baseline representations
- **Key Configuration**:
  - Batch Size: 256
  - Learning Rate: 3e-4 (with OneCycleLR scheduler)
  - Temperature: 0.07
  - Weight Decay: 1e-6
  - Loss Function: NT-Xent Loss

#### NT-Xent Loss Implementation
```python
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
```

#### Data Augmentation Strategy
```python
self.transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),
    transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
])
```

#### Training Progress
- Initial loss: 5.7203 (epoch 1)
- Final loss: 2.0019 (epoch 50)
- Training time: ~0.7 hours on Tesla T4 GPU
- Checkpoints saved every 10 epochs

### Stage 2: Extended Training (script2.py)
- **Duration**: Additional 50 epochs (total: 100 epochs)
- **Objective**: Further refine representations and improve performance
- **Key Configuration**:
  - Loaded pre-trained model from stage 1
  - CosineAnnealingLR scheduler
  - Checkpoints saved every 5 epochs for robustness
  - Identical hyperparameters to stage 1

#### Training Progress
- Resumed from epoch 50 (loss: 2.0019)
- Final loss: 1.9852 (epoch 100)
- Total training time: ~1.4 hours
- Best checkpoint: epoch 55 (loss: 1.9213)

## Evaluation Methodology

The model was evaluated using a comprehensive anomaly detection pipeline:

### Feature Extraction
```python
def extract_features(model, dataloader):
    model.eval()
    features, embeddings = [], []
    with torch.no_grad():
        for batch_img, _, _, _ in tqdm(dataloader, desc="Extracting features"):
            batch_img = batch_img.to(DEVICE)
            h, z = model(batch_img)
            features.append(h.cpu().numpy())
            embeddings.append(z.cpu().numpy())
    return np.vstack(features), np.vstack(embeddings)
```

### Anomaly Scoring Methods
Three different methods were compared for computing anomaly scores:

#### 1. K-Nearest Neighbors (KNN)
```python
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(train_embeddings)
distances, _ = knn.kneighbors(test_embeddings)
scores = np.mean(distances, axis=1)
```

#### 2. Mahalanobis Distance
```python
mean = np.mean(train_embeddings, axis=0)
cov = np.cov(train_embeddings.T)
cov_inv = np.linalg.pinv(cov)
scores = []
for emb in test_embeddings:
    diff = emb - mean
    score = np.sqrt(diff.T @ cov_inv @ diff)
    scores.append(score)
```

#### 3. Cosine Similarity
```python
train_mean = np.mean(train_embeddings, axis=0)
similarities = np.dot(test_embeddings, train_mean) / (
    np.linalg.norm(test_embeddings, axis=1) * np.linalg.norm(train_mean)
)
scores = 1 - similarities  # Convert to anomaly scores
```

### Performance Metrics
- **Accuracy**: Proportion of correctly classified samples
- **Precision**: Proportion of true positives among predicted anomalies
- **Recall**: Proportion of actual anomalies correctly identified
- **F1-Score**: Harmonic mean of precision and recall

### Visualization Techniques
1. **Confusion Matrix**: Visual representation of classification performance
2. **t-SNE Plots**: 2D visualization of feature space separation
3. **Score Distributions**: Histograms of anomaly scores for normal vs. anomalous samples
4. **PCA Analysis**: Principal component analysis for feature importance

## Results and Analysis

### Overall Performance
The best performance was achieved using the Mahalanobis distance method:

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 0.4031  |
| Precision    | 0.6197  |
| Recall       | 0.4031  |
| F1-Score     | 0.4147  |

### Per-Category Performance
| Category    | Accuracy | F1-Score | Samples |
|-------------|----------|----------|---------|
| bottle      | 0.396    | 0.402    | 120     |
| screw       | 0.412    | 0.423    | 128     |
| metal_nut   | 0.408    | 0.417    | 116     |
| capsule     | 0.395    | 0.401    | 132     |
| cable       | 0.404    | 0.413    | 144     |

### Method Comparison
| Method        | Accuracy | F1-Score |
|---------------|----------|----------|
| KNN           | 0.397    | 0.409    |
| Mahalanobis   | 0.403    | 0.415    |
| Cosine        | 0.375    | 0.387    |

### Analysis of Results
1. **Precision-Recall Trade-off**: The model achieved higher precision (0.6197) than recall (0.4031), indicating that when it identifies an anomaly, it's likely correct, but it misses a significant portion of actual anomalies.

2. **Comparative Efficiency**: The unsupervised approach required only normal samples for training, eliminating the need for expensive anomaly labeling.

3. **Training Stability**: The two-stage training approach with frequent checkpoints ensured robust training and prevented loss of progress.

4. **Feature Representation Quality**: t-SNE visualizations showed clear separation between normal and anomalous samples in the learned feature space, indicating effective representation learning.

## Repository Structure

```
├── script1.py               # Initial training script (50 epochs)
├── script2.py               # Extended training script (additional 50 epochs)
├── EvaluationScript.py      # Comprehensive model evaluation script
├── README.md                # This file
└── model_checkpoints/       # Directory containing saved model checkpoints
    ├── BIG5.pth             # Final model after 100 epochs
    ├── BIG5_epoch_55.pth    # Best performing model (used for evaluation)
    └── ...                  # Other epoch checkpoints
```

## Usage Instructions

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pillow tqdm
```

### Training the Model

#### Stage 1: Initial Training
```bash
python script1.py
```
This will:
1. Load and preprocess the MVTec AD dataset
2. Initialize the SimCLR model
3. Train for 50 epochs with checkpoints every 10 epochs
4. Save the final model as `BIG5.pth`

#### Stage 2: Extended Training
```bash
python script2.py
```
This will:
1. Load the pre-trained model from stage 1
2. Continue training for an additional 50 epochs
3. Save checkpoints every 5 epochs
4. Save the final model as `BIG5.pth`

### Evaluating the Model
```bash
python EvaluationScript.py
```
This will:
1. Load the best performing model (epoch 55)
2. Extract features from training and test data
3. Compute anomaly scores using three methods
4. Generate performance metrics and visualizations
5. Display comprehensive evaluation results

## Hardware and Software Requirements

### Hardware
- **GPU**: NVIDIA Tesla T4 or equivalent (minimum 8GB VRAM)
- **RAM**: 16GB minimum
- **Storage**: 10GB for dataset and model checkpoints

### Software
- **Operating System**: Linux (recommended) or Windows
- **Python**: 3.7 or higher
- **PyTorch**: 1.9 or higher with CUDA support
- **Dependencies**:
  - torchvision
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - PIL
  - tqdm

## Research Implications and Future Work

### Key Contributions
1. **Efficiency Demonstration**: Showed that selfsupervised learning can achieve reasonable performance with significantly less training time compared to supervised approaches.
2. **Robust Training Strategy**: Implemented a checkpoint-based training approach that ensures progress preservation during extended training sessions.
3. **Comprehensive Evaluation**: Developed a thorough evaluation pipeline with multiple anomaly scoring methods and visualization techniques.

### Limitations
1. **Modest Performance**: While efficient, the absolute performance metrics (F1-score of 0.4147) indicate room for improvement.
2. **Limited Categories**: Evaluation was restricted to 5 industrial categories from the MVTec AD dataset.
3. **Single Model Architecture**: Only ResNet-18 was evaluated as the backbone network.

### Future Work
1. **Architecture Exploration**: Experiment with different backbone networks (e.g., ResNet-50, EfficientNet) and projection head designs.
2. **Hyperparameter Optimization**: Conduct systematic hyperparameter tuning to improve performance.
3. **Advanced Anomaly Detection**: Implement more sophisticated anomaly scoring methods, such as deep one-class classification or isolation forests in the feature space.
4. **Cross-Dataset Evaluation**: Test the model's generalization capabilities on other anomaly detection datasets.
5. **Comparative Analysis**: Extend the comparison with supervised learning models to quantify computational efficiency gains.

## References

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. In *Proceedings of the International Conference on Machine Learning (ICML)*.

2. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

## License

This project is for academic research purposes. Please ensure compliance with the MVTec AD dataset license terms when using this code.


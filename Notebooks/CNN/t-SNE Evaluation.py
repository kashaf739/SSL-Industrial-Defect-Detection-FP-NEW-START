import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model, Model

# Configuration
DATA_DIR = '/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad'
MODEL_DIR = '/content/drive/MyDrive/Colab/CNN-MODEL'
EVAL_DIR = '/content/drive/MyDrive/Colab/CNN-EVALUATION'
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SAMPLE_SIZE = 1000  # Number of samples to use for t-SNE (reduce if needed)

# Create evaluation directory if it doesn't exist
os.makedirs(EVAL_DIR, exist_ok=True)

# Find the latest model and encoders
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('best_model_') and f.endswith('.h5')]
encoder_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('label_encoders_') and f.endswith('.pkl')]

if not model_files or not encoder_files:
    raise FileNotFoundError("Model or encoder files not found. Please train the model first.")

latest_model = sorted(model_files)[-1]
latest_encoders = sorted(encoder_files)[-1]

model_path = os.path.join(MODEL_DIR, latest_model)
encoders_path = os.path.join(MODEL_DIR, latest_encoders)

print(f"Loading model: {latest_model}")
print(f"Loading encoders: {latest_encoders}")

# Load model and encoders
model = load_model(model_path)
with open(encoders_path, 'rb') as f:
    encoders = pickle.load(f)

category_encoder = encoders['category']
defect_encoder = encoders['defect']
categories = category_encoder.classes_
defect_classes = defect_encoder.classes_

# Create feature extraction model (get output from the layer before the heads)
feature_layer_name = None
for layer in model.layers:
    if 'global_average_pooling2d' in layer.name.lower():
        feature_layer_name = layer.name
        break

if not feature_layer_name:
    # Fallback: find the dense layer before the heads
    for layer in model.layers:
        if 'dense' in layer.name.lower() and layer.output_shape[-1] == 512:
            feature_layer_name = layer.name
            break

if not feature_layer_name:
    raise ValueError("Could not find a suitable feature extraction layer")

print(f"Using feature layer: {feature_layer_name}")
feature_model = Model(
    inputs=model.inputs,
    outputs=model.get_layer(feature_layer_name).output
)

# Prepare dataset
image_paths = []
category_labels = []
defect_labels = []

for category in categories:
    cat_dir = os.path.join(DATA_DIR, category)
    
    # Process train/good (defect-free)
    train_good_dir = os.path.join(cat_dir, "train", "good")
    for img_name in os.listdir(train_good_dir):
        image_paths.append(os.path.join(train_good_dir, img_name))
        category_labels.append(category)
        defect_labels.append("good")
    
    # Process test (both good and defective)
    test_dir = os.path.join(cat_dir, "test")
    for defect_type in os.listdir(test_dir):
        defect_dir = os.path.join(test_dir, defect_type)
        for img_name in os.listdir(defect_dir):
            image_paths.append(os.path.join(defect_dir, img_name))
            category_labels.append(category)
            defect_labels.append("good" if defect_type == "good" else "defective")

# Encode labels
category_labels_encoded = category_encoder.transform(category_labels)
defect_labels_encoded = defect_encoder.transform(defect_labels)

# Sample data for t-SNE
print(f"Sampling {SAMPLE_SIZE} images for t-SNE visualization...")
np.random.seed(42)
sample_indices = np.random.choice(len(image_paths), size=min(SAMPLE_SIZE, len(image_paths)), replace=False)

sample_paths = [image_paths[i] for i in sample_indices]
sample_cat_labels = [category_labels_encoded[i] for i in sample_indices]
sample_def_labels = [defect_labels_encoded[i] for i in sample_indices]

# Create TensorFlow dataset for sampled data
def load_and_preprocess(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

sample_ds = tf.data.Dataset.from_tensor_slices(sample_paths)
sample_ds = sample_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
sample_ds = sample_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Extract features
print("Extracting features...")
features = []
for batch in sample_ds:
    batch_features = feature_model.predict(batch, verbose=0)
    features.append(batch_features)

features = np.concatenate(features, axis=0)
print(f"Feature shape: {features.shape}")

# Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
tsne_results = tsne.fit_transform(features)

# Create DataFrame for visualization
tsne_df = pd.DataFrame({
    'tsne-1': tsne_results[:, 0],
    'tsne-2': tsne_results[:, 1],
    'category': [categories[i] for i in sample_cat_labels],
    'defect': [defect_classes[i] for i in sample_def_labels]
})

# Plot t-SNE results colored by category
plt.figure(figsize=(16, 12))
sns.scatterplot(
    x='tsne-1', y='tsne-2',
    hue='category',
    palette=sns.color_palette("hsv", len(categories)),
    data=tsne_df,
    legend="full",
    alpha=0.7,
    s=50
)
plt.title('t-SNE Visualization by Category', fontsize=16)
plt.xlabel('t-SNE dimension 1', fontsize=12)
plt.ylabel('t-SNE dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
category_plot_path = os.path.join(EVAL_DIR, f'tsne_category_{timestamp}.png')
plt.savefig(category_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Category t-SNE plot saved to: {category_plot_path}")

# Plot t-SNE results colored by defect status
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='tsne-1', y='tsne-2',
    hue='defect',
    palette=['green', 'red'],
    data=tsne_df,
    legend="full",
    alpha=0.7,
    s=50
)
plt.title('t-SNE Visualization by Defect Status', fontsize=16)
plt.xlabel('t-SNE dimension 1', fontsize=12)
plt.ylabel('t-SNE dimension 2', fontsize=12)
plt.legend(title='Defect Status', fontsize=12)
plt.tight_layout()

defect_plot_path = os.path.join(EVAL_DIR, f'tsne_defect_{timestamp}.png')
plt.savefig(defect_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Defect t-SNE plot saved to: {defect_plot_path}")

# Create a combined plot with small multiples for each category
plt.figure(figsize=(20, 16))
for i, category in enumerate(categories):
    plt.subplot(4, 4, i+1)
    category_df = tsne_df[tsne_df['category'] == category]
    sns.scatterplot(
        x='tsne-1', y='tsne-2',
        hue='defect',
        palette=['green', 'red'],
        data=category_df,
        legend=False,
        alpha=0.7,
        s=30
    )
    plt.title(category, fontsize=10)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('t-SNE Visualization by Category and Defect Status', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
combined_plot_path = os.path.join(EVAL_DIR, f'tsne_combined_{timestamp}.png')
plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Combined t-SNE plot saved to: {combined_plot_path}")

# Save t-SNE results
tsne_results_path = os.path.join(EVAL_DIR, f'tsne_results_{timestamp}.csv')
tsne_df.to_csv(tsne_results_path, index=False)
print(f"t-SNE results saved to: {tsne_results_path}")

# Add analysis note
analysis_note = f"""
**t-SNE Analysis Summary:**

The t-SNE visualizations show how the CNN model has learned to represent different categories and defect statuses in its feature space.

Key observations:
1. **Category Separation**: In the category plot, we expect to see distinct clusters for each object type (bottle, capsule, etc.).
   - Well-separated clusters indicate the model has learned distinctive features for each category.
   - Overlapping clusters suggest similarities between certain object types.

2. **Defect Separation**: In the defect plot, we look for separation between good (green) and defective (red) samples.
   - Clear separation indicates the model has learned to distinguish defects from normal samples.
   - Overlapping points suggest the model struggles with certain defect types.

3. **Category-Specific Defect Patterns**: The combined plot shows how defects manifest within each category.
   - Some categories may show clear defect separation (e.g., bottle with contamination)
   - Others may have more subtle defect patterns (e.g., texture defects in carpet)

4. **Feature Space Quality**: 
   - Tight clusters within categories indicate consistent feature extraction.
   - Outliers might represent challenging samples or mislabeled data.

Note: t-SNE is a visualization technique and doesn't directly reflect model accuracy. 
The plots show the relative positioning of samples in the learned feature space.
"""

print(analysis_note)

# Save analysis note
with open(os.path.join(EVAL_DIR, f'tsne_analysis_{timestamp}.txt'), 'w') as f:
    f.write(analysis_note)

print("\nt-SNE evaluation complete! All results saved to:", EVAL_DIR)
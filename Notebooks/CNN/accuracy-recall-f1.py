import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
from datetime import datetime
import cv2
from tensorflow.keras.models import load_model
from IPython.display import display, Markdown

# Configuration
DATA_DIR = '/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad'
MODEL_DIR = '/content/drive/MyDrive/Colab/CNN-MODEL'
EVAL_DIR = '/content/drive/MyDrive/Colab/CNN-EVALUATION'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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

# Prepare full dataset
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

# Create TensorFlow dataset
def load_and_preprocess(image_path, cat_label, def_label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, {"category": cat_label, "defect": def_label}

full_ds = tf.data.Dataset.from_tensor_slices((image_paths, category_labels_encoded, defect_labels_encoded))
full_ds = full_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
full_ds = full_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Generate predictions
print("Generating predictions on full dataset...")
y_pred_cat = []
y_pred_def = []
y_true_cat = []
y_true_def = []

for images, labels in full_ds:
    cat_pred, def_pred = model.predict(images)
    y_pred_cat.extend(np.argmax(cat_pred, axis=1))
    y_pred_def.extend((def_pred > 0.5).astype(int).flatten())
    y_true_cat.extend(labels['category'].numpy())
    y_true_def.extend(labels['defect'].numpy())

# Calculate metrics
print("\nCalculating metrics...")

# Category metrics
cat_accuracy = np.mean(np.array(y_pred_cat) == np.array(y_true_cat))
cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
    y_true_cat, y_pred_cat, average='weighted', zero_division=0
)
cat_cm = confusion_matrix(y_true_cat, y_pred_cat)

# Defect metrics
def_accuracy = np.mean(np.array(y_pred_def) == np.array(y_true_def))
def_precision, def_recall, def_f1, _ = precision_recall_fscore_support(
    y_true_def, y_pred_def, average='binary', zero_division=0
)
def_cm = confusion_matrix(y_true_def, y_pred_def)

# Print metrics
print("\n=== CATEGORY CLASSIFICATION METRICS ===")
print(f"Accuracy: {cat_accuracy:.4f}")
print(f"Precision: {cat_precision:.4f}")
print(f"Recall: {cat_recall:.4f}")
print(f"F1-Score: {cat_f1:.4f}")

print("\n=== DEFECT CLASSIFICATION METRICS ===")
print(f"Accuracy: {def_accuracy:.4f}")
print(f"Precision: {def_precision:.4f}")
print(f"Recall: {def_recall:.4f}")
print(f"F1-Score: {def_f1:.4f}")

# Save metrics to Excel
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
metrics_path = os.path.join(EVAL_DIR, f'evaluation_metrics_{timestamp}.xlsx')

with pd.ExcelWriter(metrics_path) as writer:
    pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Category': [cat_accuracy, cat_precision, cat_recall, cat_f1],
        'Defect': [def_accuracy, def_precision, def_recall, def_f1]
    }).to_excel(writer, sheet_name='Summary', index=False)
    
    pd.DataFrame(cat_cm, index=categories, columns=categories).to_excel(writer, sheet_name='Category_Confusion_Matrix')
    pd.DataFrame(def_cm, index=defect_classes, columns=defect_classes).to_excel(writer, sheet_name='Defect_Confusion_Matrix')

print(f"\nMetrics saved to: {metrics_path}")

# Plot confusion matrices
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Category Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(def_cm, annot=True, fmt='d', cmap='Greens', xticklabels=defect_classes, yticklabels=defect_classes)
plt.title('Defect Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
cm_path = os.path.join(EVAL_DIR, f'confusion_matrices_{timestamp}.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrices saved to: {cm_path}")

# Load training history
history_files = [f for f in os.listdir(EVAL_DIR) if f.startswith('training_history_') and f.endswith('.csv')]
if history_files:
    latest_history = sorted(history_files)[-1]
    history_path = os.path.join(EVAL_DIR, latest_history)
    history_df = pd.read_csv(history_path)
    
    # Plot loss curves
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df['category_accuracy'], label='Category Accuracy')
    plt.plot(history_df['val_category_accuracy'], label='Val Category Accuracy')
    plt.plot(history_df['defect_accuracy'], label='Defect Accuracy')
    plt.plot(history_df['val_defect_accuracy'], label='Val Defect Accuracy')
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    loss_path = os.path.join(EVAL_DIR, f'loss_curves_{timestamp}.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Loss curves saved to: {loss_path}")

# Find misclassified examples
print("\nFinding misclassified examples...")

# Category misclassifications
cat_misclassified = [(i, y_true_cat[i], y_pred_cat[i]) for i in range(len(y_true_cat)) 
                     if y_true_cat[i] != y_pred_cat[i]]

# Defect misclassifications
def_misclassified = [(i, y_true_def[i], y_pred_def[i]) for i in range(len(y_true_def)) 
                     if y_true_def[i] != y_pred_def[i]]

print(f"\nFound {len(cat_misclassified)} category misclassifications")
print(f"Found {len(def_misclassified)} defect misclassifications")

# Display some misclassified examples
if cat_misclassified:
    print("\n=== CATEGORY MISCLASSIFICATION EXAMPLES ===")
    for i, true, pred in cat_misclassified[:3]:
        img_path = image_paths[i]
        true_cat = categories[true]
        pred_cat = categories[pred]
        true_def = defect_classes[y_true_def[i]]
        pred_def = defect_classes[y_pred_def[i]]
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300))
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_cat} ({true_def})\nPredicted: {pred_cat} ({pred_def})")
        plt.axis('off')
        plt.show()

if def_misclassified:
    print("\n=== DEFECT MISCLASSIFICATION EXAMPLES ===")
    for i, true, pred in def_misclassified[:3]:
        img_path = image_paths[i]
        true_cat = categories[y_true_cat[i]]
        pred_cat = categories[y_pred_cat[i]]
        true_def = defect_classes[true]
        pred_def = defect_classes[pred]
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300))
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_cat} ({true_def})\nPredicted: {pred_cat} ({pred_def})")
        plt.axis('off')
        plt.show()

# Save misclassification details
misclassified_path = os.path.join(EVAL_DIR, f'misclassified_examples_{timestamp}.xlsx')
with pd.ExcelWriter(misclassified_path) as writer:
    if cat_misclassified:
        cat_mis_df = pd.DataFrame(cat_misclassified, columns=['Index', 'True', 'Predicted'])
        cat_mis_df['True_Category'] = cat_mis_df['True'].apply(lambda x: categories[x])
        cat_mis_df['Predicted_Category'] = cat_mis_df['Predicted'].apply(lambda x: categories[x])
        cat_mis_df['Image_Path'] = cat_mis_df['Index'].apply(lambda x: image_paths[x])
        cat_mis_df.to_excel(writer, sheet_name='Category_Misclassified', index=False)
    
    if def_misclassified:
        def_mis_df = pd.DataFrame(def_misclassified, columns=['Index', 'True', 'Predicted'])
        def_mis_df['True_Defect'] = def_mis_df['True'].apply(lambda x: defect_classes[x])
        def_mis_df['Predicted_Defect'] = def_mis_df['Predicted'].apply(lambda x: defect_classes[x])
        def_mis_df['Image_Path'] = def_mis_df['Index'].apply(lambda x: image_paths[x])
        def_mis_df.to_excel(writer, sheet_name='Defect_Misclassified', index=False)

print(f"\nMisclassification details saved to: {misclassified_path}")

# Add transparency note
transparency_note = """
**IMPORTANT TRANSPARENCY NOTE:**

The model was trained and evaluated on the full dataset due to the limited availability of labeled data. 
As such, the performance metrics represent the model's ability to fit the training data rather than its generalization capacity. 
Future work should evaluate on a held-out test set to better assess generalization.
"""

display(Markdown(transparency_note))

# Save transparency note
with open(os.path.join(EVAL_DIR, f'transparency_note_{timestamp}.txt'), 'w') as f:
    f.write(transparency_note)

print("\nEvaluation complete! All results saved to:", EVAL_DIR)
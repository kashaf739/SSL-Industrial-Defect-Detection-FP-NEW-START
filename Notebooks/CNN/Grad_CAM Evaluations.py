import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model, Model
import pickle
from IPython.display import display, Markdown

# Configuration
DATA_DIR = '/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad'
MODEL_DIR = '/content/drive/MyDrive/Colab/CNN-MODEL'
EVAL_DIR = '/content/drive/MyDrive/Colab/CNN-EVALUATION'
IMG_SIZE = (224, 224)

# Find the latest model and encoders
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('best_model_') and f.endswith('.h5')]
encoder_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('label_encoders_') and f.endswith('.pkl')]

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

# Find last convolutional layer
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

print(f"Using last convolutional layer: {last_conv_layer}")

# Create Grad-CAM model for defect detection
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer).output, model.get_layer('defect').output]
)

# Grad-CAM function for defect detection
def make_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        last_conv_layer_output, defect_pred = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Use the defect prediction score
        class_channel = defect_pred[:, 0]  # Defect output is a single neuron
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to display Grad-CAM results
def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load and process original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Defect Heatmap', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img_rgb)
    plt.title('Grad-CAM Overlay', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to preprocess image for model
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)  # Add batch dimension

# Find examples for visualization
defect_examples = []
good_examples = []

for category in categories:
    cat_dir = os.path.join(DATA_DIR, category)
    
    # Get defective examples
    test_dir = os.path.join(cat_dir, "test")
    for defect_type in os.listdir(test_dir):
        if defect_type != "good":
            defect_dir = os.path.join(test_dir, defect_type)
            img_files = [os.path.join(defect_dir, f) for f in os.listdir(defect_dir)[:2]]
            defect_examples.extend(img_files)
    
    # Get good examples
    train_good_dir = os.path.join(cat_dir, "train", "good")
    img_files = [os.path.join(train_good_dir, f) for f in os.listdir(train_good_dir)[:1]]
    good_examples.extend(img_files)

print(f"\nFound {len(defect_examples)} defective examples")
print(f"Found {len(good_examples)} good examples")

# Generate Grad-CAM visualizations
print("\nGenerating Grad-CAM visualizations for defective examples...")
for i, img_path in enumerate(defect_examples[:5]):
    print(f"\nProcessing defective example {i+1}: {os.path.basename(img_path)}")
    
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Get model predictions
    cat_pred, def_pred = model.predict(img_array)
    pred_category = categories[np.argmax(cat_pred)]
    pred_defect = "defective" if def_pred > 0.5 else "good"
    defect_prob = float(def_pred[0][0])
    
    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(img_array)
    
    # Display results
    display_gradcam(img_path, heatmap)
    
    # Print prediction details
    category = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
    defect_type = os.path.basename(os.path.dirname(img_path))
    
    print(f"True: {category} ({defect_type})")
    print(f"Predicted: {pred_category} ({pred_defect})")
    print(f"Defect Probability: {defect_prob:.4f}")

print("\nGenerating Grad-CAM visualizations for good examples...")
for i, img_path in enumerate(good_examples[:3]):
    print(f"\nProcessing good example {i+1}: {os.path.basename(img_path)}")
    
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Get model predictions
    cat_pred, def_pred = model.predict(img_array)
    pred_category = categories[np.argmax(cat_pred)]
    pred_defect = "defective" if def_pred > 0.5 else "good"
    defect_prob = float(def_pred[0][0])
    
    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(img_array)
    
    # Display results
    display_gradcam(img_path, heatmap)
    
    # Print prediction details
    category = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
    
    print(f"True: {category} (good)")
    print(f"Predicted: {pred_category} ({pred_defect})")
    print(f"Defect Probability: {defect_prob:.4f}")

# Add transparency note
transparency_note = """
**IMPORTANT TRANSPARENCY NOTE:**

The Grad-CAM visualizations are generated using the model trained on the full dataset. 
These visualizations show where the model focuses its attention when making predictions, 
but they represent the model's behavior on training data rather than its generalization capability. 
The heatmaps highlight regions that the model associates with defects, which may include 
both actual defects and artifacts from the training data.
"""

display(Markdown(transparency_note))

print("\nGrad-CAM visualization complete!")
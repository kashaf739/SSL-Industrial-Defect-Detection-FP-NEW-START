import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import pandas as pd
from datetime import datetime

# Configuration
DATA_DIR = '/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad'
MODEL_DIR = '/content/drive/MyDrive/Colab/CNN-MODEL'
EVAL_DIR = '/content/drive/MyDrive/Colab/CNN-EVALUATION'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SPLIT_RATIO = 0.2

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Get category names
categories = sorted(os.listdir(DATA_DIR))
print("Categories:", categories)

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
category_encoder = LabelEncoder()
defect_encoder = LabelEncoder()

category_labels_encoded = category_encoder.fit_transform(category_labels)
defect_labels_encoded = defect_encoder.fit_transform(defect_labels)

# Split data
X_train, X_val, y_cat_train, y_cat_val, y_def_train, y_def_val = train_test_split(
    image_paths,
    category_labels_encoded,
    defect_labels_encoded,
    test_size=SPLIT_RATIO,
    stratify=category_labels_encoded,
    random_state=42
)

# Create TensorFlow datasets
def load_and_preprocess(image_path, cat_label, def_label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Data augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    
    return img, {"category": cat_label, "defect": def_label}

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_cat_train, y_def_train))
train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_cat_val, y_def_val))
val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build model
def build_model(input_shape, num_categories):
    inputs = layers.Input(shape=input_shape)
    
    # Shared backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Feature extraction
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Category head
    category_branch = layers.Dense(256, activation='relu')(x)
    category_branch = layers.Dropout(0.3)(category_branch)
    category_output = layers.Dense(num_categories, activation='softmax', name='category')(category_branch)
    
    # Defect head
    defect_branch = layers.Dense(128, activation='relu')(x)
    defect_branch = layers.Dropout(0.3)(defect_branch)
    defect_output = layers.Dense(1, activation='sigmoid', name='defect')(defect_branch)
    
    return Model(inputs=inputs, outputs=[category_output, defect_output])

model = build_model(IMG_SIZE + (3,), len(categories))
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss={
        'category': 'sparse_categorical_crossentropy',
        'defect': 'binary_crossentropy'
    },
    loss_weights={'category': 1.0, 'defect': 2.0},
    metrics={
        'category': 'accuracy',
        'defect': ['accuracy', tf.keras.metrics.AUC(name='auc')]
    }
)

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = os.path.join(MODEL_DIR, f'best_model_{timestamp}.h5')
log_path = os.path.join(EVAL_DIR, f'training_log_{timestamp}.csv')

# Model checkpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor='val_defect_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

# CSV logger callback
csv_logger = tf.keras.callbacks.CSVLogger(log_path)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, csv_logger, early_stopping],
    verbose=1
)

# Save training history
history_df = pd.DataFrame(history.history)
history_path = os.path.join(EVAL_DIR, f'training_history_{timestamp}.csv')
history_df.to_csv(history_path, index=False)

# Evaluate model
results = model.evaluate(val_ds)
print("\nValidation Results:")
print(f"Category Loss: {results[1]:.4f}, Accuracy: {results[3]:.4f}")
print(f"Defect Loss: {results[2]:.4f}, Accuracy: {results[4]:.4f}, AUC: {results[5]:.4f}")

# Generate predictions for classification report
y_pred_cat = []
y_pred_def = []
y_true_cat = []
y_true_def = []

for images, labels in val_ds:
    cat_pred, def_pred = model.predict(images)
    y_pred_cat.extend(np.argmax(cat_pred, axis=1))
    y_pred_def.extend((def_pred > 0.5).astype(int).flatten())
    y_true_cat.extend(labels['category'].numpy())
    y_true_def.extend(labels['defect'].numpy())

# Print classification reports
print("\nCategory Classification Report:")
cat_report = classification_report(
    y_true_cat,
    y_pred_cat,
    target_names=categories,
    zero_division=0,
    output_dict=True
)
print(classification_report(
    y_true_cat,
    y_pred_cat,
    target_names=categories,
    zero_division=0
))

print("\nDefect Classification Report:")
def_report = classification_report(
    y_true_def,
    y_pred_def,
    target_names=defect_encoder.classes_,
    zero_division=0,
    output_dict=True
)
print(classification_report(
    y_true_def,
    y_pred_def,
    target_names=defect_encoder.classes_,
    zero_division=0
))

# Save evaluation reports
eval_path = os.path.join(EVAL_DIR, f'evaluation_report_{timestamp}.xlsx')
with pd.ExcelWriter(eval_path) as writer:
    pd.DataFrame(cat_report).transpose().to_excel(writer, sheet_name='Category_Report')
    pd.DataFrame(def_report).transpose().to_excel(writer, sheet_name='Defect_Report')
    pd.DataFrame([{
        'Category Loss': results[1],
        'Category Accuracy': results[3],
        'Defect Loss': results[2],
        'Defect Accuracy': results[4],
        'Defect AUC': results[5]
    }]).to_excel(writer, sheet_name='Summary_Metrics')

# Save label encoders
encoders_path = os.path.join(MODEL_DIR, f'label_encoders_{timestamp}.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump({
        'category': category_encoder,
        'defect': defect_encoder
    }, f)

print(f"\nTraining complete! Model saved to: {model_path}")
print(f"Evaluation results saved to: {eval_path}")
print(f"Training history saved to: {history_path}")
print(f"Label encoders saved to: {encoders_path}")
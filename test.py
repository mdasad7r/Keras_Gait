"""import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model.cnn_encoder import build_cnn_encoder
from model.tkan import build_tkan_classifier
from dataset.casia_dataset import load_gallery_and_probe

# === CONFIGURE CHECKPOINT HERE ===
CHECKPOINT_PATH = "/content/Keras_Gait/casia-b/checkpoints/epoch_50.keras"

# === Load gallery/probe test splits ===
gallery, probes = load_gallery_and_probe()
print(f"🎯 Gallery: {gallery[0].shape} samples | Probes: {[f'{k}: {v[0].shape}' for k, v in probes.items()]}")

# === Build model ===
def build_model():
    input_shape = (
        config.SEQUENCE_LEN,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    )
    sequence_input = tf.keras.Input(shape=input_shape, name="sequence_input")

    encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    x = tf.keras.layers.TimeDistributed(encoder)(sequence_input)

    tkan_model = build_tkan_classifier(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    output = tkan_model(x)
    return tf.keras.Model(inputs=sequence_input, outputs=output, name="CNN_TKAN_GaitRecognizer")

model = build_model()
model.load_weights(CHECKPOINT_PATH)
print(f"✅ Loaded model from {CHECKPOINT_PATH}")

# === Evaluation helper ===
def evaluate(X, y, label=""):
    y_pred = np.argmax(model.predict(X, batch_size=config.BATCH_SIZE), axis=1)
    acc = accuracy_score(y, y_pred)
    print(f"\n✅ Accuracy [{label}]: {acc:.4f}")
    print(classification_report(y, y_pred, digits=4))
    
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    return acc

# === Evaluate for each condition ===
X_gallery, y_gallery = gallery
for cond, (X_probe, y_probe) in probes.items():
    print(f"\n🔍 Evaluating on probe set: {cond.upper()} (shape: {X_probe.shape})")
    evaluate(X_probe, y_probe, label=cond.upper())
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model.cnn_encoder import build_cnn_encoder
from model.tkan import build_tkan_classifier
from dataset.casia_dataset import load_gallery_and_probe

# === CONFIGURE CHECKPOINT HERE ===
CHECKPOINT_PATH = "/content/Keras_Gait/casia-b/checkpoints/epoch_50.keras"

# === Load gallery/probe test splits ===
gallery, probes = load_gallery_and_probe()
print(f"🎯 Loaded gallery shape: {gallery[0].shape}")

# === Build model ===
def build_model():
    input_shape = (
        config.SEQUENCE_LEN,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    )
    sequence_input = tf.keras.Input(shape=input_shape, name="sequence_input")

    encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    x = tf.keras.layers.TimeDistributed(encoder)(sequence_input)

    tkan_model = build_tkan_classifier(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    output = tkan_model(x)
    return tf.keras.Model(inputs=sequence_input, outputs=output, name="CNN_TKAN_GaitRecognizer")

model = build_model()
model.load_weights(CHECKPOINT_PATH)
print(f"✅ Loaded model from {CHECKPOINT_PATH}")

# === Evaluation by view angle and condition ===
view_angles = ["0", "18", "36", "54", "72", "90", "108", "126", "144", "162", "180"]
results = {
    "NM#5-6": {},
    "BG#1-2": {},
    "CL#1-2": {}
}

for condition, views in probes.items():
    print(f"\n📁 Evaluating condition: {condition}")
    acc_list = []
    for angle in view_angles:
        if angle not in views:
            print(f"⚠️ Missing view {angle} for {condition}")
            results[condition][angle] = None
            continue

        X_probe, y_probe = views[angle]
        print(f"🔍 View {angle}° - Samples: {X_probe.shape[0]}")
        y_pred = np.argmax(model.predict(X_probe, batch_size=config.BATCH_SIZE), axis=1)
        acc = accuracy_score(y_probe, y_pred)
        results[condition][angle] = acc * 100  # store as percentage
        acc_list.append(acc * 100)

    mean_acc = np.mean(acc_list)
    results[condition]["Mean"] = mean_acc
    print(f"✅ Mean Accuracy [{condition}]: {mean_acc:.2f}%")

# === Print results in CASIA-B table format ===
print("\n📊 Final Rank-1 Accuracy Table (%):\n")
header = "Condition\t" + "\t".join(view_angles) + "\tMean"
print(header)
print("-" * len(header.expandtabs()))

for condition, accs in results.items():
    row = [condition]
    for angle in view_angles:
        val = accs.get(angle)
        row.append(f"{val:.1f}" if val is not None else "-")
    row.append(f"{accs['Mean']:.1f}")
    print("\t".join(row))

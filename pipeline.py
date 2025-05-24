# pipeline.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_PATH = "data/sensor_data.csv"
SEQ_LEN = 24
BATCH_SIZE = 32
EPOCHS = 50
VAL_SPLIT = 0.2
NOISE_STD = 0.01
DROPOUT_RATE = 0.05
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["field1", "field2", "field3"]
TARGET_IDX = [0, 1, 2]

# Load & normalize dataset
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.dropna().reset_index(drop=True)
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# Build sequences
X_all, y_all = [], []
vals = df[FEATURES].values
for i in range(len(vals) - SEQ_LEN):
    X_all.append(vals[i:i+SEQ_LEN])
    y_all.append(vals[i+SEQ_LEN])

X_all = np.array(X_all, dtype="float32")
y_all = np.array(y_all, dtype="float32")

# Split dataset
n_val = int(len(X_all) * VAL_SPLIT)
X_train, X_val = X_all[:-n_val], X_all[-n_val:]
y_train, y_val = y_all[:-n_val], y_all[-n_val:]

# Augment training data
def augment(X, noise_std, dropout_rate):
    Xn = X + np.random.normal(0, noise_std, X.shape)
    Xn = np.clip(Xn, 0.0, 1.0)
    if dropout_rate > 0:
        mask = np.random.rand(*Xn.shape) < dropout_rate
        Xn[mask] = 0.0
    return Xn

X_aug = augment(X_train, NOISE_STD, DROPOUT_RATE)
y_aug = y_train.copy()
X_train = np.concatenate([X_train, X_aug])
y_train = np.concatenate([y_train, y_aug])

print(f"Training: {len(X_train)}, Validation: {len(X_val)}")

# Build TCN model
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)
    c1 = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    d1 = layers.Dropout(dropout_rate)(c1)
    c2 = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(d1)
    d2 = layers.Dropout(dropout_rate)(c2)
    return layers.Add()([x, d2])

inp = keras.Input(shape=(SEQ_LEN, len(FEATURES)))
x = inp
for d in [1, 2, 4]:
    x = tcn_block(x, 32, 3, d, 0.1)
x = layers.Lambda(lambda t: t[:, -1, :])(x)
out = layers.Dense(len(FEATURES), activation="linear")(x)
model = keras.Model(inp, out)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Train
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=2)

# Save training plots
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "training_loss.png"))

plt.figure()
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.title("MAE over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "training_mae.png"))

# Evaluate
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"\nEvaluation: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# Save predictions
pred_df = pd.DataFrame({
    "Actual_SM": y_val[:, 0], "Pred_SM": y_pred[:, 0],
    "Actual_T": y_val[:, 1], "Pred_T": y_pred[:, 1],
    "Actual_H": y_val[:, 2], "Pred_H": y_pred[:, 2],
})
pred_df.to_csv(os.path.join(MODEL_DIR, "last_predictions.csv"), index=False)

# Save model
model.save(os.path.join(MODEL_DIR, "tcn_model.h5"))

# Convert to quantized TFLite
def rep_data_gen():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(os.path.join(MODEL_DIR, "tcn_model_int8.tflite"), "wb") as f:
    f.write(tflite_model)

print("✅ TCN model quantized and saved.")

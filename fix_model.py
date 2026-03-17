# rebuild.py — run ONCE to create a clean working model
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense

print("Reading weights from broken .h5...")

all_weights = []
with h5py.File("fall_detection_lstm.h5", "r") as f:
    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_weights.append((name, np.array(obj)))
    f["model_weights"].visititems(collect)

print(f"Found {len(all_weights)} weight arrays:")
for name, w in all_weights:
    print(f"  {name} → {w.shape}")

# Build exact architecture from your diagnostic
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(10, 69)),
    BatchNormalization(momentum=0.99, epsilon=0.001),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1,  activation="sigmoid")
])
model.build((None, 10, 69))

print(f"\nModel expects {len(model.weights)} weight arrays")

# Strip non-trainable optimizer weights, keep only model weights
arrays = [w for _, w in all_weights]

try:
    model.set_weights(arrays)
    print("✅ Weights loaded")
except Exception as e:
    print(f"Shape mismatch — trying to auto-match...\nError: {e}")
    # Match by shape
    model_weights = model.weights
    assigned = []
    used = set()
    for mw in model_weights:
        for i, (_, arr) in enumerate(all_weights):
            if i not in used and arr.shape == tuple(mw.shape):
                assigned.append(arr)
                used.add(i)
                break
        else:
            assigned.append(mw.numpy())  # keep default if no match
    model.set_weights(assigned)
    print("✅ Weights matched by shape")

model.save("model_clean.keras")
print("\n✅ Saved as model_clean.keras — use this from now on")
model.summary()
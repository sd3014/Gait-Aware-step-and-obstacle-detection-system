import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("gait_features.csv")

print("Dataset shape:", df.shape)

# Remove non-numeric columns
df = df.select_dtypes(include=["number"])

print("After removing text columns:", df.shape)

# Separate features and label
X = df.drop(columns=["label"]).values
y = df["label"].values

# =============================
# Sequence length
# =============================

SEQ_LEN = 10

X_seq = []
y_seq = []

print("Creating sequences...")

for i in range(len(X) - SEQ_LEN):

    X_seq.append(X[i:i+SEQ_LEN])

    y_seq.append(y[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence dataset shape:", X_seq.shape)

# =============================
# Save sequences
# =============================

np.save("X_sequences.npy", X_seq)
np.save("y_sequences.npy", y_seq)

print("Sequences saved successfully")
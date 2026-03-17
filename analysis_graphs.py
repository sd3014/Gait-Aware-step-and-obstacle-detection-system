import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

print("Loading model...")

model = load_model("fall_detection_lstm.h5")

print("Loading dataset...")

X = np.load("X_sequences.npy", allow_pickle=True).astype("float32")
y = np.load("y_sequences.npy", allow_pickle=True).astype("int32")

# =========================
# Train/Test Split
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Predictions
# =========================

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))

# =========================
# 1 Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ADL","Fall"],
            yticklabels=["ADL","Fall"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================
# 2 ROC Curve
# =========================

fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.show()

# =========================
# 3 Precision Recall Curve
# =========================

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.show()

# =========================
# 4 Class Distribution
# =========================

import pandas as pd

pd.Series(y).value_counts().plot(kind="bar")

plt.xticks([0,1],["ADL","Fall"])

plt.title("Class Distribution")

plt.show()

# =========================
# 5 Feature Magnitude Analysis
# =========================

feature_importance = np.mean(np.abs(X_train), axis=(0,1))

plt.figure(figsize=(8,6))

plt.bar(range(len(feature_importance)), feature_importance)

plt.xlabel("Pose Feature Index")
plt.ylabel("Average Magnitude")

plt.title("Pose Feature Contribution")

plt.show()

# =========================
# 6 Fall Probability Timeline
# =========================

plt.figure(figsize=(10,4))

plt.plot(y_prob[:200])

plt.title("Fall Probability Over Time")

plt.xlabel("Frame Sequence")
plt.ylabel("Fall Probability")

plt.show()

# =========================
# 7 Prediction Distribution
# =========================

plt.figure(figsize=(6,5))

sns.histplot(y_prob, bins=50, kde=True)

plt.title("Prediction Probability Distribution")

plt.xlabel("Fall Probability")

plt.show()
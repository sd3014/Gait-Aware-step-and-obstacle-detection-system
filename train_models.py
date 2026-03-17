import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_csv("gait_features.csv")

print("Total samples:",len(df))

X = df.drop(["label","subject"],axis=1)
y = df["label"]
groups = df["subject"]

print("Splitting dataset (subject-wise)...")

gss = GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)

train_idx,test_idx = next(gss.split(X,y,groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Train size:",len(X_train))
print("Test size:",len(X_test))

print("Balancing data using SMOTE...")

smote = SMOTE(random_state=42)

X_train,y_train = smote.fit_resample(X_train,y_train)

print("Training RandomForest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

print("\nRandomForest Results")
print(classification_report(y_test,pred_rf))


print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train,y_train)

pred_xgb = xgb.predict(X_test)

print("\nXGBoost Results")
print(classification_report(y_test,pred_xgb))
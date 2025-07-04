# model.py - Predict if stock price will rise next day (Binary Classification)
# 0 - Next day closing price same or lower → Stable/Down
# 1 - Next day closing price higher → Up

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)


def train_lightgbm(df, features, label_col='Target'):
    X = df[features]
    y = df[label_col]

    # Handle class imbalance
    counts = Counter(y)
    neg, pos = counts[0], counts[1]
    scale = neg / pos if pos > 0 else 1  # Avoid div by zero

    # Train-test split (fixed, no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.25
    )

    # Train LightGBM
    model = LGBMClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.01,
        reg_lambda=0.09,  # L2 regularization
        scale_pos_weight=scale,
        random_state=42,
        subsample=0.8
    )
    model.fit(X_train, y_train)

    # Predictions & Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Save best model (overwrite only if both F1 & Recall improve)
    model_path = 'best_model.pkl'
    metrics_path = 'best_metrics.pkl'

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            best_metrics = pickle.load(f)
    else:
        best_metrics = {'f1': 0, 'recall': 0}

    if f1 > best_metrics['f1'] and recall > best_metrics['recall']:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(metrics_path, 'wb') as f:
            pickle.dump({'f1': f1, 'recall': recall}, f)
        print("✅ New best model saved (F1: {:.3f}, Recall: {:.3f})".format(f1, recall))
    else:
        print("⚠️ Model not improved (F1: {:.3f}, Recall: {:.3f})".format(f1, recall))

    return model, accuracy, report, X.columns, fig, f1, precision, recall

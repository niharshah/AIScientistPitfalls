#!/usr/bin/env python
# Import basic libraries
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
from scipy.sparse import hstack

# --------------------------
# Load SPR_BENCH dataset using provided settings.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter=",")

print("Train dataset sample:", dataset["train"][0])
print("Validation dataset sample:", dataset["validation"][0])
print("Test dataset sample:", dataset["test"][0])

# --------------------------
# Define helper functions as provided (for SWA calculation)
def count_shape_variety(sequence: str) -> int:
    # Count the number of unique shape types in the sequence.
    return len(set(token[0] for token in sequence.strip().split() if token))

def count_color_variety(sequence: str) -> int:
    # Count the number of unique color types in the sequence.
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

# --------------------------
# Prepare data lists from datasets.
# Extract sequences and labels for train, dev, and test splits.
train_seqs = [ex["sequence"] for ex in dataset["train"]]
train_labels = [int(ex["label"]) for ex in dataset["train"]]

dev_seqs = [ex["sequence"] for ex in dataset["validation"]]
dev_labels = [int(ex["label"]) for ex in dataset["validation"]]

test_seqs = [ex["sequence"] for ex in dataset["test"]]
# Since test labels may be withheld, for simulation we assume they are provided (if not, evaluation on dev is sufficient)
try:
    test_labels = [int(ex["label"]) for ex in dataset["test"]]
except Exception:
    test_labels = None

# --------------------------
# Experiment 1: Baseline Logistic Regression using TF-IDF feature extraction.
print("\nExperiment 1: Baseline Logistic Regression with TF-IDF features")
print("This experiment uses a TfidfVectorizer on the raw sequence text and trains a logistic regression classifier.")
# Modify TF-IDF vectorizer to tokenize by splitting whitespace to handle special tokens.
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_seqs)
X_dev_tfidf = tfidf_vectorizer.transform(dev_seqs)
X_test_tfidf = tfidf_vectorizer.transform(test_seqs)

# Train Logistic Regression
lr_clf = LogisticRegression(max_iter=200)
lr_clf.fit(X_train_tfidf, train_labels)

# Predict on development set
dev_preds_lr = lr_clf.predict(X_dev_tfidf)
swa_lr = shape_weighted_accuracy(dev_seqs, dev_labels, dev_preds_lr)
print("Logistic Regression Dev evaluation:")
print("  Shape-Weighted Accuracy (SWA): {:.2f}%".format(swa_lr * 100))

# --------------------------
# Experiment 2: Random Forest with Enhanced Features (TF-IDF + additional rule-based features)
print("\nExperiment 2: Random Forest with TF-IDF features plus additional shape/color variety features")
print("This experiment augments the TF-IDF representation with two extra features: the number of unique shapes and colors in each sequence.")
# Compute additional features for train, dev, test splits.
train_shape_feat = np.array([count_shape_variety(seq) for seq in train_seqs]).reshape(-1, 1)
train_color_feat = np.array([count_color_variety(seq) for seq in train_seqs]).reshape(-1, 1)
train_extra = np.hstack([train_shape_feat, train_color_feat])

dev_shape_feat = np.array([count_shape_variety(seq) for seq in dev_seqs]).reshape(-1, 1)
dev_color_feat = np.array([count_color_variety(seq) for seq in dev_seqs]).reshape(-1, 1)
dev_extra = np.hstack([dev_shape_feat, dev_color_feat])

test_shape_feat = np.array([count_shape_variety(seq) for seq in test_seqs]).reshape(-1, 1)
test_color_feat = np.array([count_color_variety(seq) for seq in test_seqs]).reshape(-1, 1)
test_extra = np.hstack([test_shape_feat, test_color_feat])

# Combine TF-IDF features with extra features
X_train_rf = hstack([X_train_tfidf, train_extra])
X_dev_rf = hstack([X_dev_tfidf, dev_extra])
X_test_rf = hstack([X_test_tfidf, test_extra])

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_rf, train_labels)

# Predict on development set
dev_preds_rf = rf_clf.predict(X_dev_rf)
swa_rf = shape_weighted_accuracy(dev_seqs, dev_labels, dev_preds_rf)
print("Random Forest Dev evaluation:")
print("  Shape-Weighted Accuracy (SWA): {:.2f}%".format(swa_rf * 100))

# --------------------------
# Choose the best model based on dev SWA and evaluate on test set.
if swa_rf >= swa_lr:
    best_model_name = "Random Forest (TF-IDF + Extra Features)"
    best_model = rf_clf
    X_test_best = X_test_rf
else:
    best_model_name = "Logistic Regression (TF-IDF Only)"
    best_model = lr_clf
    X_test_best = X_test_tfidf

print("\nSelecting best model based on dev SWA performance...")
print("Best model selected:", best_model_name)

# Evaluate on the test set
test_preds = best_model.predict(X_test_best)
if test_labels is not None:
    swa_test = shape_weighted_accuracy(test_seqs, test_labels, test_preds)
    print("\nTest Set Evaluation for {}:".format(best_model_name))
    print("  Shape-Weighted Accuracy (SWA): {:.2f}%".format(swa_test * 100))
else:
    print("\nTest labels are not available; skipping test evaluation.")

# --------------------------
# Generate Figure 1: Confusion Matrix for test set predictions (if test labels available)
if test_labels is not None:
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Figure_1.png: Confusion Matrix on Test Set ({})".format(best_model_name))
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Class 0", "Class 1"])
    plt.yticks(tick_marks, ["Class 0", "Class 1"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), 
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("Figure_1.png")
    print("\nFigure_1.png saved: Confusion Matrix for test set predictions.")
else:
    print("\nTest labels unavailable; skipping confusion matrix plotting.")

# --------------------------
# Generate Figure 2: Feature Importance (if available) or Top Coefficients.
print("\nGenerating Figure_2.png: Model feature importance visualization")
if best_model_name.startswith("Random Forest"):
    # For Random Forest, plot the feature importances.
    importances = best_model.feature_importances_
    # Get feature names from the TF-IDF vectorizer and add names for extra features.
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    extra_feature_names = ["shape_variety", "color_variety"]
    feature_names = list(tfidf_feature_names) + extra_feature_names
    # To avoid clutter, only display top 20 features by importance.
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Figure_2.png: Top 20 Feature Importances ({})".format(best_model_name))
    plt.tight_layout()
    plt.savefig("Figure_2.png")
    print("Figure_2.png saved: Top 20 feature importances for the Random Forest model.")
else:
    # For Logistic Regression, show top 20 coefficients.
    coefs = best_model.coef_[0]
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    # Get top 20 positive and negative coefficients by absolute value.
    order = np.argsort(np.abs(coefs))[-20:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(order)), coefs[order], align="center")
    plt.yticks(range(len(order)), [tfidf_feature_names[i] for i in order])
    plt.xlabel("Coefficient Value")
    plt.title("Figure_2.png: Top 20 Feature Coefficients ({})".format(best_model_name))
    plt.tight_layout()
    plt.savefig("Figure_2.png")
    print("Figure_2.png saved: Top 20 feature coefficients for the Logistic Regression model.")

print("\nAll experiments completed. The code ran all experiments and saved the result figures.")
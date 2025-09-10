import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools

# ---------- Dataset Loading ----------
# Load the SPR_BENCH dataset from local CSV files using HuggingFace's datasets library
data_files = {
    "train": "SPR_BENCH/train.csv",
    "validation": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)
print("Loaded dataset structure:")
print(dataset)

# ---------- Utility Functions (Feature Engineering and Metric Calculation) ----------
def count_shape_variety(sequence: str) -> int:
    """Count the number of unique shape types in the sequence"""
    # Each token assumed to have shape info as first character
    return len(set(token[0] for token in sequence.strip().split() if token))

def count_color_variety(sequence: str) -> int:
    """Count the number of unique color types in the sequence"""
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))

def token_count(sequence: str) -> int:
    """Return the total number of tokens in the sequence."""
    return len(sequence.strip().split())

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Compute Shape-Weighted Accuracy (SWA)"""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

# ---------- Feature Extraction ----------
def extract_features(ds):
    # Extract basic features: shape_variety, color_variety, token_count
    X = []
    y = []
    sequences = []
    for sample in ds:
        seq = sample["sequence"]
        # Compute features
        shape_var = count_shape_variety(seq)
        color_var = count_color_variety(seq)
        tcount = token_count(seq)
        # Use these as a simple feature vector
        X.append([shape_var, color_var, tcount])
        y.append(int(sample["label"]))
        sequences.append(seq)
    return np.array(X), np.array(y), sequences

# ---------- Prepare Data ----------
print("\nExtracting features for training set...")
X_train, y_train, seq_train = extract_features(dataset["train"])
print("Training set feature shape:", X_train.shape)

print("\nExtracting features for development (validation) set...")
X_dev, y_dev, seq_dev = extract_features(dataset["validation"])
print("Development set feature shape:", X_dev.shape)

print("\nExtracting features for test set...")
X_test, y_test, seq_test = extract_features(dataset["test"])
print("Test set feature shape:", X_test.shape)

# ---------- Experiment 1: Baseline Logistic Regression ----------
print("\nExperiment 1: Training Baseline Logistic Regression Model")
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)

y_dev_pred_lr = clf_lr.predict(X_dev)
swa_dev_lr = shape_weighted_accuracy(seq_dev, y_dev, y_dev_pred_lr)
print("Dev SWA for Logistic Regression (Baseline): {:.2f}%".format(swa_dev_lr * 100))

y_test_pred_lr = clf_lr.predict(X_test)
swa_test_lr = shape_weighted_accuracy(seq_test, y_test, y_test_pred_lr)
print("Test SWA for Logistic Regression (Baseline): {:.2f}%".format(swa_test_lr * 100))

# ---------- Experiment 2: Improved Model using Random Forest ----------
print("\nExperiment 2: Training Improved Random Forest Classifier Model")
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

y_dev_pred_rf = clf_rf.predict(X_dev)
swa_dev_rf = shape_weighted_accuracy(seq_dev, y_dev, y_dev_pred_rf)
print("Dev SWA for Random Forest (Improved): {:.2f}%".format(swa_dev_rf * 100))

y_test_pred_rf = clf_rf.predict(X_test)
swa_test_rf = shape_weighted_accuracy(seq_test, y_test, y_test_pred_rf)
print("Test SWA for Random Forest (Improved): {:.2f}%".format(swa_test_rf * 100))

# ---------- Select Best Model Based on Dev Performance ----------
if swa_dev_rf >= swa_dev_lr:
    best_model = "Random Forest"
    best_y_test_pred = y_test_pred_rf
    best_swa_test = swa_test_rf
    best_seq_test = seq_test
else:
    best_model = "Logistic Regression"
    best_y_test_pred = y_test_pred_lr
    best_swa_test = swa_test_lr
    best_seq_test = seq_test

print("\nBased on development set performance, the best model is:", best_model)
print("Best model Test SWA: {:.2f}%".format(best_swa_test * 100))

# ---------- Plotting Results: Figure 1 ----------
# This figure compares the development set SWA performance of both models.
models = ["Logistic Regression", "Random Forest"]
swa_dev_scores = [swa_dev_lr * 100, swa_dev_rf * 100]

plt.figure(figsize=(8,6))
bars = plt.bar(models, swa_dev_scores, color=['blue', 'green'])
plt.xlabel("Model")
plt.ylabel("Shape-Weighted Accuracy (Dev %) ")
plt.title("Figure_1.png: Dev SWA Comparison Between Baseline and Improved Models")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
plt.ylim(0, 100)
plt.savefig("Figure_1.png")
plt.close()
print("\nFigure_1.png generated: Bar chart comparing dev SWA for Logistic Regression and Random Forest.")

# ---------- Plotting Results: Figure 2 ----------
# This figure shows the confusion matrix for the best performing model on the test set.
cm = confusion_matrix(y_test, best_y_test_pred)

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cm, classes=["0", "1"], title="Figure_2.png: Confusion Matrix on Test Set (Best Model: {})".format(best_model))
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png generated: Confusion matrix of the best model on the test set.\n")

# ---------- Final Summary of Experiments ----------
print("Summary of Experiments:")
print("------------------------------------------------")
print("Baseline Logistic Regression:")
print("  Dev SWA: {:.2f}%  |  Test SWA: {:.2f}%".format(swa_dev_lr * 100, swa_test_lr * 100))
print("Improved Random Forest:")
print("  Dev SWA: {:.2f}%  |  Test SWA: {:.2f}%".format(swa_dev_rf * 100, swa_test_rf * 100))
print("Selected Best Model:", best_model)
print("Final Test SWA (Best Model): {:.2f}%".format(best_swa_test * 100))
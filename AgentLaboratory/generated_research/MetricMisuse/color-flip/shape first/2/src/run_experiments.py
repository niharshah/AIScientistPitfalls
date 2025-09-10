import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ---------------- Dataset Loading and Preprocessing ---------------- #
# Load the local SPR_BENCH dataset from CSV files
data_files = {
    "train": "./SPR_BENCH/train.csv",
    "dev": "./SPR_BENCH/dev.csv",
    "test": "./SPR_BENCH/test.csv"
}

# Load CSV files with HuggingFace datasets
dataset = load_dataset("csv", data_files=data_files)

# Process the dataset:
# 1. Split the 'sequence' column into a list of tokens.
# 2. Calculate the shape complexity: count of unique shape glyphs (first character of each token).
# 3. Calculate the color complexity: count of unique color letters (if present) from each token.
# 4. Also compute token_count as an additional feature.
dataset = dataset.map(lambda x: {
    "tokens": x["sequence"].split(),
    "shape_complexity": len({token[0] for token in x["sequence"].split()}),
    "color_complexity": len({token[1] for token in x["sequence"].split() if len(token) > 1}),
    "token_count": len(x["sequence"].split())
})

# Verify processing by printing one example from the train split.
print("Example from training data with processed features:")
print(dataset["train"][0])


# ---------------- Helper Functions ---------------- #
def count_shape_variety(sequence: str) -> int:
    """Count the number of unique shape types in the sequence (first character of each token)."""
    return len(set(token[0] for token in sequence.strip().split() if token))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Shape-Weighted Accuracy (SWA): weights each sample by its shape variety when prediction is correct."""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ---------------- Feature Extraction and Model Training ---------------- #
# For our experiment, we use three simple features extracted from the dataset:
#  - shape_complexity, color_complexity, and token_count.
# We train a Logistic Regression model to predict the label.
# Our evaluation metric is the Shape-Weighted Accuracy (SWA).

def extract_features_and_labels(split):
    # We use original columns: shape_complexity, color_complexity, token_count.
    features = []
    labels = []
    sequences = []  # keep original sequence strings for metric calculation
    for sample in dataset[split]:
        # Ensure features are numeric. The label may be string; convert to int if possible.
        feat = [sample["shape_complexity"], sample["color_complexity"], sample["token_count"]]
        features.append(feat)
        try:
            # Attempt numeric conversion for the label.
            labels.append(int(sample["label"]))
        except:
            # If conversion fails, we use the label as is.
            labels.append(sample["label"])
        # Also capture original sequence for later use in SWA computations.
        sequences.append(sample["sequence"])
    return np.array(features), np.array(labels), sequences

# Extract training features and labels.
X_train, y_train, seq_train = extract_features_and_labels("train")
X_dev, y_dev, seq_dev = extract_features_and_labels("dev")
X_test, y_test, seq_test = extract_features_and_labels("test")

print("\nStarting training of Logistic Regression classifier on training data...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Training completed.")

# ---------------- Evaluation on Dev Set ---------------- #
print("\nExperiment 1: Evaluating performance on the Development (Dev) split.")
print("This experiment aims to show both the raw accuracy and the Shape-Weighted Accuracy (SWA) on the Dev set.")
y_dev_pred = clf.predict(X_dev)
raw_acc_dev = accuracy_score(y_dev, y_dev_pred)
swa_dev = shape_weighted_accuracy(seq_dev, y_dev, y_dev_pred)
print(f"Dev Raw Accuracy: {raw_acc_dev:.4f}")
print(f"Dev Shape-Weighted Accuracy (SWA): {swa_dev:.4f}")

# Generate Figure 1: Confusion Matrix on Dev set predictions.
cm = confusion_matrix(y_dev, y_dev_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Figure_1.png: Confusion Matrix for Dev Set Predictions")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png has been saved, showing the confusion matrix for the Dev set.")

# ---------------- Evaluation on Test Set ---------------- #
print("\nExperiment 2: Evaluating performance on the Test split.")
print("This experiment is designed to display the generalization of our model on unseen data using the Shape-Weighted Accuracy (SWA) metric.")
y_test_pred = clf.predict(X_test)
raw_acc_test = accuracy_score(y_test, y_test_pred)
swa_test = shape_weighted_accuracy(seq_test, y_test, y_test_pred)
print(f"Test Raw Accuracy: {raw_acc_test:.4f}")
print(f"Test Shape-Weighted Accuracy (SWA): {swa_test:.4f}")

# Generate Figure 2: Scatter Plot of Shape Complexity vs Token Count on Test Set colored by True Label.
plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0], X_test[:,2], c=y_test, cmap="viridis", alpha=0.7)
plt.title("Figure_2.png: Test Set - Shape Complexity vs Token Count (Colored by True Label)")
plt.xlabel("Shape Complexity")
plt.ylabel("Token Count")
plt.colorbar(label="True Label")
plt.tight_layout()
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png has been saved, showing a scatter plot of shape complexity vs token count for the Test set.")

print("\nAll experiments completed. The logistic regression classifier achieved non-zero accuracies on both Dev and Test splits.")
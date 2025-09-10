import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random

from datasets import load_dataset

# Load the local SPR_BENCH dataset from CSV files
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Print out the number of samples in each split
print("Train samples:", len(dataset["train"]))
print("Dev samples:", len(dataset["dev"]))
print("Test samples:", len(dataset["test"]))

def count_shape_variety(sequence: str) -> int:
    """统计序列中不重复的形状种类数"""
    return len(set(token[0] for token in sequence.strip().split() if token))

def count_color_variety(sequence: str) -> int:
    """统计序列中不重复的颜色种类数"""
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Shape-Weighted Accuracy (SWA)"""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

def color_weighted_accuracy(sequences, y_true, y_pred):
    """Color-Weighted Accuracy (CWA)"""
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# --- Data Preparation ---
# The dataset variable is assumed to be loaded via the provided dataset code at the beginning.
# Extracting the splits from the dataset.
# It is assumed that each CSV row has the following columns: id, sequence, label.
# Converting labels to integers.
X_train = dataset["train"]["sequence"]
y_train = [int(label) for label in dataset["train"]["label"]]

X_dev = dataset["dev"]["sequence"]
y_dev = [int(label) for label in dataset["dev"]["label"]]

X_test = dataset["test"]["sequence"]
# Since test labels are withheld in practice, here we assume they are present for local evaluation.
# If test labels are not provided, one can skip evaluation on test.
y_test = [int(label) for label in dataset["test"]["label"]]

print("Data loaded successfully:")
print("Train samples:", len(X_train))
print("Dev samples:", len(X_dev))
print("Test samples:", len(X_test))

# --- Feature Extraction ---
# The original error was due to the TF-IDF Vectorizer filtering out tokens.
# Here, we update the token_pattern to accept single character tokens as well.
vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

# --- Model Training ---
# We use an MLP classifier (a simple feed-forward network) that introduces non-linearities.
# This should help to capture the complex dependencies in the symbolic sequences.
print("\nStarting model training: The MLPClassifier is trained on the train set.")
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", 
                        max_iter=300, random_state=42)
mlp_clf.fit(X_train_vec, y_train)
print("Model training complete.\n")

# --- Evaluation on Dev Set ---
print("Experiment 1: Evaluation on the Dev split")
dev_preds = mlp_clf.predict(X_dev_vec)
dev_swa = shape_weighted_accuracy(X_dev, y_dev, dev_preds)
dev_cwa = color_weighted_accuracy(X_dev, y_dev, dev_preds)
dev_acc = accuracy_score(y_dev, dev_preds) * 100.0
print("Dev set evaluation details:")
print("-> The overall accuracy (percentage of correct predictions) is: {:.2f}%".format(dev_acc))
print("-> The Shape-Weighted Accuracy (SWA) is: {:.2f}".format(dev_swa))
print("-> The Color-Weighted Accuracy (CWA) is: {:.2f}".format(dev_cwa))
print("Note: SWA weights each sample's correctness by the number of unique shapes in its sequence.")

# --- Evaluation on Test Set ---
print("\nExperiment 2: Evaluation on the Test split")
test_preds = mlp_clf.predict(X_test_vec)
test_swa = shape_weighted_accuracy(X_test, y_test, test_preds)
test_cwa = color_weighted_accuracy(X_test, y_test, test_preds)
test_acc = accuracy_score(y_test, test_preds) * 100.0
print("Test set evaluation details:")
print("-> The overall accuracy (percentage of correct predictions) is: {:.2f}%".format(test_acc))
print("-> The Shape-Weighted Accuracy (SWA) is: {:.2f}".format(test_swa))
print("-> The Color-Weighted Accuracy (CWA) is: {:.2f}".format(test_cwa))
print("This evaluation on unseen test data checks the generalization performance of our model.")

# --- Visualization 1: Histogram of Shape Variety for Correct vs. Incorrect Predictions on Dev Set ---
print("\nGenerating Figure_1.png: Histogram of shape variety in sequences for correct vs. incorrect predictions on the Dev set.")
correct_shape_varieties = []
wrong_shape_varieties = []
for seq, true, pred in zip(X_dev, y_dev, dev_preds):
    variety = count_shape_variety(seq)
    if true == pred:
        correct_shape_varieties.append(variety)
    else:
        wrong_shape_varieties.append(variety)

plt.figure(figsize=(8,6))
bins = np.arange(0, max(correct_shape_varieties + wrong_shape_varieties + [1]) + 2) - 0.5
plt.hist(correct_shape_varieties, bins=bins, alpha=0.7, label="Correct Predictions", color="green")
plt.hist(wrong_shape_varieties, bins=bins, alpha=0.7, label="Incorrect Predictions", color="red")
plt.xlabel("Unique Shape Count in Sequence")
plt.ylabel("Number of Samples")
plt.title("Distribution of Shape Variety for Correct vs. Incorrect Predictions (Dev)")
plt.legend()
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png saved successfully.\n")

# --- Visualization 2: Confusion Matrix for Dev Set ---
print("Generating Figure_2.png: Confusion Matrix for predictions on the Dev set.")
cm = confusion_matrix(y_dev, dev_preds)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Dev Set)")
plt.colorbar()
classes = sorted(list(set(y_dev)))
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png saved successfully.\n")

print("All experiments completed. The model performance metrics and visualizations have been generated.")
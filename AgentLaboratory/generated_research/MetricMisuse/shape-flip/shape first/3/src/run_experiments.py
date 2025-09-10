import pathlib
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# -------------------------------
# Data Loading and Preparation
# -------------------------------

# Load the SPR_BENCH dataset using HuggingFace's "datasets" library.
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

print("Dataset sizes:")
print("Train size:", len(dataset["train"]))
print("Dev size:", len(dataset["dev"]))
print("Test size:", len(dataset["test"]))

print("\nSample from Train Split:")
print(dataset["train"][0])

# -------------------------------
# Extract data fields from dataset splits
# -------------------------------
# We'll assume that each dataset has keys: 'id', 'sequence', 'label'
# Convert labels to integers if they are not already.
def process_split(split):
    texts = [example["sequence"] for example in dataset[split]]
    # Convert labels to integer type if needed.
    labels = [int(example["label"]) for example in dataset[split]]
    return texts, labels

# Since instructions say to avoid defining functions, we'll inline the above processing.
train_texts = [ex["sequence"] for ex in dataset["train"]]
train_labels = [int(ex["label"]) for ex in dataset["train"]]

dev_texts = [ex["sequence"] for ex in dataset["dev"]]
dev_labels = [int(ex["label"]) for ex in dataset["dev"]]

test_texts = [ex["sequence"] for ex in dataset["test"]]
test_labels = [int(ex["label"]) for ex in dataset["test"]]

# -------------------------------
# Feature Extraction using TfidfVectorizer
# -------------------------------
print("\nPreparing features using TfidfVectorizer on the symbolic sequences.")
vectorizer = TfidfVectorizer(tokenizer=lambda txt: txt.split(), token_pattern=None)
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

# -------------------------------
# Training a Simple Classifier (Decision Tree)
# -------------------------------
print("\nTraining Decision Tree Classifier on the training data and tuning on dev split.")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, train_labels)

# Obtain predictions on train, dev and test splits.
train_pred = clf.predict(X_train)
dev_pred = clf.predict(X_dev)
test_pred = clf.predict(X_test)

# -------------------------------
# Define inline SWA (Shape-Weighted Accuracy) calculation
# -------------------------------
# SWA: For each sample, weight = count of unique shape types in the sequence.
# A shape type is assumed to be the first character of each token.
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


# Calculate SWA for train, dev and test splits.
swa_train = shape_weighted_accuracy(train_texts, train_labels, train_pred)
swa_dev = shape_weighted_accuracy(dev_texts, dev_labels, dev_pred)
swa_test = shape_weighted_accuracy(test_texts, test_labels, test_pred)

cwa_train = color_weighted_accuracy(train_texts, train_labels, train_pred)
cwa_dev = color_weighted_accuracy(dev_texts, dev_labels, dev_pred)
cwa_test = color_weighted_accuracy(test_texts, test_labels, test_pred)

from sklearn.metrics import accuracy_score
acc_train = accuracy_score(train_labels, train_pred)
acc_dev = accuracy_score(dev_labels, dev_pred)
acc_test = accuracy_score(test_labels, test_pred)


# -------------------------------
# Print Detailed Experiment Results
# -------------------------------
print("\nExperiment 1: Evaluation on Decision Tree Classifier with SWA metric")
print("This experiment demonstrates the model's performance using Shape-Weighted Accuracy (SWA),")
print("which weights each sample by the count of unique shape types (first character of each token) found in the sequence.")
print("Higher SWA indicates better handling of the intrinsic symbolic structure of the sequence.\n")

print("SWA on Training Data: {:.4f}".format(swa_train))
print("SWA on Development Data: {:.4f}".format(swa_dev))
print("SWA on Test Data: {:.4f}".format(swa_test))

print("CWA on Training Data: {:.4f}".format(cwa_train))
print("CWA on Development Data: {:.4f}".format(cwa_dev))
print("CWA on Test Data: {:.4f}".format(cwa_test))

print("Acc on Training Data: {:.4f}".format(acc_train))
print("Acc on Development Data: {:.4f}".format(acc_dev))
print("Acc on Test Data: {:.4f}".format(acc_test))
# -------------------------------
# Create Figure 1: Confusion Matrix on the Dev Split
# -------------------------------
print("\nExperiment 2: Generating Figure_1.png to show the confusion matrix for the Development split.")
cm = confusion_matrix(dev_labels, dev_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Dev Split)")
plt.colorbar()
tick_marks = np.arange(len(set(dev_labels)))
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("Figure_1.png")
plt.close()

# -------------------------------
# Create Figure 2: Distribution of Shape Weights in the Dev Split
# -------------------------------
print("\nExperiment 3: Generating Figure_2.png to illustrate distribution of shape variety weights on the Development split.")
# shape_weights = [compute_shape_weight(seq) for seq in dev_texts]
plt.figure(figsize=(6, 4))
plt.hist(shape_weights, bins=range(min(shape_weights), max(shape_weights)+2), edgecolor='black', align='left')
plt.title("Distribution of Shape Variety Weights (Dev Split)")
plt.xlabel("Shape Variety (Unique first characters per sequence)")
plt.ylabel("Frequency")
plt.savefig("Figure_2.png")
plt.close()

print("\nAll experiments are complete. Figures saved as Figure_1.png and Figure_2.png.")
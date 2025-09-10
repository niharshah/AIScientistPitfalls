# Import necessary libraries
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# ---------------------------
# Dataset Loading (pre-provided code)
# ---------------------------
dataset = load_dataset('csv', data_files={
    'train': './SPR_BENCH/train.csv',
    'dev': './SPR_BENCH/dev.csv',
    'test': './SPR_BENCH/test.csv'
}, delimiter=',')

print("Sample train instance:", dataset['train'][0])
print("Number of training instances:", len(dataset['train']))
print("Number of dev instances:", len(dataset['dev']))
print("Number of test instances:", len(dataset['test']))

# ---------------------------
# Utility Functions (as provided in literature and benchmark code)
# ---------------------------
def count_shape_variety(sequence: str) -> int:
    """Count the number of unique shape types in the sequence"""
    # Assuming each token is a string where the first character indicates the shape.
    return len(set(token[0] for token in sequence.strip().split() if token))

def count_color_variety(sequence: str) -> int:
    """Count the number of unique color types in the sequence"""
    # Assuming that the second character in each token (if available) indicates the color.
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))

def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Compute Shape-Weighted Accuracy (SWA)"""
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

def color_weighted_accuracy(sequences, y_true, y_pred):
    """Color-Weighted Accuracy (CWA)"""
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_features(dataset_split):
    # For each item in the dataset, we create three features:
    #   1. Number of unique shapes in the sequence.
    #   2. Number of unique colors in the sequence.
    #   3. Total number of tokens in the sequence.
    X = []
    y = []
    sequences = []
    for item in dataset_split:
        seq = item['sequence']
        X.append([
            count_shape_variety(seq),
            count_color_variety(seq),
            len(seq.strip().split())
        ])
        y.append(int(item['label']))
        sequences.append(seq)
    return np.array(X), np.array(y), sequences

# ---------------------------
# Prepare Data for Training and Evaluation
# ---------------------------
print("\nExtracting features from training, development, and test sets...")
X_train, y_train, seq_train = extract_features(dataset['train'])
X_dev, y_dev, seq_dev = extract_features(dataset['dev'])
X_test, y_test, seq_test = extract_features(dataset['test'])

print("Feature extraction completed. Sample features from training set:")
print(X_train[:5])

# ---------------------------
# Train a Baseline Classifier (Logistic Regression)
# ---------------------------
print("\nTraining logistic regression classifier on the training set...")
clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(X_train, y_train)
print("Training completed.")

# ---------------------------
# Evaluate the Classifier on the Development Set
# ---------------------------
print("\nEvaluation on the Development Set:")
y_dev_pred = clf.predict(X_dev)
dev_acc = accuracy_score(y_dev, y_dev_pred)
dev_swa = shape_weighted_accuracy(seq_dev, y_dev, y_dev_pred)
dev_cwa = color_weighted_accuracy(seq_dev, y_dev, y_dev_pred)

print("The results on the dev set are meant to show the classifier's performance on unseen data from the same distribution as training.")
print("Standard Accuracy on Dev set: {:.2f}%".format(dev_acc * 100))
print("Shape-Weighted Accuracy (SWA) on Dev set: {:.4f}".format(dev_swa))
print("Color-Weighted Accuracy (CWA) on Dev set: {:.4f}".format(dev_cwa))

# ---------------------------
# Evaluate the Classifier on the Test Set (Final Results)
# ---------------------------
print("\nEvaluation on the Test Set:")
y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_swa = shape_weighted_accuracy(seq_test, y_test, y_test_pred)
test_cwa = color_weighted_accuracy(seq_test, y_test, y_test_pred)
print("The following results are the final performance metrics on the withheld test set to validate generalization.")
print("Standard Accuracy on Test set: {:.2f}%".format(test_acc * 100))
print("Shape-Weighted Accuracy (SWA) on Test set: {:.4f}".format(test_swa))
print("Color-Weighted Accuracy (CWA) on Test set: {:.4f}".format(test_cwa))

# ---------------------------
# Generate Figures for Analysis
# ---------------------------

# Figure 1: Histogram of Shape Variety in the Training Set
print("\nGenerating Figure_1.png: Histogram of unique shape counts in training sequences.")
shape_varieties = [count_shape_variety(seq) for seq in seq_train]
plt.figure(figsize=(8, 6))
plt.hist(shape_varieties, bins=range(min(shape_varieties), max(shape_varieties)+2), edgecolor='black', alpha=0.7)
plt.title('Distribution of Unique Shape Counts in Training Sequences')
plt.xlabel('Number of Unique Shapes')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("Figure_1.png")
plt.close()

# Figure 2: Confusion Matrix for Development Set Predictions
print("Generating Figure_2.png: Confusion matrix showing distribution of correct and incorrect predictions on the dev set.")
cm = confusion_matrix(y_dev, y_dev_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix on Dev Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("Figure_2.png")
plt.close()

print("\nAll experiments have been run successfully. The figures (Figure_1.png and Figure_2.png) have been generated for further analysis.")
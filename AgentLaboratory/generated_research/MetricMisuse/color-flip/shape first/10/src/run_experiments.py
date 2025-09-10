import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# NOTE: The SPR_BENCH dataset (train, dev, test) is already loaded into the "dataset" variable
# with each example having fields: "id", "sequence", and "label".
# The provided dataset loader code (using HuggingFace datasets) is assumed to be run before this block.

print("\n[INFO] Sample from each split:") 
print("Train sample:", dataset["train"][0])
print("Dev sample:", dataset["dev"][0])
print("Test sample:", dataset["test"][0])

# -----------------------------------------------------------------------------
# Extract features from the sequences in each split.
# We extract three features:
#   1. Total number of tokens in the sequence.
#   2. Number of unique shapes (first character of each token).
#   3. Number of unique colors (second character of each token, if available).
# These features serve as a simple representation for the hidden rule verification task.
# ----------------------------------------------------------------------------- 

print("\n[INFO] Extracting features from the dataset splits...")

def extract_features_and_labels(data):
    X = []
    y = []
    sequences = []  # keep original sequences for metric computations later
    for example in data:
        seq = example["sequence"]
        tokens = seq.strip().split()
        num_tokens = len(tokens)
        # Compute shape diversity: count unique first characters (if token exists)
        shape_diversity = len(set(token[0] for token in tokens if token))
        # Compute color diversity: count unique second characters if token length > 1
        color_diversity = len(set(token[1] for token in tokens if len(token) > 1))
        X.append([num_tokens, shape_diversity, color_diversity])
        y.append(int(example["label"]))
        sequences.append(seq)
    return np.array(X), np.array(y), sequences

X_train, y_train, seq_train = extract_features_and_labels(dataset["train"])
X_dev, y_dev, seq_dev = extract_features_and_labels(dataset["dev"])
X_test, y_test, seq_test = extract_features_and_labels(dataset["test"])

print("[INFO] Feature extraction complete. Examples of feature vectors:")
print("Train features example:", X_train[0])
print("Dev features example:", X_dev[0])
print("Test features example:", X_test[0])

# -----------------------------------------------------------------------------
# Train a logistic regression classifier as our baseline model.
# We use the Train split to train and the Dev split for tuning.
# Our evaluation metric is chosen to be Shape-Weighted Accuracy (SWA).
#
# SWA is computed by weighing each example by the number of unique shapes in its sequence.
# For each example, if the predicted label equals the true label,
# the contribution is the unique shape count; otherwise 0. Then SWA = (sum of contributions) / (sum of weights).
# -----------------------------------------------------------------------------

print("\n[INFO] Training the logistic regression classifier on the Train split...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("[INFO] Training complete.")

# Evaluate on Train split
y_train_pred = clf.predict(X_train)
weights_train = [len(set(token[0] for token in seq.strip().split() if token)) for seq in seq_train]
correct_train = [w if yt == yp else 0 for w, yt, yp in zip(weights_train, y_train, y_train_pred)]
SWA_train = sum(correct_train) / sum(weights_train) if sum(weights_train) > 0 else 0.0
print("\n[RESULT] On the Train split, the Shape-Weighted Accuracy (SWA) is: {:.4f}".format(SWA_train))

# Evaluate on Dev split
print("\n[INFO] Evaluating on Dev split (tuning set). This result indicates the model's generalization on unseen data from training distribution.")
y_dev_pred = clf.predict(X_dev)
weights_dev = [len(set(token[0] for token in seq.strip().split() if token)) for seq in seq_dev]
correct_dev = [w if yt == yp else 0 for w, yt, yp in zip(weights_dev, y_dev, y_dev_pred)]
SWA_dev = sum(correct_dev) / sum(weights_dev) if sum(weights_dev) > 0 else 0.0
print("[RESULT] On the Dev split, the Shape-Weighted Accuracy (SWA) is: {:.4f}".format(SWA_dev))

# Evaluate on Test split (final evaluation)
print("\n[INFO] Evaluating on Test split. These labels are held out and represent final performance.")
y_test_pred = clf.predict(X_test)
weights_test = [len(set(token[0] for token in seq.strip().split() if token)) for seq in seq_test]
correct_test = [w if yt == yp else 0 for w, yt, yp in zip(weights_test, y_test, y_test_pred)]
SWA_test = sum(correct_test) / sum(weights_test) if sum(weights_test) > 0 else 0.0
print("[RESULT] On the Test split, the Shape-Weighted Accuracy (SWA) is: {:.4f}".format(SWA_test))

# -----------------------------------------------------------------------------
# Generate Figure_1.png:
# This figure displays a scatter plot of the training examples showing the relationship
# between the number of unique shapes (as extracted from the sequence) and the labels.
# The visualization aims to show if shape diversity is informative for the rule-based label.
# -----------------------------------------------------------------------------

print("\n[INFO] Generating Figure_1.png: Scatter plot of Training examples (Unique Shapes vs. Label)")
unique_shapes_train = [x[1] for x in X_train]  # shape diversity is the second feature
plt.figure(figsize=(8,6))
plt.scatter(unique_shapes_train, y_train, alpha=0.6, edgecolors='b')
plt.xlabel("Unique Shape Count")
plt.ylabel("Label")
plt.title("Training examples: Unique Shape Count vs. Label")
plt.grid(True)
plt.savefig("Figure_1.png")
plt.close()
print("[INFO] Figure_1.png saved.")

# -----------------------------------------------------------------------------
# Generate Figure_2.png:
# This figure shows the confusion matrix for the Dev split predictions.
# It visualizes the counts of true positives, false positives, true negatives, and false negatives,
# providing insights into the classification performance beyond the aggregate SWA metric.
# -----------------------------------------------------------------------------

print("\n[INFO] Generating Figure_2.png: Confusion Matrix for Dev set predictions")
cm = confusion_matrix(y_dev, y_dev_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Dev Split)")
plt.savefig("Figure_2.png")
plt.close()
print("[INFO] Figure_2.png saved.")

print("\n[INFO] All experiments completed. The baseline logistic regression model achieved the above SWA accuracies on Train, Dev, and Test splits.")
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# ------------------------------------------------------------------------------------
# Provided dataset loading and feature mapping code for SPR_BENCH
# ------------------------------------------------------------------------------------
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

def compute_complexities(example):
    tokens = example['sequence'].split()
    shape_set = set()
    color_set = set()
    for token in tokens:
        if token:  # Ensure token is not empty
            shape_set.add(token[0])
            if len(token) > 1:
                color_set.add(token[1])
    example["shape_complexity"] = len(shape_set)
    example["color_complexity"] = len(color_set)
    # Also record token count as an extra feature (could be useful)
    example["token_count"] = len(tokens)
    return example

# Map the complexities onto each split of the dataset
dataset = dataset.map(compute_complexities)
print("Dataset loaded and complexities computed:")
print(dataset)

# ------------------------------------------------------------------------------------
# Utility functions for computing Shape-Weighted Accuracy (SWA)
# ------------------------------------------------------------------------------------
def shape_weighted_accuracy(sequences, y_true, y_pred):
    """Compute Shape-Weighted Accuracy: for each sequence, the weight is the number of unique shape tokens.
    If the prediction is correct, the weight is added to the numerator.
    """
    weights = [len(set(token[0] for token in seq.strip().split() if token)) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0

# ------------------------------------------------------------------------------------
# Prepare feature matrices and labels from the dataset splits for model training
# We use three simple features: shape_complexity, color_complexity, and token_count.
# ------------------------------------------------------------------------------------
def extract_features_labels(split):
    data = dataset[split]
    # Extract features as a numpy array
    X = np.array([[ex["shape_complexity"], ex["color_complexity"], ex["token_count"]] for ex in data])
    # Convert labels to integer type (if they are not already)
    y = np.array([int(ex["label"]) for ex in data])
    # We also extract the raw sequence for SWA calculation later
    sequences = [ex["sequence"] for ex in data]
    return X, y, sequences

print("\nPreparing features and labels for Train, Dev, and Test splits...")
X_train, y_train, seq_train = extract_features_labels("train")
X_dev, y_dev, seq_dev = extract_features_labels("dev")
X_test, y_test, seq_test = extract_features_labels("test")
print("Feature extraction complete.")

# ------------------------------------------------------------------------------------
# Model Training and Tuning
# We use a Logistic Regression classifier (from scikit-learn) as our baseline model.
# Train on the train split and tune (if needed) on the dev split.
# ------------------------------------------------------------------------------------
print("\nExperiment 1: Model Training and Dev Evaluation")
print("This experiment trains a Logistic Regression model using three features derived from the input sequences and evaluates its performance on the dev split using both standard accuracy and Shape-Weighted Accuracy (SWA).")

# Initialize and train the model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predictions on the dev split
y_dev_pred = clf.predict(X_dev)

# Compute standard accuracy
dev_accuracy = accuracy_score(y_dev, y_dev_pred)

# Compute Shape-Weighted Accuracy on dev split
dev_swa = shape_weighted_accuracy(seq_dev, y_dev, y_dev_pred)

print("Dev Evaluation Results:")
print("Standard Accuracy on Dev set: {:.4f}".format(dev_accuracy))
print("Shape-Weighted Accuracy (SWA) on Dev set: {:.4f}".format(dev_swa))

# ------------------------------------------------------------------------------------
# Evaluate on Test Split and generate figures
# ------------------------------------------------------------------------------------
print("\nExperiment 2: Model Evaluation on the Test Split and Visualization")
print("This experiment evaluates the trained model on the Test split. It computes the standard accuracy, confusion matrix, and Shape-Weighted Accuracy (SWA). Additionally, two figures are generated:")
print("Figure_1.png shows the distribution of shape_complexity values in the Train set, separated by label.")
print("Figure_2.png displays the confusion matrix on the Test set predictions.")

# Predictions on test split
y_test_pred = clf.predict(X_test)

# Compute standard test accuracy and SWA on test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_swa = shape_weighted_accuracy(seq_test, y_test, y_test_pred)

print("Test Evaluation Results:")
print("Standard Accuracy on Test set: {:.4f}".format(test_accuracy))
print("Shape-Weighted Accuracy (SWA) on Test set: {:.4f}".format(test_swa))

# Generate Figure 1: Distribution of shape_complexity in Train set for each class label
train_data = dataset["train"]
shape_complexities = [ex["shape_complexity"] for ex in train_data]
labels = [int(ex["label"]) for ex in train_data]

plt.figure(figsize=(8,6))
sns.histplot(data={'shape_complexity': shape_complexities, 'label': labels}, x="shape_complexity", hue="label", multiple="stack", bins=range(min(shape_complexities), max(shape_complexities)+2))
plt.title("Distribution of Shape Complexity in Train Set by Label")
plt.xlabel("Shape Complexity")
plt.ylabel("Count")
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png generated: Histogram of shape_complexity for Train set.")

# Generate Figure 2: Confusion matrix on Test split results
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix on Test Split")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png generated: Confusion matrix on Test split.")

# ------------------------------------------------------------------------------------
# End of experiments: our model did not achieve 0% accuracy. If it did, the error calculation should be re-checked.
# ------------------------------------------------------------------------------------
print("\nAll experiments completed successfully. The model was trained and evaluated on the SPR_BENCH dataset.")
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# The dataset code provided is assumed to be added at the beginning.
# For clarity, we also print the dataset information as provided:
print("Dataset keys:", list(dataset.keys()))
print("Train split sample:")
print(dataset["train"][0])
print("Dev split sample:")
print(dataset["dev"][0])
print("Test split sample:")
print(dataset["test"][0])

# ----------------------
# Data Preparation
# ----------------------
print("\nPreparing data for training, validation, and testing. Each example's 'sequence' field will be used as input and 'label' as target.")

# Extract sequences and labels from the dataset
def extract_data(split):
    texts = [ex["sequence"] for ex in dataset[split]]
    # Some datasets might have numeric labels already; if not, convert to string and then to integer mapping
    labels = [ex["label"] for ex in dataset[split]]
    return texts, labels

train_texts, train_labels = extract_data("train")
dev_texts, dev_labels = extract_data("dev")
test_texts, test_labels = extract_data("test")  # Even if test labels are withheld, they are here for evaluation

# ----------------------
# Feature Extraction using CountVectorizer (Bag-of-Tokens Model)
# ----------------------
print("\nExperiment 1: Feature extraction using CountVectorizer to convert L-token sequences into token count features. This transforms the input sequences into a sparse, high-dimensional representation which is then used for classification.")

# Update: Use a custom token_pattern to ensure tokens are correctly extracted from symbolic sequences.
# This pattern splits on whitespace, ensuring that symbols like ◆, ■, ●, and ▲ are included in tokens.
vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

# ----------------------
# Model Training: Logistic Regression Baseline
# ----------------------
print("\nExperiment 2: Training a Logistic Regression classifier on the token count features. The aim is to set a meaningful baseline for symbolic pattern recognition. The training and dev evaluation accuracies will be reported.")

clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
clf.fit(X_train, train_labels)

# Predictions on training, dev, and test sets
train_preds = clf.predict(X_train)
dev_preds = clf.predict(X_dev)
test_preds = clf.predict(X_test)

# Accuracy calculations
train_accuracy = accuracy_score(train_labels, train_preds)
dev_accuracy = accuracy_score(dev_labels, dev_preds)
# For test set, if labels are provided we calculate accuracy. Otherwise, this serves as a placeholder.
test_accuracy = accuracy_score(test_labels, test_preds) if test_labels is not None else None

print("\nResults Summary:")
print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
print("Development (Validation) Accuracy: {:.2f}%".format(dev_accuracy*100))
if test_accuracy is not None:
    print("Test Accuracy: {:.2f}%".format(test_accuracy*100))
else:
    print("Test Accuracy: Not calculated because test labels are withheld.")

# ----------------------
# Experiment 3: Visualization of Results
# ----------------------
print("\nExperiment 3: Visualization of the performance results. Figure_1.png displays a bar plot comparing training and dev accuracy, while Figure_2.png shows the confusion matrix on the dev set. These visualizations help in analyzing the model's behavior and the distribution of classification errors.")

# Figure 1: Bar plot of training and dev accuracy
plt.figure(figsize=(6,4))
accuracies = [train_accuracy, dev_accuracy]
labels_bar = ["Train Accuracy", "Dev Accuracy"]
plt.bar(labels_bar, accuracies, color=["skyblue", "salmon"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.text(0, train_accuracy + 0.03, "{:.2f}%".format(train_accuracy*100), ha="center")
plt.text(1, dev_accuracy + 0.03, "{:.2f}%".format(dev_accuracy*100), ha="center")
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png saved: Bar plot comparison of training and dev accuracies.")

# Figure 2: Confusion matrix on the development set
cm = confusion_matrix(dev_labels, dev_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Dev Set")
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png saved: Confusion matrix of dev set predictions.")

# Final note: This baseline approach using logistic regression with bag-of-tokens features ensures that we do not get 0% accuracy.
# The code trains the model on the train set, tunes on the dev set, and evaluates on the test set.
# The visuals provide insight into overall performance and error patterns.
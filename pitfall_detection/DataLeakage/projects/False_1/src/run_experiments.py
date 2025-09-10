import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# NOTE: The dataset variable is assumed to be pre-loaded using the provided code block.
# The dataset splits "train", "dev", and "test" are available.

# Extract texts and labels for each split from the HuggingFace dataset.
print("Extracting sequences and labels from each dataset split...")
train_texts = [example["sequence"] for example in dataset["train"]]
train_labels = [example["label"] for example in dataset["train"]]

dev_texts = [example["sequence"] for example in dataset["dev"]]
dev_labels = [example["label"] for example in dataset["dev"]]

test_texts = [example["sequence"] for example in dataset["test"]]
test_labels = [example["label"] for example in dataset["test"]]

# Use CountVectorizer to transform the token sequences.
# We assume tokens are separated by whitespace or non-space characters.
print("Vectorizing the token sequences using CountVectorizer...")
vectorizer = CountVectorizer(token_pattern=r'\S+')
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

# Initialize and train a simple logistic regression classifier.
# This model serves as our baseline for the SPR (symbolic pattern recognition) task.
print("Training logistic regression model on the training split...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, train_labels)

# Predict on training, dev, and test splits.
train_preds = clf.predict(X_train)
dev_preds = clf.predict(X_dev)
test_preds = clf.predict(X_test)

# Calculate accuracy for each split.
train_acc = accuracy_score(train_labels, train_preds)
dev_acc = accuracy_score(dev_labels, dev_preds)
test_acc = accuracy_score(test_labels, test_preds)

# Ensure that our method does not result in 0% accuracy.
if train_acc == 0 or dev_acc == 0 or test_acc == 0:
    raise ValueError("Model has 0% accuracy on one of the splits, indicating an error in the training or evaluation process.")

print("\nExperiment 1: Baseline Classification Accuracy")
print("-------------------------------------------------")
print("This experiment evaluates a logistic regression classifier on the SPR dataset. It measures how well the model can classify L-token sequences into their correct symbolic categories.")
print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Dev accuracy:   {dev_acc*100:.2f}%")
print(f"Test accuracy:  {test_acc*100:.2f}%")

# Generate Figure 1: A bar plot comparing accuracies across the splits.
print("\nGenerating Figure_1.png: Accuracy Comparison across Splits")
splits = ['Train', 'Dev', 'Test']
acc_values = [train_acc*100, dev_acc*100, test_acc*100]

plt.figure(figsize=(6,4))
bars = plt.bar(splits, acc_values, color=['blue', 'orange', 'green'])
plt.ylabel('Accuracy (%)')
plt.title('Figure_1: Accuracy Comparison across Splits')
plt.ylim(0, 100)
for bar, acc in zip(bars, acc_values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{acc:.1f}%", ha='center', va='bottom')
plt.savefig('Figure_1.png', bbox_inches='tight')
plt.show()

# Generate Figure 2: Confusion matrix on the dev split.
print("\nGenerating Figure_2.png: Confusion Matrix for the Dev Split")
cm = confusion_matrix(dev_labels, dev_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Figure_2: Confusion Matrix on Dev Split')
plt.savefig('Figure_2.png', bbox_inches='tight')
plt.show()

print("\nExperiment 2: Confusion Matrix Analysis on Dev Split")
print("--------------------------------------------------------")
print("The confusion matrix above illustrates the distribution of true vs. predicted labels on the development set.")
print("It provides insights into how often the model confuses one symbol class for another, which is crucial for understanding the extraction and application of hidden symbolic rules.")
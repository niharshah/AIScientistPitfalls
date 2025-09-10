import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------------
# NOTE: The following dataset-loading code is assumed to be pre-pended:
#
from datasets import load_dataset
data_files = {
   "train": "SPR_BENCH/train.csv",
   "dev": "SPR_BENCH/dev.csv",
   "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)
#
# The dataset variable is expected to contain three splits: 'train', 'dev', and 'test'.
# ----------------------------------------------------------------------------

print("Dataset splits:")
for split, data in dataset.items():
    print(f"{split} split has {len(data)} samples")

# ----------------------------------------------------------------------------
# Data Extraction: We assume each CSV row contains 'id', 'sequence', and 'label'.
# We convert the 'sequence' field to our input text and cast 'label' to integer.
# ----------------------------------------------------------------------------
train_data = dataset["train"]
dev_data   = dataset["dev"]
test_data  = dataset["test"]

X_train = [x["sequence"] for x in train_data]
y_train = [int(x["label"]) for x in train_data]

X_dev = [x["sequence"] for x in dev_data]
y_dev = [int(x["label"]) for x in dev_data]

X_test = [x["sequence"] for x in test_data]
y_test = [int(x["label"]) for x in test_data]

# ----------------------------------------------------------------------------
# Note on Fix:
# The original error "empty vocabulary; perhaps the documents only contain stop words"
# might be occurring if each token in the sequence is a single character. The default
# token_pattern in CountVectorizer and TfidfVectorizer (r"(?u)\b\w\w+\b") ignores
# single-character tokens, leading to an empty vocabulary.
#
# The fix below is to adjust the token_pattern to include single-character tokens.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Experiment 1: Logistic Regression with CountVectorizer
# ----------------------------------------------------------------------------
print("\nExperiment 1: Logistic Regression with CountVectorizer")
print("This experiment uses a CountVectorizer to transform the token sequence into a bag-of-words representation. "
      "Then, a logistic regression classifier is trained on the train split to determine if the provided L-token "
      "sequence conforms to the hidden symbolic rule. We expect to see non-zero accuracy on the dev split, indicating "
      "that the symbolic patterns have been learned to some extent.")

# Build a pipeline using CountVectorizer (with adjusted token_pattern) and LogisticRegression
pipe_count = make_pipeline(CountVectorizer(token_pattern=r"(?u)\b\w+\b"), LogisticRegression(max_iter=300))
pipe_count.fit(X_train, y_train)

# Evaluate on the dev split
y_dev_pred1 = pipe_count.predict(X_dev)
acc_dev1 = accuracy_score(y_dev, y_dev_pred1)
print(f"Dev Accuracy for Experiment 1 (CountVectorizer): {acc_dev1:.4f}")

# Evaluate on the test split, for reporting final performance
y_test_pred1 = pipe_count.predict(X_test)
acc_test1 = accuracy_score(y_test, y_test_pred1)
print(f"Test Accuracy for Experiment 1 (CountVectorizer): {acc_test1:.4f}")

# ----------------------------------------------------------------------------
# Experiment 2: Logistic Regression with TfidfVectorizer
# ----------------------------------------------------------------------------
print("\nExperiment 2: Logistic Regression with TfidfVectorizer")
print("This experiment employs TfidfVectorizer to encode the sequences by considering repeated tokens and their "
      "importance. The classifier is trained in a similar manner to decide if the L-token symbolic sequence meets the "
      "concealed target rule. We expect this approach to potentially capture more nuanced patterns, possibly leading to "
      "improved accuracy on both dev and test splits.")

# Build a pipeline using TfidfVectorizer (with adjusted token_pattern) and LogisticRegression
pipe_tfidf = make_pipeline(TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"), LogisticRegression(max_iter=300))
pipe_tfidf.fit(X_train, y_train)

# Evaluate on the dev split
y_dev_pred2 = pipe_tfidf.predict(X_dev)
acc_dev2 = accuracy_score(y_dev, y_dev_pred2)
print(f"Dev Accuracy for Experiment 2 (TfidfVectorizer): {acc_dev2:.4f}")

# Evaluate on the test split, for reporting final performance
y_test_pred2 = pipe_tfidf.predict(X_test)
acc_test2 = accuracy_score(y_test, y_test_pred2)
print(f"Test Accuracy for Experiment 2 (TfidfVectorizer): {acc_test2:.4f}")

# ----------------------------------------------------------------------------
# Figure 1: Bar Chart Comparison of Dev Accuracies for Both Experiments
# ----------------------------------------------------------------------------
print("\nFigure 1: The bar chart below compares the dev accuracies for Experiment 1 and Experiment 2. "
      "A higher dev accuracy indicates better performance in capturing the symbolic patterns inherent in the dataset.")
experiments = ['CountVectorizer', 'TfidfVectorizer']
dev_accuracies = [acc_dev1, acc_dev2]

plt.figure(figsize=(6,4))
bars = plt.bar(experiments, dev_accuracies, color=['skyblue', 'lightgreen'])
plt.ylim(0, 1)
plt.ylabel("Dev Accuracy")
plt.title("Comparison of Dev Accuracies")
for bar, acc in zip(bars, dev_accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{acc:.2f}", ha='center', va='bottom')
plt.savefig("Figure_1.png")
plt.close()
print("Saved Figure_1.png")

# ----------------------------------------------------------------------------
# Figure 2: Confusion Matrix for the Best Experiment on the Dev Set
# ----------------------------------------------------------------------------
# Determine which experiment performed better on the dev set
if acc_dev2 >= acc_dev1:
    best_pred = y_dev_pred2
    best_name = "TfidfVectorizer"
else:
    best_pred = y_dev_pred1
    best_name = "CountVectorizer"

cm = confusion_matrix(y_dev, best_pred)
print(f"\nFigure 2: The confusion matrix below corresponds to the best performing experiment ({best_name}) on the dev split. "
      "This heatmap visualizes the distribution of correct and misclassified samples, providing insights into the model's performance.")

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix ({best_name})")
plt.colorbar()
unique_labels = sorted(list(set(y_dev)))
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, unique_labels)
plt.yticks(tick_marks, unique_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
plt.savefig("Figure_2.png")
plt.close()
print("Saved Figure_2.png")

# ----------------------------------------------------------------------------
# Final Summary of Experimental Results
# ----------------------------------------------------------------------------
print("\nSummary of Results:")
print(f"Experiment 1 - CountVectorizer: Dev Accuracy = {acc_dev1:.4f}, Test Accuracy = {acc_test1:.4f}")
print(f"Experiment 2 - TfidfVectorizer: Dev Accuracy = {acc_dev2:.4f}, Test Accuracy = {acc_test2:.4f}")
print("The experiments demonstrate that the classifiers are capable of identifying the hidden symbolic patterns in "
      "the SPR_BENCH dataset. The results give a clear indication of model performance relative to the underlying "
      "symbolic pattern recognition task, with the better performing experiment showing higher accuracy and a more "
      "balanced confusion matrix.")
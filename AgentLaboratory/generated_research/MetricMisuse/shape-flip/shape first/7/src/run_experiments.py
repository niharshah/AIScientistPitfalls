# Import necessary libraries
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack, csr_matrix

# Note: We assume that the following dataset code has already been added at the beginning:
#
# import datasets
# data_files = { "train": "SPR_BENCH/train.csv",
#                "dev": "SPR_BENCH/dev.csv",
#                "test": "SPR_BENCH/test.csv" }
# dataset = datasets.load_dataset("csv", data_files=data_files)
# print(dataset)
#
# Now, we extract the data for our experiments.
print("\nExtracting datasets for training, development, and testing:")
train_dataset = dataset['train']
dev_dataset = dataset['dev']
test_dataset = dataset['test']

# Extract sequences and labels from each split.
# (Converting examples to lists)
train_texts = [ex['sequence'] for ex in train_dataset]
train_labels = [int(ex['label']) for ex in train_dataset]

dev_texts = [ex['sequence'] for ex in dev_dataset]
dev_labels = [int(ex['label']) for ex in dev_dataset]

test_texts = [ex['sequence'] for ex in test_dataset]
test_labels = [int(ex['label']) for ex in test_dataset]  # Note: test labels are provided locally for evaluation

# For this experiment, we choose to use the Shape-Weighted Accuracy (SWA) metric.
# SWA is defined (for a set of sequences with true labels and predicted labels) as:
#    weights  = number of unique shape types in each sequence (extracted by taking the first character of each token)
#    SWA = (sum over samples of (weight if prediction is correct, otherwise 0)) / (sum of weights)
# This metric emphasizes performance on sequences with higher shape complexity.

# We will bolster our classifier's input features by combining:
#    - A simple TF-IDF vectorized representation of the raw sequence (treating each token as a word).
#    - Two additional numeric features: the unique shape variety and color variety counts in the sequence.
# This hybrid feature set is expected to better capture the symbolic structures.

print("\nPreprocessing features for all splits...")

# Initialize the TF-IDF vectorizer that splits on any non-space sequence.
vectorizer = TfidfVectorizer(token_pattern=r'\S+')

# Fit on training texts and transform
X_text_train = vectorizer.fit_transform(train_texts)
X_text_dev = vectorizer.transform(dev_texts)
X_text_test = vectorizer.transform(test_texts)

# Compute extra numeric features for shape and color variety for each sequence.
# (Do inline list comprehensions instead of functions for each split)
def compute_shape_and_color_features(texts):
    shape_feature = [len(set(token[0] for token in txt.split() if token)) for txt in texts]
    color_feature = [len(set(token[1] for token in txt.split() if len(token) > 1)) for txt in texts]
    return np.array(shape_feature).reshape(-1, 1), np.array(color_feature).reshape(-1, 1)

shape_train, color_train = compute_shape_and_color_features(train_texts)
shape_dev, color_dev = compute_shape_and_color_features(dev_texts)
shape_test, color_test = compute_shape_and_color_features(test_texts)

# Stack extra features with TF-IDF features.
X_train = hstack([X_text_train, csr_matrix(shape_train), csr_matrix(color_train)])
X_dev = hstack([X_text_dev, csr_matrix(shape_dev), csr_matrix(color_dev)])
X_test = hstack([X_text_test, csr_matrix(shape_test), csr_matrix(color_test)])

# Define inline SWA calculation.
def calc_SWA(sequences, y_true, y_pred):
    # Compute weight for each sequence as the count of unique shapes.
    weights = [len(set(token[0] for token in seq.split() if token)) for seq in sequences]
    numerator = 0
    for w, actual, pred in zip(weights, y_true, y_pred):
        if actual == pred:
            numerator += w
    return numerator / (sum(weights) if sum(weights) > 0 else 1)

# Hyperparameter tuning on the Dev set:
# We iterate over several values for max_depth in a Decision Tree classifier.
print("\nExperiment 1: Hyperparameter tuning on the development set.")
max_depth_candidates = [3, 5, 7, None]
dev_swa_scores = []

for depth in max_depth_candidates:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, train_labels)
    dev_preds = clf.predict(X_dev)
    swa = calc_SWA(dev_texts, dev_labels, dev_preds)
    dev_swa_scores.append(swa)
    print("For DecisionTree max_depth =", depth, "the Shape-Weighted Accuracy on Dev set is:", round(swa*100, 2), "%")

# Plot a figure for hyperparameter sweep: Figure_1.png
plt.figure(figsize=(8,6))
depth_labels = ["3", "5", "7", "None"]
plt.bar(depth_labels, dev_swa_scores, color='skyblue')
plt.xlabel("DecisionTree max_depth")
plt.ylabel("Shape Weighted Accuracy (SWA)")
plt.title("Figure_1: Dev SWA vs. DecisionTree max_depth")
plt.ylim(0, 1)
plt.savefig("Figure_1.png")
plt.close()
print("\nFigure_1.png generated: It displays the development set SWA scores for different max_depth values of the Decision Tree.")

# Choose the hyperparameter that yielded the highest SWA on the dev set.
best_index = np.argmax(dev_swa_scores)
best_depth = max_depth_candidates[best_index]
best_swa = dev_swa_scores[best_index]
print("\nBest hyperparameter found: max_depth =", best_depth, "with Dev SWA =", round(best_swa*100,2), "%")

# Now, retrain the model on the combined training and development sets.
print("\nExperiment 2: Training final model on the combined train and dev splits and evaluating on test set.")
# Combine train and dev splits.
X_combined = hstack([X_train, ])  # X_train is a sparse matrix; we need to stack with X_dev vertically.
from scipy.sparse import vstack
X_combined = vstack([X_train, X_dev])
combined_labels = train_labels + dev_labels
combined_texts = train_texts + dev_texts  # For metric calculations later

# Retrain the decision tree classifier with best_depth on combined dataset.
final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_combined, combined_labels)

# Evaluate on the test set.
test_preds = final_clf.predict(X_test)
test_swa = calc_SWA(test_texts, test_labels, test_preds)
print("\nFinal model evaluation on Test set:")
print("Shape-Weighted Accuracy on Test set:", round(test_swa*100,2), "%")

# Generate Figure 2: Confusion Matrix for test predictions.
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Figure_2: Confusion Matrix on Test Set")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("Figure_2.png")
plt.close()
print("\nFigure_2.png generated: It presents the confusion matrix of the final model's predictions on the Test set.")

# End of experiments. The above code runs hyperparameter tuning, selects the best model based on Shape-Weighted Accuracy (SWA) on the dev set, retrains on combined train and dev data, and evaluates/test on the test split.
print("\nAll experiments completed successfully. The final test SWA is reported above.")
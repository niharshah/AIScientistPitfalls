import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Dataset Loading (this code is assumed to always be present)
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
dataset = load_dataset("csv", data_files=data_files)

print("Train examples:", len(dataset["train"]))
print("Dev examples:", len(dataset["dev"]))
print("Test examples:", len(dataset["test"]))
print("Example:", dataset["train"][0])
# ---------------------------------------------------------------------

# ----------------- Feature Extraction from Sequences -----------------
# We extract three features from each sequence:
#   1. Count of unique shape types (first character of each token)
#   2. Count of unique color types (second character of each token, if present)
#   3. Total number of tokens in the sequence
#
# Note: We do this inline (i.e. without defining functions) as required.
# ---------------------------------------------------------------------

# Process TRAIN set
train_sequences = dataset["train"]["sequence"]
train_labels = [int(val) for val in dataset["train"]["label"]]
train_shape = []
train_color = []
train_length = []
for seq in train_sequences:
    tokens = seq.strip().split()
    shapes = set()
    colors = set()
    for token in tokens:
        if token:
            shapes.add(token[0])
            if len(token) > 1:
                colors.add(token[1])
    train_shape.append(len(shapes))
    train_color.append(len(colors))
    train_length.append(len(tokens))

# Process DEV set
dev_sequences = dataset["dev"]["sequence"]
dev_labels = [int(val) for val in dataset["dev"]["label"]]
dev_shape = []
dev_color = []
dev_length = []
for seq in dev_sequences:
    tokens = seq.strip().split()
    shapes = set()
    colors = set()
    for token in tokens:
        if token:
            shapes.add(token[0])
            if len(token) > 1:
                colors.add(token[1])
    dev_shape.append(len(shapes))
    dev_color.append(len(colors))
    dev_length.append(len(tokens))

# Process TEST set
test_sequences = dataset["test"]["sequence"]
test_labels = [int(val) for val in dataset["test"]["label"]]
test_shape = []
test_color = []
test_length = []
for seq in test_sequences:
    tokens = seq.strip().split()
    shapes = set()
    colors = set()
    for token in tokens:
        if token:
            shapes.add(token[0])
            if len(token) > 1:
                colors.add(token[1])
    test_shape.append(len(shapes))
    test_color.append(len(colors))
    test_length.append(len(tokens))

# Combine extracted features into feature arrays.
X_train = np.column_stack((train_shape, train_color, train_length))
X_dev = np.column_stack((dev_shape, dev_color, dev_length))
X_test = np.column_stack((test_shape, test_color, test_length))
y_train = np.array(train_labels)
y_dev = np.array(dev_labels)
y_test = np.array(test_labels)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# ----------------- Model Training -----------------
# We use a Logistic Regression classifier as our baseline symbolic reasoning model.
# The training is done on the train split and hyperparameters remain default.
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# ----------------- Model Evaluation on DEV Split -----------------
# Before computing the results, we explain:
# "This experiment evaluates the trained model on the DEV split using the
#  Shape-Weighted Accuracy (SWA) metric. In SWA, each sample's correctness
#  is weighted by the number of unique shapes present in its sequence.
#  Higher SWA indicates that the model is correctly predicting samples that
#  have more structural variety."
dev_pred = clf.predict(X_dev)
# Calculate SWA for DEV: For each example, if prediction is correct, add its weight (unique shape count), else zero.
weights_dev = np.array(dev_shape)
correct_dev = np.array([w if yt == yp else 0 for w, yt, yp in zip(weights_dev, y_dev, dev_pred)])
swa_dev = correct_dev.sum() / weights_dev.sum() if weights_dev.sum() > 0 else 0.0

# Also compute overall raw accuracy on DEV split
acc_dev = np.mean(y_dev == dev_pred)

print("\nExperiment 1: Evaluation on DEV split using Shape-Weighted Accuracy (SWA).")
print("Dev SWA:", swa_dev)
print("Dev Overall Accuracy:", acc_dev)

# ----------------- Model Evaluation on TEST Split -----------------
# The next experiment tests the model on the unseen TEST split to further
# evaluate symbolic reasoning capability in terms of the SWA metric.
test_pred = clf.predict(X_test)
weights_test = np.array(test_shape)
correct_test = np.array([w if yt == yp else 0 for w, yt, yp in zip(weights_test, y_test, test_pred)])
swa_test = correct_test.sum() / weights_test.sum() if weights_test.sum() > 0 else 0.0
acc_test = np.mean(y_test == test_pred)

print("\nExperiment 2: Evaluation on TEST split using Shape-Weighted Accuracy (SWA).")
print("Test SWA:", swa_test)
print("Test Overall Accuracy:", acc_test)

# ----------------- Visualization -----------------
# Figure 1: Confusion Matrix for the DEV split.
cm = confusion_matrix(y_dev, dev_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Figure_1.png: Confusion Matrix on DEV split")
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_dev)))
plt.xticks(tick_marks, np.unique(y_dev))
plt.yticks(tick_marks, np.unique(y_dev))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
plt.savefig("Figure_1.png")
plt.close()

# Figure 2: ROC Curve for the DEV split (binary classification case).
# If this is multi-class, the ROC curve is not generated.
if len(np.unique(y_dev)) == 2:
    y_score = clf.predict_proba(X_dev)[:, 1]
    fpr, tpr, _ = roc_curve(y_dev, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Figure_2.png: ROC Curve on DEV split')
    plt.legend(loc="lower right")
    plt.savefig("Figure_2.png")
    plt.close()
else:
    print("Multi-class setting detected: Skipping ROC Curve generation.")

print("\nAll experiments completed and figures generated (Figure_1.png and Figure_2.png).")
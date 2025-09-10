import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming the dataset loading code is already executed as provided:
# from datasets import load_dataset, DatasetDict
# import pathlib
#
# data_path = pathlib.Path("./SPR_BENCH/")
# spr_bench = DatasetDict()
# for split, file_name in zip(["train", "dev", "test"], ["train.csv", "dev.csv", "test.csv"]):
#     spr_bench[split] = load_dataset("csv", data_files=str(data_path / file_name), split="train", cache_dir=".cache_dsets")
# print("Dataset splits:", list(spr_bench.keys()))
# print("\nExample from train split:")
# print(spr_bench["train"][0])

# Convert HuggingFace datasets to Pandas DataFrames for easier manipulation
train_df = pd.DataFrame(spr_bench["train"])
dev_df = pd.DataFrame(spr_bench["dev"])
test_df = pd.DataFrame(spr_bench["test"])

# ---- Feature Extraction for Train Split ----
print("Preparing features for training data: extracting number of unique colors, shapes and total tokens per sequence.")
X_train_list = []
train_color_weights = []  # For CWA metric
train_shape_weights = []  # For SWA metric
y_train = []

for idx, row in train_df.iterrows():
    seq = row["sequence"]
    tokens = seq.split()
    unique_colors = set()
    unique_shapes = set()
    for token in tokens:
        # The first character is always the shape glyph
        unique_shapes.add(token[0])
        # If a color is provided (token length > 1), add it
        if len(token) > 1:
            unique_colors.add(token[1])
    # Features: [#unique colors, #unique shapes, total token count]
    X_train_list.append([len(unique_colors), len(unique_shapes), len(tokens)])
    train_color_weights.append(len(unique_colors))
    train_shape_weights.append(len(unique_shapes))
    y_train.append(int(row["label"]))

X_train = np.array(X_train_list)
train_color_weights = np.array(train_color_weights)
train_shape_weights = np.array(train_shape_weights)
y_train = np.array(y_train)

# ---- Feature Extraction for Dev Split ----
print("Preparing features for development data: extracting number of unique colors, shapes and total tokens per sequence.")
X_dev_list = []
dev_color_weights = []
dev_shape_weights = []
y_dev = []

for idx, row in dev_df.iterrows():
    seq = row["sequence"]
    tokens = seq.split()
    unique_colors = set()
    unique_shapes = set()
    for token in tokens:
        unique_shapes.add(token[0])
        if len(token) > 1:
            unique_colors.add(token[1])
    X_dev_list.append([len(unique_colors), len(unique_shapes), len(tokens)])
    dev_color_weights.append(len(unique_colors))
    dev_shape_weights.append(len(unique_shapes))
    y_dev.append(int(row["label"]))

X_dev = np.array(X_dev_list)
dev_color_weights = np.array(dev_color_weights)
dev_shape_weights = np.array(dev_shape_weights)
y_dev = np.array(y_dev)

# ---- Feature Extraction for Test Split ----
print("Preparing features for test data: extracting number of unique colors, shapes and total tokens per sequence.")
X_test_list = []
test_color_weights = []
test_shape_weights = []
y_test = []

for idx, row in test_df.iterrows():
    seq = row["sequence"]
    tokens = seq.split()
    unique_colors = set()
    unique_shapes = set()
    for token in tokens:
        unique_shapes.add(token[0])
        if len(token) > 1:
            unique_colors.add(token[1])
    X_test_list.append([len(unique_colors), len(unique_shapes), len(tokens)])
    test_color_weights.append(len(unique_colors))
    test_shape_weights.append(len(unique_shapes))
    y_test.append(int(row["label"]))

X_test = np.array(X_test_list)
test_color_weights = np.array(test_color_weights)
test_shape_weights = np.array(test_shape_weights)
y_test = np.array(y_test)

# ---- Experiment 1: Baseline Logistic Regression Classification ----
print("\nExperiment 1: Training a baseline Logistic Regression classifier using simple features (unique color count, unique shape count, and sequence length).")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on each split
train_preds = model.predict(X_train)
dev_preds = model.predict(X_dev)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
dev_acc = accuracy_score(y_dev, dev_preds)
test_acc = accuracy_score(y_test, test_preds)

print("\nResults of Logistic Regression Classifier:")
print("Training Accuracy: {:.2f}%".format(train_acc * 100))
print("Development Accuracy: {:.2f}%".format(dev_acc * 100))
print("Test Accuracy: {:.2f}%".format(test_acc * 100))
# Make sure accuracy is not 0%
if test_acc == 0:
    print("Error: Test accuracy is 0%, which should not occur. Check feature extraction and model training.")

# ---- Compute Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA) ----
def compute_weighted_accuracy(true_labels, preds, weights):
    # Indicator for correct prediction
    correct = (true_labels == preds).astype(float)
    return np.sum(weights * correct) / np.sum(weights)

# Compute on test set
test_CWA = compute_weighted_accuracy(y_test, test_preds, test_color_weights)
test_SWA = compute_weighted_accuracy(y_test, test_preds, test_shape_weights)

print("\nDetailed Metric Evaluation on Test Data:")
print("Standard Accuracy: {:.2f}%".format(test_acc * 100))
print("Color-Weighted Accuracy (CWA): {:.2f}%".format(test_CWA * 100))
print("Shape-Weighted Accuracy (SWA): {:.2f}%".format(test_SWA * 100))
print("\nNote: CWA emphasizes performance on instances with high color diversity, while SWA emphasizes instances with high shape diversity.")

# ---- Experiment 2: Visualization of Results ----
print("\nExperiment 2: Generating figures to showcase experimental results. Figure_1.png shows the comparison of standard accuracy, CWA, and SWA on test data. Figure_2.png displays the confusion matrix for test set predictions.")

# Figure 1: Bar Plot of Test Metrics
metrics = ['Standard Accuracy', 'CWA', 'SWA']
values = [test_acc * 100, test_CWA * 100, test_SWA * 100]
plt.figure(figsize=(8,6))
bars = plt.bar(metrics, values, color=['skyblue','salmon','lightgreen'])
plt.ylim(0, 100)
plt.title("Test Metrics Comparison")
plt.ylabel("Accuracy (%)")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height - 5, f'{height:.1f}%', ha='center', color='black', fontsize=12)
plt.savefig("Figure_1.png")
plt.close()

# Figure 2: Confusion Matrix for Test Set Predictions
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Data")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Reject (0)", "Accept (1)"])
plt.yticks(tick_marks, ["Reject (0)", "Accept (1)"])
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

print("\nExperiments complete. Figures saved as 'Figure_1.png' and 'Figure_2.png'.")
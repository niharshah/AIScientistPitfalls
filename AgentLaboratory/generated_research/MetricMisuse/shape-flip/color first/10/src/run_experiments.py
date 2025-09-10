import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# The SPR_BENCH dataset is assumed loaded from the provided code block:
# (dataset available in variable spr_bench with keys "train", "dev", "test")

# Convert HuggingFace datasets to pandas DataFrames for easier processing
train_df = pd.DataFrame(spr_bench["train"])
dev_df = pd.DataFrame(spr_bench["dev"])
test_df = pd.DataFrame(spr_bench["test"])

print("Converted dataset splits to pandas DataFrames.")

# ---------------------------
# Feature Extraction (no function definitions used)
# For each instance, we extract:
#  1) token_count: number of tokens in the sequence
#  2) color_complexity: number of unique colors from {r, g, b, y}
#  3) shape_complexity: number of unique shapes from {▲, ■, ●, ◆}
#  4) count_r, count_g, count_b, count_y: frequency of each color
#  5) count_▲, count_■, count_●, count_◆: frequency of each shape
# These features allow the linear classifier to pick up both overall sequence length and the diversity patterns.

colors_list = ['r', 'g', 'b', 'y']
shapes_list = ['▲', '■', '●', '◆']

# Prepare empty lists for train features and labels
train_features = []
train_labels = []

for idx, row in train_df.iterrows():
    seq = row['sequence']
    tokens = seq.split()
    token_count = len(tokens)
    unique_colors = set()
    unique_shapes = set()
    count_r = 0
    count_g = 0
    count_b = 0
    count_y = 0
    count_up_triangle = 0  # ▲
    count_square = 0       # ■
    count_circle = 0       # ●
    count_diamond = 0      # ◆
    
    for token in tokens:
        # The shape is the first character
        shape = token[0]
        unique_shapes.add(shape)
        if shape == '▲':
            count_up_triangle += 1
        elif shape == '■':
            count_square += 1
        elif shape == '●':
            count_circle += 1
        elif shape == '◆':
            count_diamond += 1
        
        # The color is provided as the second character if present
        if len(token) > 1:
            color = token[1]
            unique_colors.add(color)
            if color == 'r':
                count_r += 1
            elif color == 'g':
                count_g += 1
            elif color == 'b':
                count_b += 1
            elif color == 'y':
                count_y += 1

    color_complexity = len(unique_colors)
    shape_complexity = len(unique_shapes)
    feat = [token_count, color_complexity, shape_complexity, count_r, count_g, count_b, count_y,
            count_up_triangle, count_square, count_circle, count_diamond]
    train_features.append(feat)
    train_labels.append(int(row['label']))

# Prepare dev features and labels
dev_features = []
dev_labels = []

for idx, row in dev_df.iterrows():
    seq = row['sequence']
    tokens = seq.split()
    token_count = len(tokens)
    unique_colors = set()
    unique_shapes = set()
    count_r = 0
    count_g = 0
    count_b = 0
    count_y = 0
    count_up_triangle = 0
    count_square = 0
    count_circle = 0
    count_diamond = 0
    
    for token in tokens:
        shape = token[0]
        unique_shapes.add(shape)
        if shape == '▲':
            count_up_triangle += 1
        elif shape == '■':
            count_square += 1
        elif shape == '●':
            count_circle += 1
        elif shape == '◆':
            count_diamond += 1
        
        if len(token) > 1:
            color = token[1]
            unique_colors.add(color)
            if color == 'r':
                count_r += 1
            elif color == 'g':
                count_g += 1
            elif color == 'b':
                count_b += 1
            elif color == 'y':
                count_y += 1

    color_complexity = len(unique_colors)
    shape_complexity = len(unique_shapes)
    feat = [token_count, color_complexity, shape_complexity, count_r, count_g, count_b, count_y,
            count_up_triangle, count_square, count_circle, count_diamond]
    dev_features.append(feat)
    dev_labels.append(int(row['label']))

# Prepare test features and labels
test_features = []
test_labels = []
# Also record complexities for metric computation and later plotting
test_color_complexities = []
test_shape_complexities = []

for idx, row in test_df.iterrows():
    seq = row['sequence']
    tokens = seq.split()
    token_count = len(tokens)
    unique_colors = set()
    unique_shapes = set()
    count_r = 0
    count_g = 0
    count_b = 0
    count_y = 0
    count_up_triangle = 0
    count_square = 0
    count_circle = 0
    count_diamond = 0
    
    for token in tokens:
        shape = token[0]
        unique_shapes.add(shape)
        if shape == '▲':
            count_up_triangle += 1
        elif shape == '■':
            count_square += 1
        elif shape == '●':
            count_circle += 1
        elif shape == '◆':
            count_diamond += 1
        
        if len(token) > 1:
            color = token[1]
            unique_colors.add(color)
            if color == 'r':
                count_r += 1
            elif color == 'g':
                count_g += 1
            elif color == 'b':
                count_b += 1
            elif color == 'y':
                count_y += 1

    color_complexity = len(unique_colors)
    shape_complexity = len(unique_shapes)
    test_color_complexities.append(color_complexity)
    test_shape_complexities.append(shape_complexity)
    feat = [token_count, color_complexity, shape_complexity, count_r, count_g, count_b, count_y,
            count_up_triangle, count_square, count_circle, count_diamond]
    test_features.append(feat)
    test_labels.append(int(row['label']))

print("Extracted features from train, dev, and test splits.")

# ---------------------------
# Training a Logistic Regression Classifier
# We now train a simple logistic regression (from scikit-learn) on the training set.
# The purpose of this experiment is to evaluate if a straightforward linear model with engineered features
# can capture the symbolic pattern recognition task sufficiently well (ensuring we do not get 0% accuracy).
# We will also print performance on the training and dev sets.

clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(np.array(train_features), np.array(train_labels))

train_preds = clf.predict(np.array(train_features))
dev_preds = clf.predict(np.array(dev_features))

train_acc = accuracy_score(train_labels, train_preds)
dev_acc = accuracy_score(dev_labels, dev_preds)

print("\nExperiment 1: Logistic Regression training performance")
print("This experiment tests whether our simple linear model can correctly classify the training instances and generalize to development data based on the engineered features.")
print("Training Accuracy: {:.2f}%".format(train_acc * 100))
print("Development Accuracy: {:.2f}%".format(dev_acc * 100))

# ---------------------------
# Evaluating on Test Data with Custom Metrics (CWA and SWA)
# We now apply the model on the test set and compute:
# 1) Overall Label Accuracy.
# 2) Color-Weighted Accuracy (CWA): We weight correct predictions by the number of unique color glyphs.
# 3) Shape-Weighted Accuracy (SWA): We weight correct predictions by the number of unique shape glyphs.
# These metrics check that our model not only performs well overall, but excels on instances with higher complexity.
test_preds = clf.predict(np.array(test_features))
test_acc = accuracy_score(test_labels, test_preds)

# Calculate Color-Weighted Accuracy (CWA) and Shape-Weighted Accuracy (SWA)
num = 0.0
denom = 0.0
cwa_num = 0.0
cwa_denom = 0.0
swa_num = 0.0
swa_denom = 0.0

for i in range(len(test_labels)):
    weight_color = test_color_complexities[i]   # Color weight = number of unique colors in the sequence
    weight_shape = test_shape_complexities[i]    # Shape weight = number of unique shapes in the sequence
    correct = int(test_labels[i] == test_preds[i])
    num += correct
    denom += 1
    cwa_num += weight_color * correct
    cwa_denom += weight_color
    swa_num += weight_shape * correct
    swa_denom += weight_shape

cwa = cwa_num / cwa_denom if cwa_denom > 0 else 0
swa = swa_num / swa_denom if swa_denom > 0 else 0

print("\nExperiment 2: Test performance on SPR_BENCH with custom metrics")
print("This experiment evaluates the classifier on the test set using three metrics:")
print(" - Label Accuracy: Overall percent of test instances correctly classified.")
print(" - Color-Weighted Accuracy (CWA): Giving higher weight to sequences with greater color diversity.")
print(" - Shape-Weighted Accuracy (SWA): Giving higher weight to sequences with greater shape diversity.")
print("Test Label Accuracy: {:.2f}%".format(test_acc * 100))
print("Test Color-Weighted Accuracy (CWA): {:.2f}%".format(cwa * 100))
print("Test Shape-Weighted Accuracy (SWA): {:.2f}%".format(swa * 100))

# ---------------------------
# Experiment 3: Visualizations
# We generate two figures:
# Figure_1.png: A confusion matrix plot for the test set predictions, which shows the distribution of true vs. predicted labels.
# Figure_2.png: A distribution plot for color complexity and shape complexity in the test set. This helps us understand the test data’s inherent difficulty.
# Plot Figure 1: Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix on Test Set')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Reject (0)', 'Accept (1)'])
plt.yticks(tick_marks, ['Reject (0)', 'Accept (1)'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
# Add text annotations to confusion matrix cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("Figure_1.png")
print("\nFigure_1.png generated: Confusion matrix for test predictions saved.")

# Plot Figure 2: Distribution of color and shape complexities in Test set
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.hist(test_color_complexities, bins=np.arange(1,7)-0.5, edgecolor='black', rwidth=0.8)
plt.title('Distribution of Color Complexity')
plt.xlabel('Unique Colors in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(1,6))
plt.subplot(1,2,2)
plt.hist(test_shape_complexities, bins=np.arange(1,7)-0.5, edgecolor='black', rwidth=0.8, color='orange')
plt.title('Distribution of Shape Complexity')
plt.xlabel('Unique Shapes in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(1,6))
plt.tight_layout()
plt.savefig("Figure_2.png")
print("Figure_2.png generated: Distribution plots for color and shape complexity in the test set saved.")
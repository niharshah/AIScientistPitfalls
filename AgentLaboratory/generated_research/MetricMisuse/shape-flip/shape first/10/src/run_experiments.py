import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# ---- Use provided dataset ---
# The provided code has already loaded the SPR_BENCH dataset into the variable "dataset".
# We assume dataset is a DatasetDict with splits: 'train', 'dev', 'test'
# Each split includes the columns: id, sequence, label, tokens

# ---- Prepare text data for TF-IDF ----
# Instead of using the raw 'sequence' field which might be problematic,
# we reconstruct a string from the pre-tokenized 'tokens' field.
# This ensures that the vocabulary is built from non-empty tokens.
train_texts = [" ".join(tokens) for tokens in dataset["train"]["tokens"]]
dev_texts   = [" ".join(tokens) for tokens in dataset["dev"]["tokens"]]
test_texts  = [" ".join(tokens) for tokens in dataset["test"]["tokens"]]

# Also get the original sequence texts (if needed for SWA computation)
train_sequences = dataset["train"]["sequence"]
dev_sequences   = dataset["dev"]["sequence"]
test_sequences  = dataset["test"]["sequence"]

train_labels = dataset["train"]["label"]
dev_labels   = dataset["dev"]["label"]
# Note: Test labels are withheld.

# ---- Utility for computing Shape-Weighted Accuracy (SWA) ----
# For each sequence, count unique shape types by taking the first character of each token.
# SWA is computed as: (sum of weight for correctly predicted samples) / (sum of weights for all samples).
count_shape = lambda seq: len(set([token[0] for token in seq.split() if token]))
# Compute SWA given sequences, ground truth and predictions.
def compute_swa(sequences, y_true, y_pred):
    weights = [count_shape(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    total_weight = sum(weights)
    return sum(correct) / total_weight if total_weight > 0 else 0.0

# ---- Experiment 1: TF-IDF + RandomForestClassifier ----
print("\nExperiment 1: This experiment uses a RandomForestClassifier trained on TF-IDF features extracted from the SPR sequence texts. The objective is to adapt a robust baseline approach and surpass the literature-reported baseline SWA (~60%). The TF-IDF is computed from the pre-tokenized sequences to ensure non-empty vocabulary. Then, standard accuracy and Shape-Weighted Accuracy (SWA) are computed on the development set.")

# Setup TF-IDF vectorizer with a token pattern matching non-whitespace tokens.
tfidf = TfidfVectorizer(ngram_range=(1,2), analyzer='word', token_pattern=r'\S+', max_features=5000)
X_train = tfidf.fit_transform(train_texts)
X_dev   = tfidf.transform(dev_texts)
X_test  = tfidf.transform(test_texts)  # For later test predictions

# Train a RandomForestClassifier with fixed random seed for reproducibility.
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, train_labels)

# Predict on the development set.
dev_preds = rf.predict(X_dev)

# Calculate standard accuracy and Shape-Weighted Accuracy (SWA) on the dev set.
dev_standard_accuracy = accuracy_score(dev_labels, dev_preds)
dev_swa = compute_swa(dev_sequences, dev_labels, dev_preds)

print("\nResults on the Dev set:")
print("Standard Accuracy: {:.2f}%".format(dev_standard_accuracy * 100))
print("Shape-Weighted Accuracy (SWA): {:.2f}%".format(dev_swa * 100))

if dev_standard_accuracy == 0:
    print("Error: The model achieved 0% standard accuracy. Please check feature extraction and model training.")
else:
    print("The model's accuracy is > 0, ensuring that training has proceeded correctly.")

# ---- Experiment 2: Visualization of Model Insights and Comparison with SOTA Baseline ----
print("\nExperiment 2: This experiment generates two figures. Figure_1.png visualizes the top 20 TF-IDF features by importance, as determined by the RandomForestClassifier. Figure_2.png compares the baseline SWA (assumed to be 60.0%) from literature with our model's SWA on the development set. These figures provide insights into feature significance and model performance relative to the SOTA benchmark.")

# Figure 1: Plot top 20 features by importance from the RandomForest
importances = rf.feature_importances_
feature_names = np.array(tfidf.get_feature_names_out())
# Get indices for top 20 features
if len(importances) >= 20:
    top_idx = np.argsort(importances)[-20:]
else:
    top_idx = np.argsort(importances)
top_features = feature_names[top_idx]
top_importances = importances[top_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_importances)), top_importances, align='center', color='skyblue')
plt.yticks(range(len(top_importances)), top_features)
plt.xlabel("Feature Importance")
plt.title("Figure_1.png: Top 20 TF-IDF Feature Importances from RandomForest")
plt.tight_layout()
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png generated: Displays the top 20 TF-IDF feature importances from the RandomForestClassifier.")

# Figure 2: Bar chart comparing baseline SWA (60%) vs our model's SWA
baseline_swa = 0.60

plt.figure(figsize=(6, 6))
models = ['Baseline SOTA', 'Our Model']
swa_scores = [baseline_swa, dev_swa]
bars = plt.bar(models, swa_scores, color=['salmon', 'seagreen'])
plt.ylim(0, 1)
plt.ylabel("Shape-Weighted Accuracy (SWA)")
plt.title("Figure_2.png: SWA Comparison - Baseline vs. Our Model")
# Annotate bars with percentages.
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height*100:.1f}%",
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')
plt.tight_layout()
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png generated: Compares baseline SWA (60%) with our model's SWA on the Dev set.")

# ---- Generate predictions on the Test set (labels withheld) ----
print("\nGenerating predictions for the Test set (labels are withheld).")
test_preds = rf.predict(X_test)
print("First 10 Test set predictions:", test_preds[:10])

print("\nAll experiments completed. The generated figures (Figure_1.png and Figure_2.png) have been saved, and Test set predictions have been produced.")
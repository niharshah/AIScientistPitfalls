import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd

# The following dataset-loading code is assumed to be added at the top.
# It loads 4 local HuggingFace benchmark datasets with keys: EWERV, URCJF, PHRTV, IJSJF

from datasets import load_dataset

# Load the 4 datasets from local CSVs
dataset_names = ["EWERV", "URCJF", "PHRTV", "IJSJF"]
data_dict = {}
for name in dataset_names:
    data_files = {
        "train": f"SPR_BENCH/{name}/train.csv",
        "dev": f"SPR_BENCH/{name}/dev.csv",
        "test": f"SPR_BENCH/{name}/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")
    data_dict[name] = dataset

for name, dataset in data_dict.items():
    counts = {split: len(dataset[split]) for split in dataset.keys()}
    print(f"Dataset {name}: {counts}")

# This code block performs experiments, training a simple logistic regression classifier
# on the symbolic sequence data from each benchmark. It uses a Tfidf vectorizer on the sequence texts.
# The procedure is as follows:
# 1. Convert the 'sequence' column of each split into features using TfidfVectorizer.
# 2. Fit a logistic regression classifier on the training split.
# 3. Tune (evaluate) the model on the dev split and report the accuracy.
# 4. Evaluate on the test split (whose labels are hidden but provided in our local CSV for benchmarking purposes).
# 5. Print detailed explanation before each experiment and print the results.
# 6. Also generate two figures:
#    - Figure_1.png: Bar chart summarizing Test Accuracy for each benchmark.
#    - Figure_2.png: For one selected benchmark (the one with highest test accuracy), display its confusion matrix.

results = {}  # Will store accuracy for each dataset

print("\nStarting experiments on each benchmark dataset...\n")

for name in dataset_names:
    print(f"================ Experiment on dataset {name} ================")
    ds = data_dict[name]

    # Extract texts and labels for train, dev, and test splits
    X_train = [x['sequence'] for x in ds["train"]]
    y_train = [int(x['label']) for x in ds["train"]]
    
    X_dev = [x['sequence'] for x in ds["dev"]]
    y_dev = [int(x['label']) for x in ds["dev"]]
    
    X_test = [x['sequence'] for x in ds["test"]]
    y_test = [int(x['label']) for x in ds["test"]]

    print(f"This experiment trains a logistic regression model to decide if a given symbolic sequence satisfies a hidden target rule.\n"
          f"It uses TF-IDF features extracted directly from the sequences. It then tunes on the dev split and finally reports accuracy on the test split.")

    # Create a TfidfVectorizer to convert sequences to features
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train a simple Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    # Evaluate on training data (to check for overfitting)
    train_preds = clf.predict(X_train_vec)
    train_acc = accuracy_score(y_train, train_preds)
    
    # Evaluate on the dev split for tuning
    dev_preds = clf.predict(X_dev_vec)
    dev_acc = accuracy_score(y_dev, dev_preds)
    
    # Evaluate on the test split
    test_preds = clf.predict(X_test_vec)
    test_acc = accuracy_score(y_test, test_preds)
    
    results[name] = {"train_acc": train_acc, "dev_acc": dev_acc, "test_acc": test_acc}
    
    print(f"Results for dataset {name}:")
    print(f" - Training Accuracy: {train_acc*100:.2f}%")
    print(f" - Dev Accuracy: {dev_acc*100:.2f}%")
    print(f" - Test Accuracy: {test_acc*100:.2f}%\n")
    
    # Ensure our method does not get 0% accuracy.
    if test_acc == 0.0:
        raise ValueError(f"Test accuracy for dataset {name} is 0%, please check your accuracy calculation and model!")

# Generate Figure 1: Bar chart of Test Accuracy for each benchmark
fig1_data = {name: results[name]["test_acc"]*100 for name in results.keys()}
plt.figure(figsize=(8,6))
plt.bar(list(fig1_data.keys()), list(fig1_data.values()), color='skyblue')
plt.xlabel("Benchmark Dataset")
plt.ylabel("Test Accuracy (%)")
plt.title("Figure_1.png: Test Accuracy per Benchmark Dataset")
plt.ylim([0, 100])
plt.savefig("Figure_1.png")
plt.close()
print("Figure_1.png generated: Bar chart of Test Accuracy per Benchmark Dataset.")

# Determine the best performing dataset based on test accuracy for additional analysis
best_dataset = max(results.keys(), key=lambda k: results[k]["test_acc"])
print(f"\nThe best performing dataset based on test accuracy is: {best_dataset}.")

# For the best performing dataset generate a confusion matrix on the test split
ds_best = data_dict[best_dataset]
X_test_best = [x['sequence'] for x in ds_best["test"]]
y_test_best = [int(x['label']) for x in ds_best["test"]]
X_test_best_vec = vectorizer.transform(X_test_best)  # reusing last vectorizer; in practice, use corresponding one if different

# For consistency, re-train the model on the best dataset separately using its own vectorizer:
vectorizer_best = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
X_train_best = [x['sequence'] for x in ds_best["train"]]
y_train_best = [int(x['label']) for x in ds_best["train"]]
X_train_best_vec = vectorizer_best.fit_transform(X_train_best)
clf_best = LogisticRegression(max_iter=1000)
clf_best.fit(X_train_best_vec, y_train_best)
X_test_best_vec = vectorizer_best.transform(X_test_best)
y_pred_best = clf_best.predict(X_test_best_vec)
cm = confusion_matrix(y_test_best, y_pred_best)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Figure_2.png: Confusion Matrix for {best_dataset} Test Data")
plt.savefig("Figure_2.png")
plt.close()
print("Figure_2.png generated: Confusion Matrix for the best performing dataset's ("
      f"{best_dataset}) Test Data.")

# Summary: print all benchmark results
print("\nFinal Summary of Experimental Results:")
for name in results:
    print(f"Dataset {name}: Training Acc = {results[name]['train_acc']*100:.2f}%, "
          f"Dev Acc = {results[name]['dev_acc']*100:.2f}%, Test Acc = {results[name]['test_acc']*100:.2f}%")
    
print("\nAll experiments completed successfully!")
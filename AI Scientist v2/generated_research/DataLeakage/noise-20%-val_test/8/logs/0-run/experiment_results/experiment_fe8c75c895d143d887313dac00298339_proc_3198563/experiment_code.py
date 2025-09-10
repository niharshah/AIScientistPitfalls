import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import torch
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from collections import Counter
import joblib
import json

# -----------------------------------------------------------------------------
# GPU / device (required although sklearn is CPU-bound)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------
# Dataset loader (copied from provided utility)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


# -----------------------------------------------------------------------------
def tokenize(seq: str) -> List[str]:
    # If whitespace exists, use it; otherwise character-level
    return seq.split() if (" " in seq) else list(seq)


def build_vocab(seqs: List[str], max_tokens: int = 100) -> List[str]:
    cnt = Counter()
    for s in seqs:
        cnt.update(tokenize(s))
    most_common = [tok for tok, _ in cnt.most_common(max_tokens)]
    return most_common


def vectorize(seq: str, vocab: List[str]) -> np.ndarray:
    toks = tokenize(seq)
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)  # +1 for length
    tok_cnt = Counter(toks)
    for i, tok in enumerate(vocab):
        vec[i] = tok_cnt.get(tok, 0)
    vec[-1] = len(toks)
    return vec


# -----------------------------------------------------------------------------
# Load data
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# Build vocabulary on training data
train_seqs = spr["train"]["sequence"]
vocab = build_vocab(train_seqs, max_tokens=100)
print("Vocab size:", len(vocab))


# Vectorise all splits
def split_to_matrix(split):
    X = np.stack([vectorize(s, vocab) for s in spr[split]["sequence"]])
    y = np.array(spr[split]["label"])
    return X, y


X_train, y_train = split_to_matrix("train")
X_dev, y_dev = split_to_matrix("dev")
X_test, y_test = split_to_matrix("test")

# -----------------------------------------------------------------------------
# Experiment tracking dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -----------------------------------------------------------------------------
# Model (Decision Tree)
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
print("Decision tree trained.")

# Single "epoch" evaluation
y_train_pred = tree.predict_proba(X_train)
y_dev_pred = tree.predict_proba(X_dev)

train_loss = log_loss(y_train, y_train_pred)
dev_loss = log_loss(y_dev, y_dev_pred)
print(f"Epoch 1: validation_loss = {dev_loss:.4f}")

experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
experiment_data["SPR_BENCH"]["losses"]["val"].append(dev_loss)

train_acc = accuracy_score(y_train, np.argmax(y_train_pred, axis=1))
dev_acc = accuracy_score(y_dev, np.argmax(y_dev_pred, axis=1))
experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_acc)
experiment_data["SPR_BENCH"]["metrics"]["val"].append(dev_acc)

print(f"Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f}")


# -----------------------------------------------------------------------------
# SEFA on test split
def rule_predict(sample_vec: np.ndarray) -> int:
    # Re-run the tree decision path manually (sklearn helper)
    node_id = 0
    while tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]:
        feature = tree.tree_.feature[node_id]
        thresh = tree.tree_.threshold[node_id]
        node_id = (
            tree.tree_.children_left[node_id]
            if sample_vec[feature] <= thresh
            else tree.tree_.children_right[node_id]
        )
    # At leaf, return majority class
    classes = tree.tree_.value[node_id][0]
    return int(np.argmax(classes))


y_test_pred = []
sefa_hits = 0
for vec, gt in zip(X_test, y_test):
    model_label = int(tree.predict(vec.reshape(1, -1))[0])
    rule_label = rule_predict(vec)
    y_test_pred.append(model_label)
    if (model_label == rule_label) and (rule_label == gt):
        sefa_hits += 1

sefa = sefa_hits / len(y_test)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test Acc: {acc_test:.4f} | SEFA: {sefa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_test
experiment_data["SPR_BENCH"]["metrics"]["test_sefa"] = sefa
experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = acc_test

# -----------------------------------------------------------------------------
# Save experiment artifacts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
joblib.dump(tree, os.path.join(working_dir, "decision_tree.joblib"))
with open(os.path.join(working_dir, "vocab.json"), "w") as fp:
    json.dump(vocab, fp)
print("Artifacts saved to ./working")

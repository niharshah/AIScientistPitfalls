import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix
import itertools

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------- helper --------
nlayers_runs = experiment_data.get("nlayers", {})
test_f1_scores = {}
for k, exp in nlayers_runs.items():
    preds, gts = exp.get("predictions", []), exp.get("ground_truth", [])
    if preds and gts:
        test_f1_scores[k] = f1_score(gts, preds, average="macro")

# -------- plot 1: loss curves --------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, exp in nlayers_runs.items():
        epochs = exp["epochs"]
        axes[0].plot(epochs, exp["losses"]["train"], label=k)
        axes[1].plot(epochs, exp["losses"]["val"], label=k)
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    fig.suptitle("SPR_BENCH Loss Curves\nLeft: Train Loss, Right: Validation Loss")
    fig.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------- plot 2: F1 curves --------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, exp in nlayers_runs.items():
        epochs = exp["epochs"]
        axes[0].plot(epochs, exp["metrics"]["train_f1"], label=k)
        axes[1].plot(epochs, exp["metrics"]["val_f1"], label=k)
    axes[0].set_title("Train Macro F1")
    axes[1].set_title("Validation Macro F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
    fig.suptitle("SPR_BENCH F1 Curves\nLeft: Train F1, Right: Validation F1")
    fig.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# -------- plot 3: test F1 bar chart --------
try:
    if test_f1_scores:
        keys, vals = list(test_f1_scores.keys()), list(test_f1_scores.values())
        plt.figure(figsize=(6, 4))
        plt.bar(keys, vals, color="skyblue")
        plt.ylabel("Macro F1")
        plt.xlabel("nlayers setting")
        plt.title("SPR_BENCH Test F1 vs nlayers")
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar chart: {e}")
    plt.close()

# -------- plot 4: confusion matrix of best model --------
try:
    if test_f1_scores:
        best_key = max(test_f1_scores, key=test_f1_scores.get)
        best_exp = nlayers_runs[best_key]
        cm = confusion_matrix(best_exp["ground_truth"], best_exp["predictions"])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH Confusion Matrix ({best_key})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7
            )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{best_key}.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- print metrics --------
for k, v in test_f1_scores.items():
    print(f"{k}: Test Macro F1 = {v:.4f}")

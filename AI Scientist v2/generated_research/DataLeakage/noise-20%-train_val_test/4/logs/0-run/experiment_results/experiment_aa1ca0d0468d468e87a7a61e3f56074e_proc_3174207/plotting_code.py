import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment results
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run = experiment_data.get("no_char_bigram_count", {}).get("spr_bench", {})


# helper
def safe_get(dic, *keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else default


epochs = safe_get(run, "epochs", default=[])
train_loss = safe_get(run, "losses", "train", default=[])
val_loss = safe_get(run, "losses", "val", default=[])
train_f1 = safe_get(run, "metrics", "train_f1", default=[])
val_f1 = safe_get(run, "metrics", "val_f1", default=[])
preds = safe_get(run, "predictions", default=[])
gts = safe_get(run, "ground_truth", default=[])
test_f1 = safe_get(run, "metrics", "test_f1", default=None)

# ------------------------------------------------------------------
# 1) Loss curves
# ------------------------------------------------------------------
try:
    if epochs and train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) F1 curves
# ------------------------------------------------------------------
try:
    if epochs and train_f1 and val_f1:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("spr_bench: Training vs Validation Macro-F1")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_f1_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix
# ------------------------------------------------------------------
try:
    if preds and gts:
        num_labels = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "spr_bench: Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------
# print evaluation metric
# ------------------------------------------------------------------
if test_f1 is not None:
    print(f"Test Macro-F1: {test_f1:.4f}")

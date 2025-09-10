import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["feature_scale"]["SPR_BENCH"]
    cfgs = ed["configs"]
    best_cfg = ed["best_config"]
    best_idx = cfgs.index(best_cfg)
except Exception as e:
    print(f"Error loading or parsing experiment data: {e}")
    exit()


# Helper to safely fetch list-like data
def fetch(arr, idx):
    a = arr[idx]
    return np.asarray(a, dtype=float)


epochs = np.arange(1, len(ed["metrics"]["train_acc"][best_idx]) + 1)

# 1) Accuracy curves
try:
    plt.figure()
    plt.plot(epochs, fetch(ed["metrics"]["train_acc"], best_idx), label="Train")
    plt.plot(epochs, fetch(ed["metrics"]["val_acc"], best_idx), label="Validation")
    plt.title(f"SPR_BENCH Accuracy Curves ({best_cfg})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_BENCH_accuracy_{best_cfg}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure()
    plt.plot(epochs, fetch(ed["losses"]["train"], best_idx), label="Train")
    plt.plot(epochs, fetch(ed["losses"]["val"], best_idx), label="Validation")
    plt.title(f"SPR_BENCH Loss Curves ({best_cfg})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_BENCH_loss_{best_cfg}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Rule fidelity curve
try:
    plt.figure()
    plt.plot(epochs, fetch(ed["metrics"]["rule_fidelity"], best_idx))
    plt.title(f"SPR_BENCH Rule Fidelity ({best_cfg})")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    fname = os.path.join(working_dir, f"SPR_BENCH_rule_fidelity_{best_cfg}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating fidelity plot: {e}")
    plt.close()

# 4) Final validation accuracy across configs
try:
    plt.figure(figsize=(8, 4))
    final_vals = [float(arr[-1]) for arr in ed["metrics"]["val_acc"]]
    plt.bar(
        range(len(cfgs)),
        final_vals,
        color=["red" if c == best_cfg else "gray" for c in cfgs],
    )
    plt.xticks(range(len(cfgs)), cfgs, rotation=45, ha="right")
    plt.title("SPR_BENCH Final Validation Accuracy per Config")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_all_configs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# 5) Confusion matrix for best config (if predictions exist)
try:
    preds = np.asarray(ed["predictions"])
    gts = np.asarray(ed["ground_truth"])
    if preds.size and gts.size:
        num_classes = len(np.unique(np.concatenate([preds, gts])))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(
            f"SPR_BENCH Confusion Matrix ({best_cfg})\nLeft: Ground Truth, Right: Predicted"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))
        fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{best_cfg}.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

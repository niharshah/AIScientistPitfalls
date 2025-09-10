import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def rcwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# safely fetch first exp/dataset keys
exp_key = next(iter(experiment_data.keys()), None)
ds_key = next(iter(experiment_data.get(exp_key, {}).keys()), None)
exp_dict = experiment_data.get(exp_key, {}).get(ds_key, {})

losses = exp_dict.get("losses", {})
metrics = exp_dict.get("metrics", {})
preds = exp_dict.get("predictions", np.array([]))
gts = exp_dict.get("ground_truth", np.array([]))
timestamps = exp_dict.get("timestamps", [])
epochs = list(range(1, len(losses.get("train", [])) + 1))

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure()
    if "train" in losses and losses["train"]:
        plt.plot(epochs, losses["train"], label="Train Loss")
    if "val" in losses and losses["val"]:
        plt.plot(epochs, losses["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_key} Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_key}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Plot 2: RCWA curve ----------
try:
    plt.figure()
    if "val_rcwa" in metrics and metrics["val_rcwa"]:
        plt.plot(epochs, metrics["val_rcwa"], label="Val RCWA", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("RCWA")
    plt.title(f"{ds_key} Validation RCWA Curve")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_key}_rcwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RCWA plot: {e}")
    plt.close()

# ---------- compute & print test metrics ----------
test_rcwa = swa = cwa = np.nan
if preds.size and gts.size:
    seqs = exp_dict.get("sequences", [])  # may not exist
    if not len(seqs):  # fallback: fill dummy strings so length matches
        seqs = [""] * len(preds)
    test_rcwa = rcwa(seqs, gts, preds)
    swa = sum(
        count_shape_variety(s) if y == p else 0 for s, y, p in zip(seqs, gts, preds)
    ) / max(sum(count_shape_variety(s) for s in seqs), 1)
    cwa = sum(
        count_color_variety(s) if y == p else 0 for s, y, p in zip(seqs, gts, preds)
    ) / max(sum(count_color_variety(s) for s in seqs), 1)
    print(f"Test RCWA={test_rcwa:.4f}, SWA={swa:.4f}, CWA={cwa:.4f}")

# ---------- Plot 3: Bar chart of aggregate metrics ----------
try:
    if not np.isnan(test_rcwa):
        plt.figure()
        plt.bar(
            ["RCWA", "SWA", "CWA"],
            [test_rcwa, swa, cwa],
            color=["skyblue", "lightgreen", "salmon"],
        )
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(f"{ds_key} Test Aggregate Metrics")
        plt.savefig(os.path.join(working_dir, f"{ds_key}_aggregate_metrics.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregate metrics plot: {e}")
    plt.close()

# ---------- Plot 4: Confusion heatmap (limited size) ----------
try:
    if preds.size and gts.size:
        from collections import Counter

        max_classes = 20
        labels = sorted(set(gts) | set(preds))[:max_classes]
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for y, p in zip(gts, preds):
            if y in idx and p in idx:
                cm[idx[y], idx[p]] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_key} Confusion Matrix (Top {len(labels)} classes)")
        plt.savefig(os.path.join(working_dir, f"{ds_key}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

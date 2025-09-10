import matplotlib.pyplot as plt
import numpy as np
import os

# prepare paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dict safely
def get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k, None)
    return d if d is not None else default


# select our run
run_key = "no_positional_embedding"
ds_key = "spr_bench"
ed = get(experiment_data, run_key, ds_key, default={})

epochs = np.array(get(ed, "epochs", default=[]))
train_loss = np.array(get(ed, "losses", "train", default=[]))
val_loss = np.array(get(ed, "losses", "val", default=[]))
train_f1 = np.array(get(ed, "metrics", "train_f1", default=[]))
val_f1 = np.array(get(ed, "metrics", "val_f1", default=[]))
preds = np.array(get(ed, "predictions", default=[]))
gts = np.array(get(ed, "ground_truth", default=[]))

# 1) Loss curve
try:
    if epochs.size and train_loss.size and val_loss.size:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) F1 curve
try:
    if epochs.size and train_f1.size and val_f1.size:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("spr_bench F1 Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_f1_curve.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Confusion matrix (heat-map)
try:
    if preds.size and gts.size and preds.shape[0] == gts.shape[0]:
        num_labels = int(max(max(preds), max(gts)) + 1)
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            "spr_bench Confusion Matrix\nLeft: Ground Truth (rows), Right: Predictions (cols)"
        )
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

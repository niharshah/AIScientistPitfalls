import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dicts
def get_path(dic, path, default=None):
    for p in path:
        dic = dic.get(p, {})
    return dic if dic else default


run_key = ["NoColorEmbedding", "SPR_BENCH"]
loss_train = get_path(experiment_data, run_key + ["losses", "train"], [])
loss_val = get_path(experiment_data, run_key + ["losses", "val"], [])
metrics_val = get_path(experiment_data, run_key + ["metrics", "val"], [])
metrics_test = get_path(experiment_data, run_key + ["metrics", "test"], {})
preds = get_path(experiment_data, run_key + ["predictions"], [])
gts = get_path(experiment_data, run_key + ["ground_truth"], [])

epochs = list(range(1, len(loss_train) + 1))

# ------------------- figure 1: losses ------------- #
try:
    plt.figure()
    if loss_train:
        plt.plot(epochs, loss_train, label="Train Loss")
    if loss_val:
        plt.plot(epochs, loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------- figure 2: metrics ------------- #
try:
    plt.figure()
    if metrics_val:
        cwa = [m.get("CWA", np.nan) for m in metrics_val]
        swa = [m.get("SWA", np.nan) for m in metrics_val]
        gcwa = [m.get("GCWA", np.nan) for m in metrics_val]
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Weighted Accuracies")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------------- figure 3: confusion matrix ---- #
try:
    if preds and gts:
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------- print final metrics ----------- #
if metrics_test:
    print("Final Test Metrics:")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.3f}")

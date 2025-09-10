import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
exp_file = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_metrics = {}  # store best metric per dataset for comparison

for dset, ddata in experiment_data.items():
    # ----- numeric summaries -----
    preds = np.asarray(ddata.get("predictions", []))
    gts = np.asarray(ddata.get("ground_truth", []))
    acc = (preds == gts).mean() if preds.size else float("nan")
    val_metrics = np.asarray(ddata.get("metrics", {}).get("val", []))
    best_val_metric = val_metrics.max() if val_metrics.size else float("nan")
    print(f"{dset}: accuracy={acc:.4f}  best_val_metric={best_val_metric:.4f}")
    best_metrics[dset] = best_val_metric

    # ----- 1. loss curves -----
    try:
        plt.figure()
        epochs = np.arange(1, len(ddata["losses"]["train"]) + 1)
        plt.plot(epochs, ddata["losses"]["train"], label="train")
        plt.plot(epochs, ddata["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # ----- 2. validation metric curve -----
    try:
        if val_metrics.size:
            plt.figure()
            plt.plot(np.arange(1, len(val_metrics) + 1), val_metrics, color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Comp-Weighted Accuracy")
            plt.title(f"{dset} – Validation Comp-Weighted Accuracy")
            fname = os.path.join(working_dir, f"{dset}_val_metric.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting metric for {dset}: {e}")
        plt.close()

    # ----- 3. confusion matrix -----
    try:
        if preds.size and gts.size:
            num_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset} – Confusion Matrix\n(rows = GT, cols = Pred)")
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

# ----- 4. cross-dataset comparison -----
try:
    if best_metrics:
        plt.figure()
        names, vals = zip(*best_metrics.items())
        plt.bar(names, vals, color="skyblue")
        plt.ylabel("Best Validation Comp-Weighted Accuracy")
        plt.title("Dataset Comparison – Best Validation Metric")
        fname = os.path.join(working_dir, "datasets_best_val_metric.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error plotting dataset comparison: {e}")
    plt.close()

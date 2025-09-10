import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate over datasets ----------
for ds_name, ds_dict in experiment_data.items():
    epochs = np.array(ds_dict.get("epochs", []))
    train_loss = np.array(ds_dict.get("losses", {}).get("train", []))
    val_loss = np.array(ds_dict.get("losses", {}).get("val", []))
    val_metrics = ds_dict.get("metrics", {}).get("val", [])
    preds = np.array(ds_dict.get("predictions", []))
    gts = np.array(ds_dict.get("ground_truth", []))

    # ---- Plot 1: loss curves ----
    try:
        plt.figure()
        plt.plot(epochs, train_loss, "--", label="Train")
        plt.plot(epochs, val_loss, "-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train (--), Right: Validation (—)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # ---- Plot 2: validation metrics ----
    try:
        if val_metrics:
            swa = np.array([m["SWA"] for m in val_metrics])
            cwa = np.array([m["CWA"] for m in val_metrics])
            scaa = np.array([m["SCAA"] for m in val_metrics])
            plt.figure()
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, scaa, label="SCAA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{ds_name} Validation Metrics Across Epochs")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            raise ValueError("No validation metrics found")
    except Exception as e:
        print(f"Error creating metrics plot for {ds_name}: {e}")
        plt.close()

    # ---- Plot 3: confusion matrix ----
    try:
        if preds.size and gts.size:
            labels = sorted(set(gts.tolist() + preds.tolist()))
            n = len(labels)
            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(gts, preds):
                cm[labels.index(t), labels.index(p)] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xticks(range(n), labels)
            plt.yticks(range(n), labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds_name} Confusion Matrix\nDataset: {ds_name}")
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            raise ValueError("Predictions/ground truth missing")
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---- Plot 4: label distribution ----
    try:
        if preds.size and gts.size:
            uniq = sorted(set(gts.tolist() + preds.tolist()))
            gt_counts = [np.sum(gts == u) for u in uniq]
            pr_counts = [np.sum(preds == u) for u in uniq]
            x = np.arange(len(uniq))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_counts, width, label="Predictions")
            plt.xlabel("Class Label")
            plt.ylabel("Count")
            plt.title(
                f"{ds_name} Label Distribution\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xticks(x, uniq)
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_label_distribution.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            raise ValueError("Predictions/ground truth missing")
    except Exception as e:
        print(f"Error creating label distribution for {ds_name}: {e}")
        plt.close()

    # ---- console summary ----
    if val_metrics:
        best_scaa = max(m["SCAA"] for m in val_metrics)
        best_cwa = max(m["CWA"] for m in val_metrics)
        best_swa = max(m["SWA"] for m in val_metrics)
        print(
            f"{ds_name} – best validation SCAA={best_scaa:.3f}, "
            f"CWA={best_cwa:.3f}, SWA={best_swa:.3f}"
        )

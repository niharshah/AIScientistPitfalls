import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------
try:
    edata = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    edata = {}

for dname, dct in edata.items():
    metrics = dct.get("metrics", {})
    preds = np.array(dct.get("predictions", []))
    gts = np.array(dct.get("ground_truth", []))

    epochs = np.arange(1, len(metrics.get("train_loss", [])) + 1)

    # --------------- 1) loss curve ------------------------
    try:
        plt.figure()
        plt.plot(epochs, metrics["train_loss"], label="Train Loss")
        plt.plot(epochs, metrics["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname} Loss Curves")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # --------------- 2) accuracy metrics -----------------
    try:
        plt.figure()
        for key, lab in [
            ("val_swa", "SWA"),
            ("val_cwa", "CWA"),
            ("val_zsrta", "ZSRTA"),
        ]:
            if key in metrics and len(metrics[key]):
                plt.plot(epochs, metrics[key], label=lab)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname} Validation Accuracies")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dname}_val_accs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # --------------- 3) confusion matrix ----------------
    try:
        if preds.size and gts.size:
            cm = confusion_matrix(gts, preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap="Blues")
            plt.title(f"{dname} Confusion Matrix")
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # --------------- 4) class frequency bar chart --------
    try:
        if preds.size and gts.size:
            classes = np.arange(max(preds.max(), gts.max()) + 1)
            gt_counts = [(gts == c).sum() for c in classes]
            pr_counts = [(preds == c).sum() for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_counts, width, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(f"{dname} Class Distribution")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_class_distribution.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating class distribution for {dname}: {e}")
        plt.close()

    # --------------- print final metrics -----------------
    try:
        if len(metrics.get("val_loss", [])):
            idx = -1  # last epoch
            print(
                f"{dname.upper()} FINAL VAL METRICS -- "
                f"loss: {metrics['val_loss'][idx]:.4f}, "
                f"SWA: {metrics['val_swa'][idx]:.3f}, "
                f"CWA: {metrics['val_cwa'][idx]:.3f}, "
                f"ZSRTA: {metrics['val_zsrta'][idx]:.3f}"
            )
        if preds.size and gts.size:
            test_acc = (preds == gts).mean()
            print(f"{dname.upper()} TEST ACCURACY: {test_acc:.3f}")
    except Exception as e:
        print(f"Error printing metrics for {dname}: {e}")

print("Finished generating plots.")

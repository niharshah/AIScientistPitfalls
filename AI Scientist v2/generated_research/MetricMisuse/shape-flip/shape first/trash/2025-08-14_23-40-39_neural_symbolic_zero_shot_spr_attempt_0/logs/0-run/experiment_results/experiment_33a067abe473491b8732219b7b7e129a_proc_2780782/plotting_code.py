import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- per-dataset plots ----------
swa_curves = {}  # collect for optional comparison plot
for dname, ddata in experiment_data.items():
    metrics = ddata.get("metrics", {})
    preds = ddata.get("predictions", {})
    gts = ddata.get("ground_truth", {})
    ep = np.arange(1, len(metrics.get("train_loss", [])) + 1)

    # --- 1. train / val loss ---
    try:
        plt.figure()
        plt.plot(ep, metrics.get("train_loss", []), label="Train Loss")
        plt.plot(ep, metrics.get("val_loss", []), label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # --- 2. validation SWA curve ---
    try:
        plt.figure()
        plt.plot(ep, metrics.get("val_swa", []), label="Val SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dname}: Validation SWA Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_swa_curve.png")
        plt.savefig(fname)
        plt.close()
        swa_curves[dname] = (ep, metrics.get("val_swa", []))
    except Exception as e:
        print(f"Error creating SWA plot for {dname}: {e}")
        plt.close()

    # --- 3. confusion matrix on test ---
    try:
        t_preds = np.array(preds.get("test", []))
        t_gts = np.array(gts.get("test", []))
        if t_preds.size and t_gts.size:
            classes = np.unique(np.concatenate([t_gts, t_preds]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(t_gts, t_preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{dname}: Test Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(classes)
            plt.yticks(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dname}_test_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()

            # print simple metrics
            total = cm.sum()
            acc = np.trace(cm) / total if total else 0.0
            print(
                f"{dname} | Final VAL loss {metrics.get('val_loss', [np.nan])[-1]:.4f} "
                f"| VAL SWA {metrics.get('val_swa', [np.nan])[-1]:.4f} "
                f"| TEST acc {acc:.4f}"
            )
        else:
            print(f"{dname}: Missing test predictions or labels; skipping CM.")
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

# ---------- 4. cross-dataset SWA comparison ----------
if len(swa_curves) > 1:
    try:
        plt.figure()
        for name, (ep, curve) in swa_curves.items():
            plt.plot(ep, curve, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation SWA")
        plt.title("Dataset Comparison: Validation SWA Curves")
        plt.legend()
        fname = os.path.join(working_dir, "comparison_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()

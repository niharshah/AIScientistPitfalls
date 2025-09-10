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
    experiment_data = None

if experiment_data:
    swa_per_ds = {}  # keep for later comparison plot
    for ds_name, ds_blob in experiment_data.items():
        metrics = ds_blob.get("metrics", {})
        train_loss = np.asarray(metrics.get("train_loss", []), dtype=float)
        val_loss = np.asarray(metrics.get("val_loss", []), dtype=float)
        val_swa = np.asarray(metrics.get("val_swa", []), dtype=float)
        epochs = np.arange(1, len(train_loss) + 1)

        # ---------- 1. loss curves ----------
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name}: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()

        # ---------- 2. SWA curve ----------
        try:
            plt.figure()
            plt.plot(epochs, val_swa, label="Val SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{ds_name}: Validation SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_swa_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot for {ds_name}: {e}")
            plt.close()

        swa_per_ds[ds_name] = val_swa

        # ---------- 3. confusion matrix ----------
        try:
            preds = np.asarray(ds_blob["predictions"]["test"], dtype=int)
            gts = np.asarray(ds_blob["ground_truth"]["test"], dtype=int)
            classes = np.arange(max(preds.max(initial=0), gts.max(initial=0)) + 1)
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{ds_name}: Test Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(classes)
            plt.yticks(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{ds_name}_test_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()

        # ---------- metrics print ----------
        total = cm.sum() if "cm" in locals() else 0
        acc = np.trace(cm) / total if total else 0.0
        last_swa = val_swa[-1] if val_swa.size else np.nan
        print(f"[{ds_name}] TEST accuracy: {acc:.4f} | last Val-SWA: {last_swa:.4f}")

    # ---------- 4. combined SWA comparison ----------
    if len(swa_per_ds) > 1:
        try:
            plt.figure()
            for ds_name, swa in swa_per_ds.items():
                ep = np.arange(1, len(swa) + 1)
                plt.plot(ep, swa, label=f"{ds_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("Dataset Comparison: Validation SWA")
            plt.legend()
            fname = os.path.join(working_dir, "all_datasets_swa_comparison.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating combined SWA plot: {e}")
            plt.close()

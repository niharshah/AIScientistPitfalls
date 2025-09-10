import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# ---------------------------------------------------------------------
for dname, exp in experiment_data.items():
    try:
        epochs = np.array(exp["epochs"])
        train_losses = np.array(exp["losses"]["train"])
        val_metrics = exp["metrics"]["val"]
        train_metrics = exp["metrics"]["train"]
    except KeyError as e:
        print(f"Dataset {dname} missing key {e}, skipping.")
        continue

    val_cwa = np.array([m["cwa"] for m in val_metrics])
    val_swa = np.array([m["swa"] for m in val_metrics])
    val_cpx = np.array([m["cpx"] for m in val_metrics])
    train_cpx = np.array([m["cpx"] for m in train_metrics])

    best_epoch = int(epochs[np.argmax(val_cpx)])
    best_val_cpx = float(val_cpx.max())
    print(f"{dname}: Best Validation CpxWA = {best_val_cpx:.4f} @ epoch {best_epoch}")

    # -------------------------- Plot 1 -------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_losses, marker="o", label="Train Loss")
        plt.title(f"{dname}: Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_train_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train loss plot for {dname}: {e}")
        plt.close()

    # -------------------------- Plot 2 -------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_cpx, marker="o", label="Train CpxWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(
            f"{dname}: Complexity-Weighted Accuracy\nLeft: Train, Right: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_cpxwa_train_val_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA curve for {dname}: {e}")
        plt.close()

    # -------------------------- Plot 3 -------------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, marker="o", label="Val CWA")
        plt.plot(epochs, val_swa, marker="^", label="Val SWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(f"{dname}: Weighted Accuracy Comparison (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dname}_val_weighted_accuracy_comparison.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy comparison plot for {dname}: {e}")
        plt.close()

    # -------------------------- Plot 4 -------------------------------
    try:
        preds = np.array(exp.get("predictions", []))
        gts = np.array(exp.get("ground_truth", []))
        if preds.size and preds.size == gts.size:
            num_cls = len(np.unique(np.concatenate([preds, gts])))
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.title(f"{dname}: Confusion Matrix (Validation)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

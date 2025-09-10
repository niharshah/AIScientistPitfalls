import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# -------------------------------------------------------------
for ds_name, ds in experiment_data.items():
    epochs = np.array(ds["epochs"])
    tr_losses = np.array(ds["losses"]["train"])
    tr_metrics = ds["metrics"]["train"]
    val_metrics = ds["metrics"]["val"]

    val_cwa = np.array([m["cwa"] for m in val_metrics])
    val_swa = np.array([m["swa"] for m in val_metrics])
    val_cpx = np.array([m["cpx"] for m in val_metrics])
    tr_cpx = np.array([m["cpx"] for m in tr_metrics])

    best_ep_idx = int(np.argmax(val_cpx))
    best_ep = int(epochs[best_ep_idx])
    best_val_cpx = float(val_cpx[best_ep_idx])
    print(f"{ds_name}: Best Validation CpxWA {best_val_cpx:.4f} @ epoch {best_ep}")

    # 1) Training loss curve
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, marker="o", label="Train Loss")
        plt.title(f"{ds_name}: Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Train vs Val CpxWA
    try:
        plt.figure()
        plt.plot(epochs, tr_cpx, marker="o", label="Train CpxWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(f"{ds_name}: Complexity-Weighted Accuracy\nLeft: Train, Right: Val")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_cpxwa_train_val_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA curve for {ds_name}: {e}")
        plt.close()

    # 3) Validation weighted-accuracy comparison
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, marker="o", label="Val CWA")
        plt.plot(epochs, val_swa, marker="^", label="Val SWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(f"{ds_name}: Weighted Accuracy Comparison (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_val_weighted_accuracy_cmp.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy comparison for {ds_name}: {e}")
        plt.close()

    # 4) Confusion matrix of best epoch predictions
    try:
        preds = np.array(ds.get("predictions", []))
        gts = np.array(ds.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{ds_name}: Confusion Matrix (Best Epoch)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

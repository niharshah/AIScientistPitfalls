import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate through datasets ----------
for dset_name, dset in experiment_data.items():
    metrics = dset.get("metrics", {})
    preds = np.array(dset.get("predictions", []))
    gts = np.array(dset.get("ground_truth", []))

    # ---- 1) train/val loss ----
    try:
        train_loss = np.array(
            metrics.get("train_loss", [])
        )  # shape (N,2) [epoch, value]
        val_loss = np.array(metrics.get("val_loss", []))
        if train_loss.size and val_loss.size:
            plt.figure()
            plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train Loss")
            plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dset_name}: Training vs Validation Loss\n(Left: Ground Truth, Right: Generated Samples)"
            )
            plt.legend()
            fname = f"{dset_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # ---- 2) validation ACS ----
    try:
        val_acs = np.array(metrics.get("val_ACS", []))
        if val_acs.size:
            plt.figure()
            plt.plot(val_acs[:, 0], val_acs[:, 1], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("ACS")
            plt.ylim(0, 1)
            plt.title(
                f"{dset_name}: Validation ACS over Epochs\n(Left: Ground Truth, Right: Generated Samples)"
            )
            fname = f"{dset_name}_ACS_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating ACS plot for {dset_name}: {e}")
        plt.close()

    # ---- 3) confusion matrix ----
    try:
        if preds.size and gts.size:
            num_classes = max(max(preds), max(gts)) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dset_name}: Confusion Matrix\n(Left: Ground Truth, Right: Generated Samples)"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            fname = f"{dset_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

print("Finished saving plots to", working_dir)

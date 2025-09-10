import matplotlib.pyplot as plt
import numpy as np
import os

# --------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --------- plotting ---------
for ds_name, ds_dict in experiment_data.items():
    losses = ds_dict.get("losses", {})
    metrics = ds_dict.get("metrics", {})
    preds = np.array(ds_dict.get("predictions", []))
    gts = np.array(ds_dict.get("ground_truth", []))

    # -- plot 1: loss curves --
    try:
        plt.figure()
        if losses.get("train"):  # safeguard against missing keys
            plt.plot(losses["train"], label="Train Loss")
        if losses.get("val"):
            plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"{ds_name} – Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        save_path = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {ds_name}: {e}")
        plt.close()

    # -- plot 2: validation CWA2 curve --
    try:
        if metrics.get("val_cwa2"):
            plt.figure()
            plt.plot(metrics["val_cwa2"], marker="o")
            plt.title(f"{ds_name} – Validation CWA2 over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("CWA2")
            save_path = os.path.join(working_dir, f"{ds_name}_val_cwa2.png")
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating CWA2 plot for {ds_name}: {e}")
        plt.close()

    # -- plot 3: confusion matrix heatmap --
    try:
        if preds.size and gts.size:
            num_classes = max(preds.max(), gts.max()) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(
                f"{ds_name} – Confusion Matrix (Ground Truth rows, Predictions cols)"
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            save_path = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # --------- print final metric ---------
    if metrics.get("val_cwa2"):
        print(f'{ds_name} final Validation CWA2: {metrics["val_cwa2"][-1]:.4f}')

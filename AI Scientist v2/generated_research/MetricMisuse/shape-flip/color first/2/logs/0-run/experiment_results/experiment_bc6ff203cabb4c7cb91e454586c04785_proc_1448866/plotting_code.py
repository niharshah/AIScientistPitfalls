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


# ---------- helper ----------
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).sum() / max(1, len(y_true))


# ---------- plotting ----------
for ds_name, ds_log in experiment_data.items():
    # ---- pre-extract ----
    losses = ds_log.get("losses", {})
    metrics = ds_log.get("metrics", {})
    preds = ds_log.get("predictions", [])
    gts = ds_log.get("ground_truth", [])
    epochs_loss = [e for e, _ in losses.get("train", [])]

    # 1) Loss curves ----------------------------------------------------------
    try:
        plt.figure(figsize=(7, 4))
        if losses.get("train"):
            plt.plot(epochs_loss, [v for _, v in losses["train"]], "--", label="train")
        if losses.get("val"):
            plt.plot(epochs_loss, [v for _, v in losses["val"]], "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} - Training / Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Metric curves --------------------------------------------------------
    try:
        plt.figure(figsize=(7, 4))
        if metrics.get("train"):
            plt.plot(epochs_loss, [v for _, v in metrics["train"]], "--", label="train")
        if metrics.get("val"):
            plt.plot(epochs_loss, [v for _, v in metrics["val"]], "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title(f"{ds_name} - Training / Validation PCWA")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_pcwa_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds_name}: {e}")
        plt.close()

    # 3) Final test accuracy bar ---------------------------------------------
    try:
        if preds and gts:
            acc = accuracy(gts, preds)
            plt.figure(figsize=(4, 4))
            plt.bar([ds_name], [acc], color="skyblue")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} - Test Accuracy")
            plt.tight_layout()
            fname = f"{ds_name}_test_accuracy.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            print(f"{ds_name} test accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

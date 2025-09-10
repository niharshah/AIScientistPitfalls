import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- Load experiment data ------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------- Helper for safe fetch ------------------ #
def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


# ------------------- Iterate and plot ----------------------- #
for model_name, datasets in experiment_data.items():
    for ds_name, records in datasets.items():
        # Fetch arrays
        train_acc = _safe_get(records, "metrics", "train", default=[])
        val_acc = _safe_get(records, "metrics", "val", default=[])
        train_loss = _safe_get(records, "losses", "train", default=[])
        val_loss = _safe_get(records, "losses", "val", default=[])
        test_acc = _safe_get(records, "test_acc", default=None)

        epochs = range(1, len(train_acc) + 1)

        # --------------- Accuracy curve ---------------- #
        try:
            plt.figure()
            plt.plot(epochs, train_acc, label="Train")
            plt.plot(epochs, val_acc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{model_name} on {ds_name}\nTraining vs Validation Accuracy")
            plt.legend()
            save_name = f"{ds_name}_accuracy_curve_{model_name}.png"
            plt.savefig(os.path.join(working_dir, save_name))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {ds_name}: {e}")
            plt.close()

        # --------------- Loss curve -------------------- #
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{model_name} on {ds_name}\nTraining vs Validation Loss")
            plt.legend()
            save_name = f"{ds_name}_loss_curve_{model_name}.png"
            plt.savefig(os.path.join(working_dir, save_name))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()

        # --------------- Print final metrics ----------- #
        final_train = train_acc[-1] if train_acc else None
        final_val = val_acc[-1] if val_acc else None
        if final_train is not None:
            print(
                f"{model_name} | {ds_name}: Train Acc={final_train:.4f}, Val Acc={final_val:.4f}, Test Acc={test_acc:.4f}"
                if test_acc is not None
                else f"{model_name} | {ds_name}: Train Acc={final_train:.4f}, Val Acc={final_val:.4f}"
            )

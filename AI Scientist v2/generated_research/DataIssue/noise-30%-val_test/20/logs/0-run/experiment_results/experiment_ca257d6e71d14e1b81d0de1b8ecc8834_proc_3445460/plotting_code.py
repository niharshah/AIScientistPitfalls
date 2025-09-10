import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------- #
# paths & data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# --------------------------------------------------------------------------- #
# helper to extract metric arrays safely
def get_metric_array(metric_list, key):
    if not metric_list:
        return np.array([])
    return np.array([m.get(key, np.nan) for m in metric_list])


# --------------------------------------------------------------------------- #
# iterate over datasets
for dset_name, logs in experiment_data.items():
    # ---------- 1) loss curves --------------------------------------------- #
    try:
        train_losses = np.array(logs["losses"].get("train", []))
        val_losses = np.array(logs["losses"].get("val", []))
        if train_losses.size and val_losses.size:
            epochs = np.arange(1, len(train_losses) + 1)
            plt.figure()
            plt.plot(epochs, train_losses, label="Train", color="tab:blue")
            plt.plot(epochs, val_losses, label="Val", color="tab:orange")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name}: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset_name}: {e}")
        plt.close()

    # ---------- 2) metric curves (macro-F1 & CWA) --------------------------- #
    try:
        val_metrics = logs["metrics"].get("val", [])
        macro_f1 = get_metric_array(val_metrics, "macro_f1")
        cwa = get_metric_array(val_metrics, "cwa")
        if macro_f1.size and cwa.size:
            epochs = np.arange(1, len(macro_f1) + 1)
            plt.figure()
            plt.plot(epochs, macro_f1, label="Macro-F1", color="tab:green")
            plt.plot(epochs, cwa, label="CWA", color="tab:red")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dset_name}: Validation Metrics")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_metric_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric curves for {dset_name}: {e}")
        plt.close()

    # ---------- 3) final-epoch bar plot ------------------------------------- #
    try:
        if macro_f1.size and cwa.size:
            final_vals = [macro_f1[-1], cwa[-1]]
            labels = ["Macro-F1", "CWA"]
            colors = ["tab:green", "tab:red"]
            plt.figure()
            plt.bar(labels, final_vals, color=colors)
            for x, y in zip(labels, final_vals):
                plt.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom")
            plt.ylabel("Score")
            plt.title(f"{dset_name}: Final-Epoch Metrics")
            fname = os.path.join(working_dir, f"{dset_name}_final_metrics_bar.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating bar plot for {dset_name}: {e}")
        plt.close()

    # ---------- 4) confusion matrix ---------------------------------------- #
    try:
        preds = np.array(logs.get("predictions", []))
        gts = np.array(logs.get("ground_truth", []))
        if preds.size and gts.size and preds.shape == gts.shape:
            cm = np.zeros((2, 2), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{dset_name}: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.colorbar()
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

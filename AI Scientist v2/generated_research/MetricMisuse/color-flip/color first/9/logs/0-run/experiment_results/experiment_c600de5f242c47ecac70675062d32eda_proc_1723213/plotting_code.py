import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Only proceed if data loaded
for dset_name, data in experiment_data.items():
    # ------------- Figure 1: Loss curves -----------------------------
    try:
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title(f"{dset_name}: Training vs Validation Loss")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # ------------- Figure 2: Validation metric curves ----------------
    try:
        metrics_hist = data["metrics"]["val"]  # list of dicts
        cwa = [m["cwa"] for m in metrics_hist]
        swa = [m["swa"] for m in metrics_hist]
        dwhs = [m["dwhs"] for m in metrics_hist]
        epochs = np.arange(1, len(cwa) + 1)

        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, dwhs, label="DWHS")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(f"{dset_name}: Validation Metrics over Epochs")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset_name}_val_metrics.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot for {dset_name}: {e}")
        plt.close()

    # ------------- Figure 3: Confusion matrix ------------------------
    try:
        preds = np.array(data["predictions"])
        truth = np.array(data["ground_truth"])
        labels = np.unique(np.concatenate([truth, preds]))
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(truth, preds):
            cm[label_to_idx[t], label_to_idx[p]] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dset_name}: Confusion Matrix (Test set)")
        save_path = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # ------------- Figure 4: Test metrics bar chart ------------------
    try:
        test_metrics = data.get("test_metrics", {})
        keys = list(test_metrics.keys())
        values = [test_metrics[k] for k in keys]

        plt.figure()
        plt.bar(keys, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title(f"{dset_name}: Final Test Metrics")
        save_path = os.path.join(working_dir, f"{dset_name}_test_metrics.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart for {dset_name}: {e}")
        plt.close()

    # ----------- Print numeric test metrics --------------------------
    try:
        print(f"{dset_name} test metrics:", data["test_metrics"])
    except Exception as e:
        print(f"Could not print test metrics for {dset_name}: {e}")

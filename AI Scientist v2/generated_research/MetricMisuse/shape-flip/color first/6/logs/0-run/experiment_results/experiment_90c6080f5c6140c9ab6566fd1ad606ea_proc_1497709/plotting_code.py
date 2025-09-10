import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_list = ["variety", "freq", "mod", "union"]

    # --------- 1) validation-accuracy curve ---------
    try:
        plt.figure()
        for ds in ds_list:
            acc = experiment_data["multi_rule_ablation"][ds]["losses"]["val_acc"]
            plt.plot(range(1, len(acc) + 1), acc, label=ds)
        plt.title("Validation Accuracy – Multi-rule Dataset")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "val_accuracy_multi_rule.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation accuracy plot: {e}")
        plt.close()

    # --------- 2) training & validation loss curves ---------
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, ds in enumerate(ds_list):
            ax = axes.flat[idx]
            losses = experiment_data["multi_rule_ablation"][ds]["losses"]
            ax.plot(losses["train_loss"], label="train")
            ax.plot(losses["val_loss"], label="val")
            ax.set_title(f"{ds} – Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("Training / Validation Loss Curves")
        plt.tight_layout()
        fname = os.path.join(working_dir, "loss_curves_all_ds.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve figure: {e}")
        plt.close()

    # --------- 3) transfer-accuracy heatmap ---------
    try:
        heat = np.zeros((4, 3))  # rows: trained on, cols: test set (var, freq, mod)
        for i, train_ds in enumerate(ds_list):
            col_map = experiment_data["multi_rule_ablation"][train_ds]["transfer_acc"]
            for j, test_ds in enumerate(["variety", "freq", "mod"]):
                heat[i, j] = col_map[test_ds]["accuracy"]
        fig, ax = plt.subplots()
        im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["variety", "freq", "mod"])
        ax.set_yticks(range(4))
        ax.set_yticklabels(ds_list)
        ax.set_title("Transfer Accuracy Heat-map\nRows: Trained on, Columns: Tested on")
        for i in range(4):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{heat[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if heat[i, j] < 0.5 else "black",
                )
        fig.colorbar(im, ax=ax, label="Accuracy")
        fname = os.path.join(working_dir, "transfer_accuracy_heatmap.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating transfer heatmap: {e}")
        plt.close()

    # --------- 4) confusion matrix for 'variety' ---------
    try:
        preds = experiment_data["multi_rule_ablation"]["variety"]["predictions"]
        gts = experiment_data["multi_rule_ablation"]["variety"]["ground_truth"]
        num_cls = 3
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Confusion Matrix – Variety Dataset")
        ax.set_xticks(range(num_cls))
        ax.set_yticks(range(num_cls))
        for i in range(num_cls):
            for j in range(num_cls):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        fname = os.path.join(working_dir, "confusion_matrix_variety.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

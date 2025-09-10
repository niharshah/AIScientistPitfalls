import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
# load experiment data
# -----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -----------------------------------------------------------
# helper
# -----------------------------------------------------------
def save_fig(fig, name):
    fig.tight_layout()
    fname = os.path.join(working_dir, name)
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved {fname}")


# -----------------------------------------------------------
# plotting for each dataset
# -----------------------------------------------------------
for dset, d in experiment_data.items():
    # -------- Plot 1: loss curves ----------
    try:
        fig = plt.figure()
        epochs_tr, losses_tr = (
            zip(*d["losses"]["train"]) if d["losses"]["train"] else ([], [])
        )
        epochs_val, losses_val = (
            zip(*d["losses"]["val"]) if d["losses"]["val"] else ([], [])
        )
        if epochs_tr:
            plt.plot(epochs_tr, losses_tr, label="Train")
        if epochs_val:
            plt.plot(epochs_val, losses_val, label="Validation")
        plt.title(f"{dset} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        save_fig(fig, f"{dset}_loss_curve.png")
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # -------- Plot 2: validation Macro-F1 ----------
    try:
        fig = plt.figure()
        epochs_val, f1_vals = (
            zip(*d["metrics"]["val"]) if d["metrics"]["val"] else ([], [])
        )
        plt.plot(epochs_val, f1_vals, marker="o")
        plt.title(f"{dset} Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        save_fig(fig, f"{dset}_val_macroF1.png")
    except Exception as e:
        print(f"Error creating Macro-F1 plot for {dset}: {e}")
        plt.close()

    # -------- Plot 3: confusion matrix ----------
    try:
        y_true = np.array(d.get("ground_truth", []))
        y_pred = np.array(d.get("predictions", []))
        if y_true.size and y_pred.size:
            num_cls = int(max(y_true.max(), y_pred.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            fig = plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset} Confusion Matrix")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="red" if cm[i, j] else "black",
                        fontsize=8,
                    )
            save_fig(fig, f"{dset}_confusion_matrix.png")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # -------- Plot 4: final metrics ----------
    try:
        final = d.get("final_metrics", {})
        if final:
            fig = plt.figure()
            names, vals = zip(*final.items())
            plt.bar(names, vals)
            plt.ylim(0, 1.0)
            plt.title(f"{dset} Final Test Metrics")
            for i, v in enumerate(vals):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
            save_fig(fig, f"{dset}_final_metrics_bar.png")
    except Exception as e:
        print(f"Error creating final metrics bar for {dset}: {e}")
        plt.close()

    # -------- Print metrics ----------
    if "final_metrics" in d:
        print(f"{dset} final metrics:", d["final_metrics"])

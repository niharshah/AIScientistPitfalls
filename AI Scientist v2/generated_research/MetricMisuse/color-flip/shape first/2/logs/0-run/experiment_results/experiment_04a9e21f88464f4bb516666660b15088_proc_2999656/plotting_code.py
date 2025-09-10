import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper
def _safe(d, *keys):
    for k in keys:
        d = d.get(k, {})
    return d


# iterate over datasets
for dset, logs in experiment_data.items():

    # -------- FIG 1: loss curves --------------
    try:
        tr_loss = logs.get("losses", {}).get("train", [])
        val_loss = logs.get("losses", {}).get("val", [])
        if tr_loss and val_loss:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            epochs = range(1, len(tr_loss) + 1)
            ax[0].plot(epochs, tr_loss, label="Train")
            ax[1].plot(epochs, val_loss, label="Validation", color="orange")
            ax[0].set_title("Left: Train Loss")
            ax[1].set_title("Right: Validation Loss")
            for a in ax:
                a.set_xlabel("Epoch")
                a.set_ylabel("Loss")
                a.legend()
            fig.suptitle(f"{dset} Loss Curves")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dset}: {e}")
        plt.close()

    # -------- FIG 2: metric curves ------------
    try:
        val_metrics = logs.get("metrics", {}).get("val", [])
        if val_metrics:
            swa = [m["swa"] for m in val_metrics]
            cwa = [m["cwa"] for m in val_metrics]
            ccwa = [m["ccwa"] for m in val_metrics]
            hwa = [m["hwa"] for m in val_metrics]
            epochs = range(1, len(swa) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, ccwa, label="CCWA")
            plt.plot(epochs, hwa, label="HWA")
            plt.title(f"{dset} Validation Metrics Across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_val_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting metric curves for {dset}: {e}")
        plt.close()

    # -------- FIG 3: best metric bar ----------
    try:
        if val_metrics:
            best_vals = {
                "SWA": max(swa),
                "CWA": max(cwa),
                "CCWA": max(ccwa),
                "HWA": max(hwa),
            }
            names, vals = zip(*best_vals.items())
            plt.figure(figsize=(6, 4))
            plt.bar(names, vals)
            plt.title(f"{dset} Best Validation Metrics")
            plt.ylabel("Best Score")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_best_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting best metric bar for {dset}: {e}")
        plt.close()

    # -------- FIG 4: confusion matrix ---------
    try:
        preds = logs.get("predictions", [])
        gts = logs.get("ground_truth", [])
        if preds and gts:
            labels = sorted(set(gts + preds))[:15]  # cap at 15 labels
            lab2idx = {lab: i for i, lab in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                if t in lab2idx and p in lab2idx:
                    cm[lab2idx[t], lab2idx[p]] += 1
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"{dset} Confusion Matrix (Top {len(labels)})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

    # -------- print headline metric ----------
    try:
        if val_metrics:
            best_ccwa = max(m["ccwa"] for m in val_metrics)
            print(f"{dset}: Best CCWA = {best_ccwa:.4f}")
    except Exception as e:
        print(f"Error printing best CCWA for {dset}: {e}")

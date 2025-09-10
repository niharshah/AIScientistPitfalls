import matplotlib.pyplot as plt
import numpy as np
import os

# ------------- paths & loading -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None:
    raise SystemExit("No experiment data to plot.")


# Helper for safe retrieval
def _get(dct, *keys, default=None):
    for k in keys:
        dct = dct.get(k, {})
    return dct if dct != {} else default


# ---------- per-dataset plots ----------
cm_plotted = 0
for dset, data in experiment_data.items():
    losses_tr = _get(data, "losses", "train", default=[])
    losses_val = _get(data, "losses", "val", default=[])
    val_metrics = _get(data, "metrics", "val", default=[])
    val_swa = [m.get("SWA") for m in val_metrics if m and "SWA" in m]

    # 1. loss curves
    try:
        plt.figure()
        if losses_tr:
            plt.plot(losses_tr, "--", label="train")
        if losses_val:
            plt.plot(losses_val, "-", label="val")
        plt.title(f"{dset} Loss Curves\nLeft: Training, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # 2. validation SWA
    try:
        if val_swa:
            plt.figure()
            plt.plot(val_swa, label="SWA")
            plt.title(f"{dset} Validation Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_val_SWA.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting SWA for {dset}: {e}")
        plt.close()

    # 3. final test metrics bar chart
    try:
        tm = data.get("test_metrics", {})
        if tm:
            labels = list(tm.keys())
            vals = list(tm.values())
            plt.figure()
            plt.bar(labels, vals)
            plt.title(f"{dset} Final Test Metrics")
            plt.ylabel("Score")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_test_metrics.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting test metrics for {dset}: {e}")
        plt.close()

    # 4. confusion matrix (limit total to 5)
    try:
        if cm_plotted < 5 and data.get("ground_truth") and data.get("predictions"):
            gt = data["ground_truth"]
            pr = data["predictions"]
            lbls = sorted(set(gt))
            if len(lbls) <= 30:
                cm = np.zeros((len(lbls), len(lbls)), dtype=int)
                idx = {l: i for i, l in enumerate(lbls)}
                for t, p in zip(gt, pr):
                    cm[idx[t], idx[p]] += 1
                plt.figure(figsize=(4, 4))
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.title(f"{dset} Confusion Matrix")
                plt.xticks(range(len(lbls)), lbls, rotation=90, fontsize=6)
                plt.yticks(range(len(lbls)), lbls, fontsize=6)
                for i in range(len(lbls)):
                    for j in range(len(lbls)):
                        plt.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=5,
                        )
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{dset}_confusion.png"))
                plt.close()
                cm_plotted += 1
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

    # Print test metrics
    print(f"{dset} test metrics:", data.get("test_metrics", {}))

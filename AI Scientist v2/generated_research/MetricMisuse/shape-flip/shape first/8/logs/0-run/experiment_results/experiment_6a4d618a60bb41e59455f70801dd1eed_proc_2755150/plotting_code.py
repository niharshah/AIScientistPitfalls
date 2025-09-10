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

# ---------- iterate over datasets ----------
for dset_name, dct in experiment_data.items():
    losses = dct.get("losses", {})
    metrics = dct.get("metrics", {})
    preds = np.array(dct.get("predictions", []))
    gts = np.array(dct.get("ground_truth", []))
    epochs = list(range(1, len(losses.get("train", [])) + 1))

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        if losses.get("train"):
            plt.plot(epochs, losses["train"], label="Train", color="tab:blue")
        if losses.get("val"):
            plt.plot(epochs, losses["val"], label="Validation", color="tab:orange")
        plt.title(f"{dset_name} – Training/Validation Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {dset_name}: {e}")
        plt.close()

    # ---------- plot 2: SWA curves ----------
    try:
        plt.figure()
        if metrics.get("train_swa"):
            plt.plot(epochs, metrics["train_swa"], label="Train SWA", color="tab:green")
        if metrics.get("val_swa"):
            plt.plot(
                epochs, metrics["val_swa"], label="Validation SWA", color="tab:red"
            )
        plt.title(f"{dset_name} – Shape-Weighted Accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dset_name}: {e}")
        plt.close()

    # ---------- compute final metrics ----------
    test_swa = metrics.get("test_swa", np.nan)
    test_acc = (
        (preds == gts).mean() if len(preds) == len(gts) and len(preds) else np.nan
    )

    # ---------- plot 3: final test SWA bar ----------
    try:
        plt.figure()
        xs = np.arange(2)
        ys = [test_swa, 0.5]  # baseline random for binary
        plt.bar(xs, ys, color=["tab:purple", "tab:gray"])
        plt.xticks(xs, ["Model SWA", "Random Baseline"])
        plt.title(f"{dset_name} – Final Test SWA")
        plt.ylabel("SWA")
        fname = os.path.join(working_dir, f"{dset_name}_test_swa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test SWA bar for {dset_name}: {e}")
        plt.close()

    # ---------- plot 4: confusion matrix bar ----------
    try:
        if len(preds) == len(gts) and len(preds):
            tp = int(((preds == 1) & (gts == 1)).sum())
            fp = int(((preds == 1) & (gts == 0)).sum())
            tn = int(((preds == 0) & (gts == 0)).sum())
            fn = int(((preds == 0) & (gts == 1)).sum())
            plt.figure()
            bars = [tp, fp, tn, fn]
            labels = ["TP", "FP", "TN", "FN"]
            plt.bar(
                range(4), bars, color=["tab:green", "tab:red", "tab:blue", "tab:orange"]
            )
            plt.xticks(range(4), labels)
            plt.title(f"{dset_name} – Confusion Matrix Counts")
            plt.ylabel("Count")
            fname = os.path.join(working_dir, f"{dset_name}_confusion_counts.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot for {dset_name}: {e}")
        plt.close()

    # ---------- print evaluation metrics ----------
    print(
        f"{dset_name.upper()} – Test SWA: {test_swa:.3f}, Test Accuracy: {test_acc:.3f}"
    )

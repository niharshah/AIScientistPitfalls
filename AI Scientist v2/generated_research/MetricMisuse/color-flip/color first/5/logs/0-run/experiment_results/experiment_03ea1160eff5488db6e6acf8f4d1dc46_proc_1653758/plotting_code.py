import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, info in experiment_data.items():
    # ------------- collect -------------
    losses = info.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    val_metrics = info.get("metrics", {}).get("val", [])
    test_metrics = info.get("metrics", {}).get("test", {})
    preds = np.asarray(info.get("predictions", []))
    gts = np.asarray(info.get("ground_truth", []))
    n_epochs = max(len(train_loss), len(val_loss))

    # ---- 1. loss curves ----
    try:
        if train_loss and val_loss:
            plt.figure(figsize=(6, 4))
            epochs = np.arange(1, n_epochs + 1)
            plt.plot(epochs, train_loss, label="train")
            plt.plot(epochs, val_loss, linestyle="--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} — Train vs Val Loss\n(Left: train, Right: val)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset}: {e}")
        plt.close()

    # ---- 2. validation metric curves ----
    try:
        if val_metrics:
            keys = ["acc", "cwa", "swa", "ccwa"]
            epochs = np.arange(1, len(val_metrics) + 1)
            plt.figure(figsize=(6, 4))
            for k in keys:
                vals = [m.get(k, np.nan) for m in val_metrics]
                plt.plot(epochs, vals, label=k.upper())
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.title(f"{dset} — Validation Metrics over Epochs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_val_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot for {dset}: {e}")
        plt.close()

    # ---- 3. test metrics bar ----
    try:
        if test_metrics:
            plt.figure(figsize=(6, 4))
            metric_names = ["acc", "cwa", "swa", "ccwa"]
            values = [test_metrics.get(m, np.nan) for m in metric_names]
            plt.bar(metric_names, values, color="skyblue")
            plt.ylim(0, 1)
            plt.title(f"{dset} — Test Metrics Summary")
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dset}_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metric bar for {dset}: {e}")
        plt.close()

    # ---- 4. confusion matrix ----
    try:
        if preds.size and gts.size:
            n_cls = int(max(gts.max(), preds.max()) + 1)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset} — Confusion Matrix")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # ---- print test metrics ----
    if test_metrics:
        print(f"\n{dset} TEST METRICS")
        for k, v in test_metrics.items():
            print(f"{k.upper():5s}: {v:.3f}")

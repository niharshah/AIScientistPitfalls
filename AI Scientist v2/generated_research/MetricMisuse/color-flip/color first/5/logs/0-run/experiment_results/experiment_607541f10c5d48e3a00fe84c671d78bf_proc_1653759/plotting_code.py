import matplotlib.pyplot as plt
import numpy as np
import os

# ---- setup ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH_cluster_hist"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)

    # 1. loss curves -------------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, exp["losses"]["train"], label="train")
        plt.plot(epochs, exp["losses"]["val"], linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH_cluster_hist — Loss Curves\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_cluster_hist_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. validation metric curves ----------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        vals = exp["metrics"]["val"]
        for key, lab in zip(
            ["acc", "cwa", "swa", "ccwa"], ["ACC", "CWA", "SWA", "CCWA"]
        ):
            plt.plot(epochs, [m[lab.lower()] for m in vals], label=lab)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH_cluster_hist — Validation Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_cluster_hist_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot: {e}")
        plt.close()

    # 3. test metrics bar -------------------------------------------------------
    try:
        plt.figure(figsize=(5, 4))
        test_m = exp["metrics"]["test"]
        names = ["acc", "cwa", "swa", "ccwa"]
        vals = [test_m.get(k, np.nan) for k in names]
        plt.bar(names, vals, color="skyblue")
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title("SPR_BENCH_cluster_hist — Test Metrics")
        fname = os.path.join(working_dir, "SPR_BENCH_cluster_hist_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()

    # 4. silhouette per cluster -------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        sil = exp["silhouette"]
        keys = sorted(sil.keys())
        vals = [sil[k] for k in keys]
        plt.bar(keys, vals, color="orange")
        plt.xlabel("Cluster ID")
        plt.ylabel("Mean Silhouette (shifted)")
        plt.title("SPR_BENCH_cluster_hist — Cluster Silhouette Scores")
        fname = os.path.join(working_dir, "SPR_BENCH_cluster_hist_silhouette.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating silhouette plot: {e}")
        plt.close()

    # 5. confusion matrix -------------------------------------------------------
    try:
        gt = np.array(exp["ground_truth"], dtype=int)
        pr = np.array(exp["predictions"], dtype=int)
        n_cls = max(gt.max(), pr.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gt, pr):
            cm[g, p] += 1
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH_cluster_hist — Confusion Matrix")
        fname = os.path.join(working_dir, "SPR_BENCH_cluster_hist_confusion.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---- print test metrics ----
    print("Held-out test metrics:")
    for k, v in exp["metrics"]["test"].items():
        print(f"{k.upper():5s}: {v:.3f}")

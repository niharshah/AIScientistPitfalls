import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    run = experiment_data["SPR_BENCH"]
    train_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    val_swa = run["metrics"]["val_swa"]
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    labels = sorted(set(gts))  # numeric indices

    # ---------- 1) Loss curves ----------
    try:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, "b-o", label="Train Loss")
        plt.plot(epochs, val_loss, "r-o", label="Val Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2) Validation SWA ----------
    try:
        # filter out potential None placeholders
        swa_vals = [v for v in val_swa if v is not None]
        swa_epochs = range(1, len(swa_vals) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(swa_epochs, swa_vals, "g-s")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc")
        plt.title("SPR_BENCH: Validation SWA over Epochs")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------- 3) Prediction vs Ground Truth counts ----------
    try:
        gt_counts = [np.sum(gts == l) for l in labels]
        pred_counts = [np.sum(preds == l) for l in labels]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Label Index")
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Test Set – GT vs Predictions")
        plt.xticks(x, [str(l) for l in labels])
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_GT_vs_Pred.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating count comparison plot: {e}")
        plt.close()

    # ---------- quick metrics printout ----------
    try:
        test_swa = run["metrics"]["test"]["swa"]
        print(f"Overall Test SWA: {test_swa:.3f}")
        print("Label counts (GT, Pred):")
        for l, gt_c, pr_c in zip(labels, gt_counts, pred_counts):
            print(f"  Label {l}: {gt_c} / {pr_c}")
    except Exception as e:
        print(f"Error printing metrics: {e}")

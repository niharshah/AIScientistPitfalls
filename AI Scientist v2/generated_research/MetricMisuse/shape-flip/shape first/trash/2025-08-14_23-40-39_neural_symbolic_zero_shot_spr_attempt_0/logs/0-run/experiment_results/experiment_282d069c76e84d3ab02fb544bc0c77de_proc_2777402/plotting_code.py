import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    opt_data = experiment_data.get("optimizer_type", {})
    optimizers = list(opt_data.keys())
    epochs = range(1, len(next(iter(opt_data.values()))["metrics"]["train_loss"]) + 1)

    # -------- Fig 1: Loss curves --------
    try:
        plt.figure(figsize=(6, 4))
        for opt in optimizers:
            m = opt_data[opt]["metrics"]
            plt.plot(epochs, m["train_loss"], label=f"{opt} train")
            plt.plot(epochs, m["val_loss"], "--", label=f"{opt} val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- Fig 2: Validation BPS curves --------
    try:
        plt.figure(figsize=(6, 4))
        for opt in optimizers:
            bps = opt_data[opt]["metrics"]["val_bps"]
            plt.plot(epochs, bps, label=opt)
        plt.xlabel("Epoch")
        plt.ylabel("BPS")
        plt.title("SPR_BENCH: Validation BPS per Optimizer")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_bps_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating BPS curve plot: {e}")
        plt.close()

    # -------- Fig 3: Final dev/test BPS bar chart --------
    try:
        dev_scores = [opt_data[o]["final_scores"]["dev_bps"] for o in optimizers]
        test_scores = [opt_data[o]["final_scores"]["test_bps"] for o in optimizers]
        x = np.arange(len(optimizers))
        width = 0.35

        plt.figure(figsize=(6, 4))
        plt.bar(x - width / 2, dev_scores, width, label="Dev")
        plt.bar(x + width / 2, test_scores, width, label="Test")
        plt.xticks(x, optimizers)
        plt.ylabel("BPS")
        plt.title("SPR_BENCH: Final BPS Scores")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_final_bps_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final BPS bar plot: {e}")
        plt.close()

    # -------- Fig 4: Confusion matrix for Adam (dev) --------
    try:
        import itertools

        opt = optimizers[0]  # first optimizer (e.g., Adam)
        preds = opt_data[opt]["predictions"]["dev"]
        labels = opt_data[opt]["ground_truth"]["dev"]
        n_cls = max(labels) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1

        plt.figure(figsize=(5, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH Dev Confusion Matrix ({opt})")
        tick_marks = np.arange(n_cls)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        plt.tight_layout()
        fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{opt}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

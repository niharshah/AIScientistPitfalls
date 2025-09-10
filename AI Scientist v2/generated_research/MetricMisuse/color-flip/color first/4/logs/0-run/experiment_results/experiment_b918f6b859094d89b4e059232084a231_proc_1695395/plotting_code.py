import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    bench = experiment_data["token_order_shuffled"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    bench = None

if bench:
    losses_tr = bench["losses"]["train"]
    losses_val = bench["losses"]["val"]
    metrics_val = bench["metrics"]["val"]  # list of dicts
    preds = np.array(bench.get("predictions", []))
    gts = np.array(bench.get("ground_truth", []))

    epochs = np.arange(1, len(losses_tr) + 1)

    # ---------- plot losses ----------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- plot validation metrics ----------
    try:
        plt.figure()
        acc = [m["acc"] for m in metrics_val]
        cwa = [m["CWA"] for m in metrics_val]
        swa = [m["SWA"] for m in metrics_val]
        comp = [m["CompWA"] for m in metrics_val]
        plt.plot(epochs, acc, label="ACC")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, comp, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ---------- plot confusion matrix ----------
    try:
        if preds.size and gts.size:
            num_classes = int(max(max(preds), max(gts))) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH Confusion Matrix")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        else:
            print("Predictions or ground truths missing; skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print final metrics ----------
    if metrics_val:
        last = metrics_val[-1]
        print("Final Validation Metrics:", last)
    if preds.size and gts.size:
        print("Confusion Matrix:\n", cm)

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data ----------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["AtomicGlyphEmbedding"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = list(range(1, len(ed["losses"]["train"]) + 1))

    # ---------- 1) loss curves ---------- #
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (AtomicGlyphEmbedding)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2) validation metrics ---------- #
    try:
        plt.figure()
        val_metrics = ed["metrics"]["val"]
        for metric in ["CWA", "SWA", "GCWA"]:
            vals = [m.get(metric, np.nan) for m in val_metrics]
            plt.plot(epochs, vals, label=metric)
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Metrics (AtomicGlyphEmbedding)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metric curve: {e}")
        plt.close()

    # ---------- print final test metrics ---------- #
    print("Final test metrics:", ed["metrics"]["test"])

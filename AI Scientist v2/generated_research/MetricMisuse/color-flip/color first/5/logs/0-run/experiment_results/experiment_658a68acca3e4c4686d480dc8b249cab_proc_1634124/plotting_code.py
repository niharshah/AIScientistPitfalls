import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    bench_key = experiment_data["ngram_range_tuning"]["SPR_BENCH"]
    runs = bench_key["runs"]
    best_ngram = bench_key.get("best_ngram", None)
    test_metrics = bench_key.get("metrics", {}).get("test", {})
    # ---- 1. Loss curves ----
    try:
        plt.figure(figsize=(6, 4))
        for r in runs:
            ngram = r["ngram"]
            epochs = np.arange(1, len(r["losses"]["train"]) + 1)
            plt.plot(epochs, r["losses"]["train"], label=f"{ngram} train")
            plt.plot(epochs, r["losses"]["val"], linestyle="--", label=f"{ngram} val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH — Train vs Val Loss per n-gram\n(Left: train solid, Right: val dashed)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---- 2. Validation accuracy curves ----
    try:
        plt.figure(figsize=(6, 4))
        for r in runs:
            ngram = r["ngram"]
            accs = [m["acc"] for m in r["metrics"]["val"]]
            epochs = np.arange(1, len(accs) + 1)
            plt.plot(epochs, accs, label=f"{ngram}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH — Validation Accuracy across Epochs")
        plt.legend(title="n-gram range")
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---- 3. Best-model test metrics ----
    try:
        plt.figure(figsize=(6, 4))
        metric_names = ["acc", "cwa", "swa", "compwa"]
        values = [test_metrics.get(m, np.nan) for m in metric_names]
        plt.bar(metric_names, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title(f"SPR_BENCH — Test Metrics (best n-gram {best_ngram})")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()

    # ---- Print metrics ----
    if test_metrics:
        print("Best n-gram:", best_ngram)
        for k, v in test_metrics.items():
            print(f"{k.upper():6s}: {v:.3f}")

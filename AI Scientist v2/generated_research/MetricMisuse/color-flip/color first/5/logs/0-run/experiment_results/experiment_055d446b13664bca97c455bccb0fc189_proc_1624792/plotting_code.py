import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- load data ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("batch_size", {}).get("SPR_BENCH", {}).get("runs", [])

# helper to avoid crashes if no data
if not runs:
    print("No runs found in experiment_data; nothing to plot.")
else:
    # ---------- PLOT 1: Loss curves ----------
    try:
        plt.figure(figsize=(7, 5))
        for run in runs:
            bs = run["batch_size"]
            epochs = range(1, len(run["losses"]["train"]) + 1)
            plt.plot(epochs, run["losses"]["train"], label=f"Train bs={bs}")
            plt.plot(epochs, run["losses"]["val"], linestyle="--", label=f"Val bs={bs}")
        plt.title("SPR_BENCH Loss Curves (Train vs Val)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        out = os.path.join(working_dir, "SPR_BENCH_loss_curves_bs.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- PLOT 2: Validation accuracy per epoch ----------
    try:
        plt.figure(figsize=(7, 5))
        for run in runs:
            bs = run["batch_size"]
            accs = [ep["acc"] for ep in run["metrics"]["val"]]
            plt.plot(range(1, len(accs) + 1), accs, label=f"bs={bs}")
        plt.title("SPR_BENCH Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        out = os.path.join(working_dir, "SPR_BENCH_val_accuracy_bs.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Error creating val accuracy plot: {e}")
        plt.close()

    # ---------- PLOT 3: Final test metrics ----------
    try:
        metrics = ["acc", "cwa", "swa", "compwa"]
        x = np.arange(len(runs))
        width = 0.18
        plt.figure(figsize=(8, 5))
        for i, m in enumerate(metrics):
            vals = [run["metrics"]["test"][m] for run in runs]
            plt.bar(x + i * width, vals, width=width, label=m.upper())
        plt.xticks(x + width * 1.5, [f"bs={run['batch_size']}" for run in runs])
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics vs Batch Size")
        plt.legend()
        out = os.path.join(working_dir, "SPR_BENCH_test_metrics_bs.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ---------- print summary ----------
    print("\nFinal Test Metrics")
    for run in runs:
        bs = run["batch_size"]
        m = run["metrics"]["test"]
        print(
            f"bs={bs:3}: ACC={m['acc']:.3f}  CWA={m['cwa']:.3f}  SWA={m['swa']:.3f}  CompWA={m['compwa']:.3f}"
        )

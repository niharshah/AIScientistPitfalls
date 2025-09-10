import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    sweep = experiment_data["weight_decay"]["SPR_BENCH"]
    runs = sweep["runs"]
    decays = sorted(runs.keys(), key=lambda x: float(x))
    best_decay = sweep["best_decay"]
    # ---------- FIGURE 1 : loss curves ----------
    try:
        plt.figure()
        for d in decays:
            tr = runs[d]["losses"]["train"]
            val = runs[d]["losses"]["val"]
            epochs = np.arange(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"train wd={d}")
            plt.plot(epochs, val, "--", label=f"val wd={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "SPR_BENCH: Training & Validation Loss vs Epoch\n(Weight-Decay Sweep)"
        )
        plt.legend(fontsize="small")
        fp = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(fp)
        plt.close()
    except Exception as e:
        print(f"Error creating loss-curve plot: {e}")
        plt.close()

    # ---------- FIGURE 2 : validation accuracy ----------
    try:
        plt.figure()
        for d in decays:
            accs = [m["acc"] for m in runs[d]["metrics"]["val"]]
            epochs = np.arange(1, len(accs) + 1)
            plt.plot(epochs, accs, label=f"wd={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy vs Epoch\n(Weight-Decay Sweep)")
        plt.legend(fontsize="small")
        fp = os.path.join(working_dir, "SPR_BENCH_val_accuracy_weight_decay.png")
        plt.savefig(fp)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- FIGURE 3 : test metrics for best decay ----------
    try:
        best_run = runs[str(best_decay)]["metrics"]["test"]
        metrics_names = list(best_run.keys())
        metrics_values = [best_run[m] for m in metrics_names]
        plt.figure()
        plt.bar(metrics_names, metrics_values, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"SPR_BENCH: Test Metrics for Best Weight Decay ({best_decay})")
        fp = os.path.join(
            working_dir, f"SPR_BENCH_test_metrics_best_decay_{best_decay}.png"
        )
        plt.savefig(fp)
        plt.close()
    except Exception as e:
        print(f"Error creating test-metric plot: {e}")
        plt.close()

    # ---------- print summary ----------
    print(f"Best decay: {best_decay}")
    print("Test metrics:", best_run)

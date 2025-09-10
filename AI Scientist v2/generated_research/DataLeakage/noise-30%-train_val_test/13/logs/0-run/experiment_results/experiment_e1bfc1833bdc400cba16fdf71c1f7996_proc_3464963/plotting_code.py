import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_runs = experiment_data.get("weight_decay", {})
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    # ------------- Figure 1: loss curves ----------------
    try:
        plt.figure()
        for i, (run_name, run_dict) in enumerate(wd_runs.items()):
            epochs = run_dict["epochs"]
            plt.plot(
                epochs,
                run_dict["losses"]["train"],
                linestyle="-",
                marker="o",
                color=colors[i % len(colors)],
                label=f"{run_name} train",
            )
            plt.plot(
                epochs,
                run_dict["losses"]["val"],
                linestyle="--",
                marker="x",
                color=colors[i % len(colors)],
                label=f"{run_name} val",
            )
        plt.title(
            "SPR_BENCH: Training vs Validation Loss\n(TransformerClassifier, varying weight decay)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------- Figure 2: F1 curves ------------------
    try:
        plt.figure()
        for i, (run_name, run_dict) in enumerate(wd_runs.items()):
            epochs = run_dict["epochs"]
            plt.plot(
                epochs,
                run_dict["metrics"]["train_f1"],
                linestyle="-",
                marker="o",
                color=colors[i % len(colors)],
                label=f"{run_name} train",
            )
            plt.plot(
                epochs,
                run_dict["metrics"]["val_f1"],
                linestyle="--",
                marker="x",
                color=colors[i % len(colors)],
                label=f"{run_name} val",
            )
        plt.title(
            "SPR_BENCH: Training vs Validation Macro-F1\n(TransformerClassifier, varying weight decay)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # ------------- Figure 3: test F1 bar chart ----------
    try:
        plt.figure()
        run_names = list(wd_runs.keys())
        test_f1s = [wd_runs[r]["test_f1"] for r in run_names]
        x = np.arange(len(run_names))
        plt.bar(x, test_f1s, color=colors[: len(run_names)])
        plt.xticks(x, run_names, rotation=45)
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 by Weight Decay")
        for idx, val in enumerate(test_f1s):
            plt.text(idx, val + 0.01, f"{val:.2f}", ha="center", va="bottom")
        fname = os.path.join(working_dir, "SPR_BENCH_test_f1_weight_decay.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar chart: {e}")
        plt.close()

    print("Plots saved to", working_dir)

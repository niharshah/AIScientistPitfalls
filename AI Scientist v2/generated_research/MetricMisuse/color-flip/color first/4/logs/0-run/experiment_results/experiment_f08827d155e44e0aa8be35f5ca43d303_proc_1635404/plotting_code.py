import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ------------------ #
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick exit if nothing to plot
if not experiment_data:
    print("No experiment data found. Nothing to plot.")
else:
    # ---------- reshape data for easy access ---------- #
    keys = sorted(experiment_data.keys())  # e.g. embed_dim_16 ...
    epochs = len(next(iter(experiment_data.values()))["SPR_BENCH"]["losses"]["train"])

    # gather per-epoch curves
    losses_train = {k: experiment_data[k]["SPR_BENCH"]["losses"]["train"] for k in keys}
    losses_val = {k: experiment_data[k]["SPR_BENCH"]["losses"]["val"] for k in keys}
    val_metrics = {k: experiment_data[k]["SPR_BENCH"]["metrics"]["val"] for k in keys}

    # helper to pull metric list from dict list
    def extract(metric_name, data_list):
        return [d[metric_name] for d in data_list]

    # ----------------------- FIGURE 1: LOSS CURVES ----------------------- #
    try:
        plt.figure()
        for k in keys:
            plt.plot(range(1, epochs + 1), losses_train[k], label=f"{k} train")
            plt.plot(
                range(1, epochs + 1), losses_val[k], linestyle="--", label=f"{k} val"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs. Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------- FIGURE 2: VALIDATION ACCURACY ------------------- #
    try:
        plt.figure()
        for k in keys:
            acc = extract("acc", val_metrics[k])
            plt.plot(range(1, epochs + 1), acc, label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Validation Accuracy Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------ FIGURE 3: FINAL-EPOCH WEIGHTED METRIC COMPARISON (BAR) ------- #
    try:
        metrics_names = ["acc", "cwa", "swa", "pcwa"]
        x = np.arange(len(keys))  # embed dims on x-axis
        width = 0.18
        plt.figure(figsize=(8, 5))
        for i, m in enumerate(metrics_names):
            vals = [val_metrics[k][-1][m] for k in keys]
            plt.bar(x + i * width - width * 1.5, vals, width=width, label=m.upper())
        plt.xticks(x, [k.split("_")[-1] for k in keys])
        plt.ylim(0, 1)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Final-Epoch Weighted Accuracy Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric bar chart: {e}")
        plt.close()

    # ------------------- Print final epoch numbers ------------------- #
    print("\nFinal-epoch Validation Metrics:")
    for k in keys:
        last = val_metrics[k][-1]
        print(
            f"{k}: ACC={last['acc']:.3f}  CWA={last['cwa']:.3f}  SWA={last['swa']:.3f}  PCWA={last['pcwa']:.3f}"
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if not experiment_data:
    print("No experiment data found to plot.")
    exit()


# ---------- helper ----------
def get_metric_list(metrics_list, key):
    return [m[key] for m in metrics_list]


# limit to at most 5 epochs when plotting many epochs
def plot_x(vals):
    step = max(1, len(vals) // 8)
    return range(1, len(vals) + 1, step), vals[::step]


# ---------- iterate over experiments ----------
for exp_name, exp in experiment_data.items():
    tr_losses = exp["losses"]["train"]
    val_losses = exp["losses"]["val"]
    tr_metrics = exp["metrics"]["train"]
    val_metrics = exp["metrics"]["val"]
    test_metrics = exp["metrics"]["test"]

    # -------- plot 1: loss curves --------
    try:
        plt.figure()
        plt.plot(*plot_x(tr_losses), label="Train")
        plt.plot(*plot_x(val_losses), label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{exp_name} Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{exp_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {exp_name}: {e}")
        plt.close()

    # -------- per-metric curves --------
    for metric_key in ["CWA", "SWA", "GCWA"]:
        try:
            plt.figure()
            plt.plot(*plot_x(get_metric_list(tr_metrics, metric_key)), label="Train")
            plt.plot(
                *plot_x(get_metric_list(val_metrics, metric_key)), label="Validation"
            )
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel(metric_key)
            plt.title(f"{exp_name} {metric_key} vs Epoch")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{exp_name}_{metric_key}_curves.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error plotting {metric_key} for {exp_name}: {e}")
            plt.close()

    # -------- plot 5: test metrics --------
    try:
        plt.figure()
        labels = ["CWA", "SWA", "GCWA"]
        vals = [test_metrics[k] for k in labels]
        plt.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"{exp_name} Test-set Metrics\nLeft: CWA, Center: SWA, Right: GCWA")
        plt.savefig(os.path.join(working_dir, f"{exp_name}_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting test metrics for {exp_name}: {e}")
        plt.close()

    # -------- console print --------
    print(f"Test metrics for {exp_name}: {test_metrics}")

import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- load data ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "synthetic_SPR"
run_key = "transformer_simclr"
run = experiment_data.get(run_key, {})


def unpack(store, path):
    """return epochs, values arrays given nested path list/tuple"""
    try:
        items = store
        for p in path:
            items = items[p]
        ep, vals = zip(*items)
        return np.array(ep), np.array(vals)
    except Exception:
        return np.array([]), np.array([])


plot_count, max_plots = 0, 5

# 1. Train / Val loss curves
if plot_count < max_plots:
    try:
        ep_tr, tr_loss = unpack(run, ("losses", "train"))
        ep_va, va_loss = unpack(run, ("losses", "val"))
        if len(ep_tr):
            plt.figure()
            plt.plot(ep_tr, tr_loss, label="Train")
            plt.plot(ep_va, va_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"Loss Curves ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.legend()
            fname = f"{dataset_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()
    plot_count += 1


# Helper to plot a single metric curve
def metric_curve(metric_name, color):
    try:
        ep, vals = unpack(run, ("metrics", metric_name))
        if len(ep):
            plt.figure()
            plt.plot(ep, vals, color=color)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.title(
                f"{metric_name} over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
            )
            fname = f"{dataset_name}_{metric_name}_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error plotting {metric_name}: {e}")
        plt.close()


# 2-4. Metric curves
for m, c in [("SWA", "tab:blue"), ("CWA", "tab:orange"), ("CompWA", "tab:green")]:
    if plot_count >= max_plots:
        break
    metric_curve(m, c)
    plot_count += 1

# 5. Final metric bar chart
if plot_count < max_plots:
    try:
        finals = []
        labels = []
        for m in ["SWA", "CWA", "CompWA"]:
            ep, vals = unpack(run, ("metrics", m))
            if len(vals):
                finals.append(vals[-1])
                labels.append(m)
        if finals:
            x = np.arange(len(finals))
            plt.figure()
            plt.bar(x, finals, color="skyblue")
            plt.xticks(x, labels)
            plt.ylabel("Final Value")
            plt.title(f"Final Metric Values ({dataset_name})")
            fname = f"{dataset_name}_final_metrics_bar.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating final metric bar chart: {e}")
        plt.close()
    plot_count += 1

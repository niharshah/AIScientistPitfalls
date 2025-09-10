import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    runs = experiment_data["learning_rate"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

lrs = sorted(runs.keys())  # e.g. ['lr_5e-04', 'lr_1e-03', ...]
if not lrs:
    print("No runs to plot; exiting.")
    exit()


# ---------- helper ----------
def plot_curve(metric_path, title, ylabel, fname):
    try:
        plt.figure()
        for lr in lrs:
            x = runs[lr]["epochs"]
            # metric_path like ('losses','train') etc.
            y = runs[lr]
            for key in metric_path:
                y = y[key]
            plt.plot(x, y, label=lr.replace("lr_", "lr="))
        plt.title(f"{title} (spr_bench)")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(working_dir, fname)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating {fname}: {e}")
        plt.close()


# ---------- 1) train loss ----------
plot_curve(
    ("losses", "train"),
    "Training Loss vs Epochs",
    "Loss",
    "spr_bench_train_loss_curves.png",
)

# ---------- 2) dev loss ----------
plot_curve(
    ("losses", "dev"),
    "Validation Loss vs Epochs",
    "Loss",
    "spr_bench_dev_loss_curves.png",
)

# ---------- 3) train PHA ----------
plot_curve(
    ("metrics", "train_PHA"),
    "Training PHA vs Epochs",
    "PHA",
    "spr_bench_train_PHA_curves.png",
)

# ---------- 4) dev PHA ----------
plot_curve(
    ("metrics", "dev_PHA"),
    "Validation PHA vs Epochs",
    "PHA",
    "spr_bench_dev_PHA_curves.png",
)

# ---------- 5) bar chart of test metrics ----------
try:
    metrics = ["SWA", "CWA", "PHA"]
    x = np.arange(len(lrs))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics):
        vals = [runs[lr]["test_metrics"][m] for lr in lrs]
        plt.bar(x + i * width, vals, width, label=m)
    plt.xticks(x + width, [lr.replace("lr_", "lr=") for lr in lrs])
    plt.ylabel("Score")
    plt.title("Test Metrics Comparison (spr_bench)")
    plt.legend()
    plt.tight_layout()
    fname = "spr_bench_test_metrics_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

print("Plotting complete; figures saved to ./working")

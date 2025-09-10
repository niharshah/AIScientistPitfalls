import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["remove_shape_features"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = exp.get("epochs", [])
    train_loss = exp.get("losses", {}).get("train", [])
    dev_loss = exp.get("losses", {}).get("dev", [])
    train_pha = exp.get("metrics", {}).get("train_PHA", [])
    dev_pha = exp.get("metrics", {}).get("dev_PHA", [])
    test_metrics = exp.get("test_metrics", {})

    # -------- Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, dev_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Color-Only Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_color_only_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- Plot 2: PHA curves
    try:
        plt.figure()
        plt.plot(epochs, train_pha, label="Train PHA")
        plt.plot(epochs, dev_pha, label="Validation PHA")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("SPR_BENCH Color-Only PHA Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_color_only_pha_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PHA curve plot: {e}")
        plt.close()

    # -------- Plot 3: Test metrics bar chart
    try:
        metrics_names = ["SWA", "CWA", "PHA"]
        metrics_vals = [test_metrics.get(m, np.nan) for m in metrics_names]
        plt.figure()
        plt.bar(
            metrics_names, metrics_vals, color=["tab:blue", "tab:orange", "tab:green"]
        )
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Color-Only Test Metrics")
        for x, v in zip(metrics_names, metrics_vals):
            plt.text(x, v + 0.01, f"{v:.2f}", ha="center")
        plt.savefig(os.path.join(working_dir, "spr_bench_color_only_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot: {e}")
        plt.close()

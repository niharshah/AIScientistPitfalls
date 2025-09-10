import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    log = experiment_data["joint_token_only"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    log = None

if log:
    epochs = log.get("epochs", [])
    train_loss = log.get("losses", {}).get("train", [])
    dev_loss = log.get("losses", {}).get("dev", [])
    train_pha = log.get("metrics", {}).get("train_PHA", [])
    dev_pha = log.get("metrics", {}).get("dev_PHA", [])
    test_metrics = log.get("test_metrics", {})

    # ------------------- Plot 1: Loss curves ---------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, dev_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench — Loss Curves (Joint Token Only)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------- Plot 2: PHA curves ----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_pha, label="Train PHA")
        plt.plot(epochs, dev_pha, label="Validation PHA")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("spr_bench — PHA Curves (Joint Token Only)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_PHA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating PHA curve: {e}")
        plt.close()

    # ------------------- Plot 3: Test metrics bar chart ----------------------
    try:
        metrics_names, metrics_vals = (
            zip(*test_metrics.items()) if test_metrics else ([], [])
        )
        plt.figure()
        plt.bar(
            metrics_names, metrics_vals, color=["tab:blue", "tab:orange", "tab:green"]
        )
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("spr_bench — Test Metrics (Joint Token Only)")
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics chart: {e}")
        plt.close()

    # ------------------- Print metrics ---------------------------------------
    if test_metrics:
        print("Final Test Metrics:", test_metrics)
else:
    print("No log data available to plot.")

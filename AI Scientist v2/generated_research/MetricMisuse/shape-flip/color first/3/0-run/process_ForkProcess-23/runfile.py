import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["attr_only"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # ------ Plot 1: Loss curves ------
    try:
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
        plt.plot(epochs, exp["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------ Plot 2: BWA curves ------
    try:
        train_bwa = [m["BWA"] for m in exp["metrics"]["train"]]
        val_bwa = [m["BWA"] for m in exp["metrics"]["val"]]
        epochs = np.arange(1, len(train_bwa) + 1)
        plt.figure()
        plt.plot(epochs, train_bwa, label="Train BWA")
        plt.plot(epochs, val_bwa, label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title("SPR_BENCH – Training vs Validation BWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_BWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve: {e}")
        plt.close()

    # ------ Plot 3: Test metrics bar chart ------
    try:
        test_metrics = exp.get("test_metrics", {})
        labels = ["BWA", "CWA", "SWA", "StrWA"]
        values = [test_metrics.get(k, 0) for k in labels]
        plt.figure()
        plt.bar(labels, values, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Test Metrics Summary")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()

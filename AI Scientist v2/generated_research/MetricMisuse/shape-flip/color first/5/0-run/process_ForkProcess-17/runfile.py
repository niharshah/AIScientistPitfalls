import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    variants = ["directed", "undirected"]
    # ---------- FIGURE 1: loss curves ----------
    try:
        plt.figure()
        for v in variants:
            losses_tr = experiment_data["edge_direction"][v]["losses"]["train"]
            losses_vl = experiment_data["edge_direction"][v]["losses"]["val"]
            epochs = np.arange(1, len(losses_tr) + 1)
            plt.plot(epochs, losses_tr, label=f"{v}-train")
            plt.plot(epochs, losses_vl, linestyle="--", label=f"{v}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves per Variant (synthetic_or_SPR dataset)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_or_SPR.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- FIGURE 2: CmpWA curves ----------
    try:
        plt.figure()
        for v in variants:
            cpx_tr = experiment_data["edge_direction"][v]["metrics"]["train"]["CmpWA"]
            cpx_vl = experiment_data["edge_direction"][v]["metrics"]["val"]["CmpWA"]
            epochs = np.arange(1, len(cpx_tr) + 1)
            plt.plot(epochs, cpx_tr, label=f"{v}-train")
            plt.plot(epochs, cpx_vl, linestyle="--", label=f"{v}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("CmpWA Curves per Variant (synthetic_or_SPR dataset)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "CmpWA_curves_synthetic_or_SPR.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CmpWA plot: {e}")
        plt.close()

    # ---------- FIGURE 3: test metrics bar ----------
    try:
        metrics_names = ["loss", "CWA", "SWA", "CmpWA"]
        x = np.arange(len(metrics_names))
        width = 0.35
        plt.figure()
        for idx, v in enumerate(variants):
            tm = [
                experiment_data["edge_direction"][v]["test_metrics"][m]
                for m in metrics_names
            ]
            plt.bar(x + idx * width, tm, width=width, label=v)
        plt.xticks(x + width / 2, metrics_names)
        plt.ylabel("Score")
        plt.title("Test Metrics Comparison (synthetic_or_SPR dataset)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "test_metrics_comparison_synthetic_or_SPR.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # ---------- print test metrics ----------
    for v in variants:
        print(
            f"Test metrics for {v}: {experiment_data['edge_direction'][v]['test_metrics']}"
        )

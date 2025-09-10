import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["single_hop_rgcn"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = ed.get("epochs", [])
    # ---- 1. loss curve ----
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="train")
        plt.plot(epochs, ed["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Val Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # helper to plot a metric curve
    def plot_metric(mname):
        try:
            plt.figure()
            plt.plot(epochs, ed["metrics"]["train"][mname], label="train")
            plt.plot(epochs, ed["metrics"]["val"][mname], label="val")
            plt.xlabel("Epoch")
            plt.ylabel(mname)
            plt.title(f"SPR_BENCH: {mname} over Epochs")
            plt.legend()
            fname = f"SPR_BENCH_{mname}_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating {mname} plot: {e}")
            plt.close()

    # ---- 2-4. metric curves ----
    for m in ["CWA", "SWA", "CplxWA"]:
        if m in ed["metrics"]["train"]:
            plot_metric(m)

    # ---- 5. test metrics bar chart ----
    try:
        test_metrics = ed["metrics"].get("test", {})
        if test_metrics:
            labels = list(test_metrics.keys())
            values = [test_metrics[k] for k in labels]
            plt.figure()
            plt.bar(labels, values)
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Test Metrics")
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
            plt.close()
            print("Test Metrics:")
            for k, v in test_metrics.items():
                print(f"{k}: {v:.3f}")
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

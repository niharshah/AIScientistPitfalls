import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["shape_based"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = ed["epochs"]
    tr_loss = ed["losses"].get("train", [])
    tr_metrics = ed["metrics"].get("train", [])
    va_metrics = ed["metrics"].get("val", [])

    # Helper to pull metric lists safely
    def grab(metric_name, split_list):
        return [m.get(metric_name, np.nan) for m in split_list]

    metric_map = [
        ("cpx", "Complexity-Weighted Accuracy"),
        ("cwa", "Color-Weighted Accuracy"),
        ("swa", "Shape-Weighted Accuracy"),
    ]

    # 1. Training loss
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, marker="o")
        plt.title("SPR_BENCH – Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, "SPR_BENCH_train_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2-4. Accuracy curves
    for key, pretty in metric_map:
        try:
            plt.figure()
            plt.plot(epochs, grab(key, tr_metrics), marker="o", label="Train")
            if va_metrics:
                plt.plot(epochs, grab(key, va_metrics), marker="s", label="Val")
            plt.title(f"SPR_BENCH – {pretty}")
            plt.xlabel("Epoch")
            plt.ylabel(pretty)
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_{key}_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating {key} plot: {e}")
            plt.close()

    # Print final validation metrics for quick reference
    if va_metrics:
        last = va_metrics[-1]
        print(
            "Final Validation Metrics:",
            f"CpxWA={last.get('cpx'):.4f}, "
            f"CWA={last.get('cwa'):.4f}, "
            f"SWA={last.get('swa'):.4f}",
        )

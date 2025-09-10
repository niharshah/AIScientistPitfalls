import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- per-dataset curves ----------
for ds_name, ds in experiment_data.items():
    ep = ds.get("epochs", [])
    # 1) loss curves -------------------------------------------------------
    try:
        if ep and "losses" in ds:
            plt.figure()
            plt.plot(ep, ds["losses"]["train"], label="train")
            plt.plot(ep, ds["losses"]["val"], "--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} – Training vs Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()

    # helpers
    def plot_metric(metric_key, pretty):
        try:
            if ep and metric_key in ds["metrics"]["train"]:
                plt.figure()
                plt.plot(ep, ds["metrics"]["train"][metric_key], label="train")
                plt.plot(ep, ds["metrics"]["val"][metric_key], "--", label="val")
                plt.xlabel("Epoch")
                plt.ylabel(pretty)
                plt.title(f"{ds_name} – Training vs Validation {pretty}")
                plt.legend()
                plt.tight_layout()
                fname = f"{ds_name}_{metric_key}_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting {pretty} for {ds_name}: {e}")
            plt.close()

    plot_metric("cwa", "Color-Weighted Accuracy")
    plot_metric("swa", "Shape-Weighted Accuracy")
    plot_metric("cmp", "Complexity-Weighted Accuracy")

# ---------- test-set summary plot (all datasets) ----------
try:
    if experiment_data:
        ds_names = list(experiment_data.keys())
        metrics = ["cwa", "swa", "cmp"]
        bar_width = 0.25
        x = np.arange(len(ds_names))
        plt.figure(figsize=(6, 4))
        for i, m in enumerate(metrics):
            vals = [experiment_data[d]["test"].get(m, 0) for d in ds_names]
            plt.bar(x + i * bar_width, vals, width=bar_width, label=m.upper())
        plt.xticks(x + bar_width, ds_names)
        plt.ylim(0, 1)
        plt.ylabel("Weighted Accuracy")
        plt.title("Test-set Performance by Dataset")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "test_weighted_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test summary plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    for dset_name, dct in experiment_data.items():
        losses = dct.get("losses", {})
        metrics_val = dct.get("metrics", {}).get("val", [])
        metrics_test = dct.get("metrics", {}).get("test", {})

        # ---- 1. Loss curves ----
        try:
            tr_loss, val_loss = losses.get("train", []), losses.get("val", [])
            if tr_loss and val_loss:
                epochs = np.arange(1, len(tr_loss) + 1)
                plt.figure(figsize=(6, 4))
                plt.plot(epochs, tr_loss, label="train")
                plt.plot(epochs, val_loss, linestyle="--", label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{dset_name} — Train vs Val Loss")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # ---- 2. Validation metric curves ----
        try:
            if metrics_val:
                epochs = np.arange(1, len(metrics_val) + 1)
                for key in ["acc", "cwa", "swa", "ccwa"]:
                    vals = [m.get(key, np.nan) for m in metrics_val]
                    if not np.all(np.isnan(vals)):
                        plt.figure(figsize=(6, 4))
                        plt.plot(epochs, vals, marker="o")
                        plt.ylim(0, 1)
                        plt.xlabel("Epoch")
                        plt.ylabel(key.upper())
                        plt.title(
                            f"{dset_name} — Validation {key.upper()} across Epochs"
                        )
                        fname = os.path.join(working_dir, f"{dset_name}_val_{key}.png")
                        plt.savefig(fname)
                        plt.close()
        except Exception as e:
            print(f"Error creating val metric plot for {dset_name}: {e}")
            plt.close()

        # ---- 3. Test metrics bar chart ----
        try:
            if metrics_test:
                metric_names = ["acc", "cwa", "swa", "ccwa"]
                values = [metrics_test.get(m, np.nan) for m in metric_names]
                plt.figure(figsize=(6, 4))
                plt.bar(metric_names, values, color="skyblue")
                plt.ylim(0, 1)
                plt.title(f"{dset_name} — Test Metrics")
                for i, v in enumerate(values):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
                fname = os.path.join(working_dir, f"{dset_name}_test_metrics.png")
                plt.savefig(fname)
                plt.close()
                print(f"\n{dset_name} TEST METRICS:")
                for k, v in zip(metric_names, values):
                    print(f"  {k.upper():4s}: {v:.3f}")
        except Exception as e:
            print(f"Error creating test metric plot for {dset_name}: {e}")
            plt.close()

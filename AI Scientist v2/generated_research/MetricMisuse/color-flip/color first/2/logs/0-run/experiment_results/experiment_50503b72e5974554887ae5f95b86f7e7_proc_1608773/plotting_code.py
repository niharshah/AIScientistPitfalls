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
    experiment_data = {}

# ---------- iterate datasets & experiments ----------
for dset, exp_dict in experiment_data.items():
    for exp_name, blob in exp_dict.items():
        tr_loss = blob["losses"].get("train", [])
        val_loss = blob["losses"].get("val", [])
        val_metrics = blob["metrics"].get("val", [])
        test_metrics = blob["metrics"].get("test", {})
        epochs = range(1, len(tr_loss) + 1)

        # -------- plot 1: loss curves --------
        try:
            plt.figure()
            if tr_loss:
                plt.plot(epochs, tr_loss, label="Train")
            if val_loss:
                plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} – {exp_name} Loss Curves")
            plt.legend()
            fname = f"{dset}_{exp_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset}/{exp_name}: {e}")
            plt.close()

        # -------- plot 2: validation metrics curves --------
        try:
            if val_metrics:
                plt.figure()
                keys = ["CWA", "SWA", "GCWA"]
                for k in keys:
                    vals = [m[k] for m in val_metrics]
                    plt.plot(epochs, vals, label=k)
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.ylim(0, 1)
                plt.title(f"{dset} – {exp_name} Validation Metrics")
                plt.legend()
                fname = f"{dset}_{exp_name}_val_metric_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating val metric plot for {dset}/{exp_name}: {e}")
            plt.close()

        # -------- plot 3: test metrics bar --------
        try:
            if test_metrics:
                plt.figure()
                labels = list(test_metrics.keys())
                vals = [test_metrics[k] for k in labels]
                plt.bar(labels, vals)
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title(
                    f"{dset} – {exp_name} Test Metrics\nLeft: CWA, Center: SWA, Right: GCWA"
                )
                fname = f"{dset}_{exp_name}_test_metrics.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating test metric bar for {dset}/{exp_name}: {e}")
            plt.close()

        # -------- print test metrics --------
        if test_metrics:
            print(f"{dset}/{exp_name} test metrics:", test_metrics)

import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if not experiment_data:
    print("No experiment data found, nothing to plot.")
else:
    # discover data structure
    tags = list(experiment_data.keys())
    datasets = set(dname for tag in tags for dname in experiment_data[tag].keys())

    for dname in datasets:
        # collect per-tag series
        train_loss, val_loss = {}, {}
        val_cwa, val_swa, val_cva, epochs = {}, {}, {}, {}
        test_metrics = {}
        for tag in tags:
            if dname not in experiment_data[tag]:
                continue
            ed = experiment_data[tag][dname]
            train_loss[tag] = ed["losses"]["train"]
            val_loss[tag] = ed["losses"]["val"]
            epochs[tag] = list(range(1, len(train_loss[tag]) + 1))
            # validation metrics
            val_cwa[tag] = [m["cwa"] for m in ed["metrics"]["val"]]
            val_swa[tag] = [m["swa"] for m in ed["metrics"]["val"]]
            val_cva[tag] = [m["cva"] for m in ed["metrics"]["val"]]
            # test metrics
            tm = ed["metrics"].get("test", {})
            if tm:
                test_metrics[tag] = tm

        # ---------------- plot 1 : Loss curves ----------------
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for tag in train_loss:
                axes[0].plot(epochs[tag], train_loss[tag], label=tag)
                axes[1].plot(epochs[tag], val_loss[tag], label=tag)
            axes[0].set_title("Train Loss")
            axes[1].set_title("Validation Loss")
            for ax in axes:
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Cross-Entropy")
                ax.legend()
            fig.suptitle(f"{dname} Loss Curves (Left: Train, Right: Validation)")
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves for {dname}: {e}")
            plt.close()

        # ---------------- plot 2 : Validation metrics ----------------
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for tag in val_cwa:
                axes[0].plot(epochs[tag], val_cwa[tag], label=tag)
                axes[1].plot(epochs[tag], val_swa[tag], label=tag)
                axes[2].plot(epochs[tag], val_cva[tag], label=tag)
            titles = [
                "Color-Weighted Acc.",
                "Shape-Weighted Acc.",
                "Composite Variety Acc.",
            ]
            for ax, t in zip(axes, titles):
                ax.set_title(t)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.legend()
            fig.suptitle(f"{dname} Validation Metrics Over Epochs")
            fname = os.path.join(working_dir, f"{dname}_val_metrics.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating validation metric curves for {dname}: {e}")
            plt.close()

        # ---------------- plot 3 : Test metrics bar ----------------
        try:
            if test_metrics:
                width = 0.25
                tags_sorted = sorted(test_metrics.keys())
                indices = np.arange(len(tags_sorted))
                cwa_vals = [test_metrics[t]["cwa"] for t in tags_sorted]
                swa_vals = [test_metrics[t]["swa"] for t in tags_sorted]
                cva_vals = [test_metrics[t]["cva"] for t in tags_sorted]

                plt.figure(figsize=(10, 5))
                plt.bar(indices - width, cwa_vals, width, label="CWA")
                plt.bar(indices, swa_vals, width, label="SWA")
                plt.bar(indices + width, cva_vals, width, label="CVA")
                plt.xticks(indices, tags_sorted, rotation=45, ha="right")
                plt.ylabel("Accuracy")
                plt.title(f"{dname} Test Metrics Comparison")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dname}_test_metrics_bar.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating test metric bar for {dname}: {e}")
            plt.close()

    # -------- print final test metrics --------
    print("\nTest-set performance:")
    for tag in tags:
        for dname in experiment_data[tag]:
            tm = experiment_data[tag][dname]["metrics"].get("test", {})
            if tm:
                print(
                    f"{tag} | {dname}: CWA={tm['cwa']:.4f}, SWA={tm['swa']:.4f}, CVA={tm['cva']:.4f}"
                )

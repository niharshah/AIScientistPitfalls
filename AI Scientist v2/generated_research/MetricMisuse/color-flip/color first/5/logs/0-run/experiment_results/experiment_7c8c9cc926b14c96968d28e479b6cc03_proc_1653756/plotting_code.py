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
    ds_names = list(experiment_data.keys())
    test_accs = {}

    for ds in ds_names:
        ds_data = experiment_data.get(ds, {})
        losses = ds_data.get("losses", {})
        metrics_val = ds_data.get("metrics", {}).get("val", [])
        test_metrics = ds_data.get("metrics", {}).get("test", {})

        epochs = np.arange(1, len(losses.get("train", [])) + 1)

        # ---- 1. Loss curves ----
        try:
            plt.figure(figsize=(6, 4))
            if losses.get("train"):
                plt.plot(epochs, losses["train"], label="train")
            if losses.get("val"):
                plt.plot(epochs, losses["val"], linestyle="--", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds} — Train vs Val Loss\n(Left: train, Right: val)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve plot for {ds}: {e}")
            plt.close()

        # ---- 2. Validation metric curves ----
        try:
            if metrics_val:
                plt.figure(figsize=(6, 4))
                accs = [m.get("acc") for m in metrics_val]
                cwas = [m.get("cwa") for m in metrics_val]
                swas = [m.get("swa") for m in metrics_val]
                ccwas = [m.get("ccwa") for m in metrics_val]
                for arr, lbl in zip(
                    [accs, cwas, swas, ccwas], ["ACC", "CWA", "SWA", "CCWA"]
                ):
                    if any(a is not None for a in arr):
                        plt.plot(epochs, arr, label=lbl)
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.ylim(0, 1)
                plt.title(f"{ds} — Validation Metrics Across Epochs")
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_val_metrics.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating val metric plot for {ds}: {e}")
            plt.close()

        # ---- 3. Test metric bar chart ----
        try:
            if test_metrics:
                plt.figure(figsize=(6, 4))
                metric_names = ["acc", "cwa", "swa", "ccwa"]
                values = [test_metrics.get(m, np.nan) for m in metric_names]
                plt.bar(metric_names, values, color="skyblue")
                plt.ylim(0, 1)
                plt.title(f"{ds} — Test Metrics Summary")
                for i, v in enumerate(values):
                    if not np.isnan(v):
                        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
                fname = os.path.join(working_dir, f"{ds}_test_metrics.png")
                plt.savefig(fname)
                plt.close()
            test_accs[ds] = test_metrics.get("acc", np.nan)
        except Exception as e:
            print(f"Error creating test metric plot for {ds}: {e}")
            plt.close()

        # ---- Print metrics ----
        if test_metrics:
            print(f"\n{ds} TEST METRICS:")
            for k, v in test_metrics.items():
                print(f"  {k.upper():5s}: {v:.3f}")

    # ---- 4. Inter-dataset comparison ----
    if len(test_accs) > 1:
        try:
            plt.figure(figsize=(6, 4))
            names = list(test_accs.keys())
            vals = [test_accs[n] for n in names]
            plt.bar(names, vals, color="lightgreen")
            plt.ylim(0, 1)
            plt.title("Test Accuracy Comparison Across Datasets")
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, "dataset_test_accuracy_comparison.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating cross-dataset plot: {e}")
            plt.close()

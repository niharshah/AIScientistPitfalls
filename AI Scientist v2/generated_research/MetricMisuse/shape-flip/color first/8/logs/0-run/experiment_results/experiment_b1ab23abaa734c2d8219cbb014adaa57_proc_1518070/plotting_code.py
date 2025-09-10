import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------------- per-run curves -----------------
for exp_name, datasets in experiment_data.items():
    for dname, track in datasets.items():
        losses = track.get("losses", {})
        metrics = track.get("metrics", {})
        # --- loss curve ---
        try:
            plt.figure()
            if "train" in losses and losses["train"]:
                plt.plot(losses["train"], label="train")
            if "val" in losses and losses["val"]:
                plt.plot(losses["val"], label="val")
            plt.title(f"{dname} – Loss Curve ({exp_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"{dname}_{exp_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {exp_name}: {e}")
            plt.close()
        # --- accuracy curve ---
        try:
            plt.figure()
            if "train" in metrics and metrics["train"]:
                plt.plot(metrics["train"], label="train")
            if "val" in metrics and metrics["val"]:
                plt.plot(metrics["val"], label="val")
            plt.title(f"{dname} – Accuracy Curve ({exp_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = f"{dname}_{exp_name}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {exp_name}: {e}")
            plt.close()

# ----------------- complexity-weighted accuracy comparison -----------------
try:
    plt.figure()
    exp_names, cwa_values = [], []
    for exp_name, datasets in experiment_data.items():
        cwa = datasets["SPR_BENCH"].get("comp_weighted_accuracy")
        if cwa is not None:
            exp_names.append(exp_name)
            cwa_values.append(cwa)
    if exp_names:
        plt.bar(exp_names, cwa_values, color="skyblue")
        plt.title("SPR_BENCH – Complexity-Weighted Accuracy Comparison")
        plt.ylabel("Comp-WA")
        fname = "SPR_BENCH_cwa_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CWA comparison plot: {e}")
    plt.close()

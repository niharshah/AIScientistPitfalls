import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ setup ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# container for cross-dataset comparison later
final_cwa2_all = {}

# ---------------- per-dataset plots ----------------
for dset, dct in experiment_data.items():
    losses_tr = dct.get("losses", {}).get("train", [])
    losses_val = dct.get("losses", {}).get("val", [])
    metrics_val = dct.get("metrics", {}).get("val", [])
    # unpack metric lists
    cwa2_curve = [m.get("CWA2", np.nan) for m in metrics_val]
    cwa_curve = [m.get("CWA", np.nan) for m in metrics_val]
    swa_curve = [m.get("SWA", np.nan) for m in metrics_val]

    # record final CWA2 for cross-dataset comparison
    final_cwa2_all[dset] = cwa2_curve[-1] if cwa2_curve else np.nan

    # 1) loss curves
    try:
        plt.figure()
        if losses_tr:
            plt.plot(losses_tr, label="train")
        if losses_val:
            plt.plot(losses_val, label="val")
        plt.title(f"{dset} Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # 2) weighted-accuracy curves
    try:
        plt.figure()
        if cwa2_curve:
            plt.plot(cwa2_curve, label="CWA2")
        if cwa_curve:
            plt.plot(cwa_curve, label="CWA")
        if swa_curve:
            plt.plot(swa_curve, label="SWA")
        plt.title(f"{dset} Validation Weighted-Accuracy Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_weighted_acc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves for {dset}: {e}")
        plt.close()

    # 3) final metric bar chart
    try:
        plt.figure()
        bars = ["CWA2", "CWA", "SWA"]
        vals = [
            cwa2_curve[-1] if cwa2_curve else np.nan,
            cwa_curve[-1] if cwa_curve else np.nan,
            swa_curve[-1] if swa_curve else np.nan,
        ]
        plt.bar(bars, vals)
        plt.title(f"{dset} Final Validation Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dset}_final_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final metric bar chart for {dset}: {e}")
        plt.close()

# 4) cross-dataset CWA2 comparison (only one if single dataset)
try:
    plt.figure()
    plt.bar(list(final_cwa2_all.keys()), list(final_cwa2_all.values()))
    plt.title("Final CWA2 Across Datasets")
    plt.ylabel("CWA2")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "cross_dataset_final_CWA2.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison plot: {e}")
    plt.close()

# ---------------- print summary ----------------
print("\n=== Final Validation Metrics ===")
for dset, cwa2 in final_cwa2_all.items():
    dct = (
        experiment_data[dset]["metrics"]["val"][-1]
        if experiment_data[dset]["metrics"]["val"]
        else {}
    )
    print(
        f"{dset}: CWA2={cwa2:.4f}, "
        f"CWA={dct.get('CWA', np.nan):.4f}, "
        f"SWA={dct.get('SWA', np.nan):.4f}"
    )

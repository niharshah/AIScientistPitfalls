import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load saved experiment dictionary
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # convenience handle
    bs_dict = experiment_data.get("batch_size", {})
    # limit epoch-level figures to at most 5 experiments
    shown_keys = list(bs_dict.keys())[:5]

    # 1) per-experiment loss curves
    for key in shown_keys:
        try:
            ed = bs_dict[key]
            train_loss = ed["losses"]["train"]
            val_loss = ed["losses"]["val"]

            plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{key}: Train vs Val Loss")
            plt.legend()
            fname = f"{key}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {key}: {e}")
            plt.close()

    # 2) per-experiment harmonic-weighted accuracy curves
    for key in shown_keys:
        try:
            ed = bs_dict[key]
            # metrics[i] = (swa, cwa, hwa)
            hwa_vals = [m[2] for m in ed["metrics"]["val"]]
            plt.figure()
            plt.plot(hwa_vals, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic-Weighted Accuracy")
            plt.title(f"{key}: Validation HWA")
            fname = f"{key}_hwa_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating HWA plot for {key}: {e}")
            plt.close()

    # 3) aggregate bar chart of best validation HWA for each batch size
    try:
        keys = list(bs_dict.keys())
        best_hwa = [max([m[2] for m in bs_dict[k]["metrics"]["val"]]) for k in keys]
        plt.figure()
        plt.bar(range(len(keys)), best_hwa)
        plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
        plt.ylabel("Best Validation HWA")
        plt.title("Best Validation HWA vs Batch Size (spr_bench)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_best_hwa_vs_bs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregate HWA bar chart: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_dict = experiment_data.get("weight_decay", {})
    tags = list(wd_dict.keys())  # e.g. ['wd_0', 'wd_1e-5', ...]
    # numeric wd values for x-axis ordering
    wd_vals = [float(t.split("_", 1)[1]) for t in tags]
    order = np.argsort(wd_vals)
    tags = [tags[i] for i in order]
    wd_vals = [wd_vals[i] for i in order]

    # gather metrics
    test_compwa = [wd_dict[t]["metrics"]["test_compwa"] for t in tags]
    max_epochs = max(len(wd_dict[t]["epochs"]) for t in tags)

    # --------------------- Figure 1 : Test CompWA bar ----------------------
    try:
        plt.figure()
        plt.bar(range(len(wd_vals)), test_compwa)
        plt.xticks(range(len(wd_vals)), [str(w) for w in wd_vals])
        plt.xlabel("Weight Decay")
        plt.ylabel("Test CompWA")
        plt.title("Synthetic SPR_BENCH: Test CompWA vs Weight Decay")
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_test_compwa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # -------- Figure 2 : Validation CompWA curves across epochs ------------
    try:
        plt.figure()
        for t, w in zip(tags, wd_vals):
            epochs = wd_dict[t]["epochs"]
            vals = wd_dict[t]["metrics"]["val_compwa"]
            if vals:
                plt.plot(epochs, vals, marker="o", label=f"wd={w}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation CompWA")
        plt.title("Synthetic SPR_BENCH: Val CompWA over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_val_compwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA curve plot: {e}")
        plt.close()

    # -------- Figure 3 : Train/Val loss curves across epochs --------------
    try:
        plt.figure()
        for t, w in zip(tags, wd_vals):
            ep = wd_dict[t]["epochs"]
            tr = wd_dict[t]["losses"]["train"]
            vl = wd_dict[t]["losses"]["val"]
            plt.plot(ep, tr, label=f"train wd={w}")
            plt.plot(ep, vl, linestyle="--", label=f"val wd={w}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Synthetic SPR_BENCH: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "synthetic_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------------------- print summary table -----------------------------
    print("\nFinal Test CompWA per weight decay")
    for w, c in zip(wd_vals, test_compwa):
        print(f"  wd={w:<10} -> {c:.4f}")

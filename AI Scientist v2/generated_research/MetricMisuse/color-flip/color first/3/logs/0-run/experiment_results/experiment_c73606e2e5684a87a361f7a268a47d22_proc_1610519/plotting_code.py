import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------- #
# 1. Load experiment data                                          #
# ---------------------------------------------------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# Collect global best-SNWA for cross-dataset bar chart
best_snwa_global, ds_labels = [], []

# ---------------------------------------------------------------- #
# 2. Per-dataset plots                                             #
# ---------------------------------------------------------------- #
for ds_name, ds_dict in exp.items():
    losses = ds_dict.get("losses", {})
    metrics = ds_dict.get("metrics", {})
    # ----------------------------- loss curves ------------------- #
    try:
        plt.figure()
        if "train" in losses and losses["train"]:
            ep_tr = unpack(losses["train"], 0)
            tr_loss = unpack(losses["train"], 1)
            plt.plot(ep_tr, tr_loss, "--", label="train")
        if "val" in losses and losses["val"]:
            ep_val = unpack(losses["val"], 0)
            val_loss = unpack(losses["val"], 1)
            plt.plot(ep_val, val_loss, "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{ds_name}: Train vs. Val Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()
    # ----------------------------- HCSA curves ------------------- #
    try:
        plt.figure()
        if "val" in metrics and metrics["val"]:
            ep = unpack(metrics["val"], 0)
            hcs = [t[3] for t in metrics["val"]]
            plt.plot(ep, hcs, label="HCSA")
            plt.xlabel("Epoch")
            plt.ylabel("HCSA")
            plt.title(f"{ds_name}: Validation HCSA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_val_HCSA.png")
            plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating HCSA curve for {ds_name}: {e}")
        plt.close()
    # ----------------------------- best HCSA bar (per run) ------- #
    try:
        # If multiple runs/variants are stored, expect dict-of-runs style
        runs = ds_dict.get("runs", None)
        if runs:
            best_vals, lbls = [], []
            for run_name, run in runs.items():
                hcs_list = [t[3] for t in run["metrics"]["val"]]
                if hcs_list:
                    best_vals.append(max(hcs_list))
                    lbls.append(run_name)
            if best_vals:
                plt.figure()
                plt.bar(range(len(best_vals)), best_vals, tick_label=lbls)
                plt.ylabel("Best Val HCSA")
                plt.title(f"{ds_name}: Best HCSA per Run")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_best_HCSA_bar.png")
                plt.savefig(fname, dpi=150)
                plt.close()
                print(f"Saved {fname}")
        # record best SNWA for global comparison
        snwa_list = [t[4] for t in metrics.get("val", [])]
        if snwa_list:
            best_snwa_global.append(max(snwa_list))
            ds_labels.append(ds_name)
    except Exception as e:
        print(f"Error creating per-run bar chart for {ds_name}: {e}")
        plt.close()

# ---------------------------------------------------------------- #
# 3. Cross-dataset comparison plots                                #
# ---------------------------------------------------------------- #
# -------- overlay HCSA curves ----------------------------------- #
try:
    plt.figure()
    for ds_name, ds_dict in exp.items():
        metrics = ds_dict.get("metrics", {})
        if metrics.get("val"):
            ep = unpack(metrics["val"], 0)
            hcs = [t[3] for t in metrics["val"]]
            plt.plot(ep, hcs, label=ds_name)
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title("Validation HCSA: Dataset Comparison")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "comparison_val_HCSA.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating cross-dataset HCSA plot: {e}")
    plt.close()

# -------- bar chart of best SNWA -------------------------------- #
try:
    if best_snwa_global:
        plt.figure()
        plt.bar(range(len(best_snwa_global)), best_snwa_global, tick_label=ds_labels)
        plt.ylabel("Best Val SNWA")
        plt.title("Best Validation SNWA per Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(working_dir, "comparison_best_SNWA.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating global SNWA bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------- #
# 4. Summary printout                                              #
# ---------------------------------------------------------------- #
print("\nBest metrics summary:")
for ds_name, ds_dict in exp.items():
    val_metrics = ds_dict.get("metrics", {}).get("val", [])
    if val_metrics:
        hcs_list = [t[3] for t in val_metrics]
        snwa_list = [t[4] for t in val_metrics]
        ep_list = unpack(val_metrics, 0)
        best_hcs_idx = int(np.argmax(hcs_list))
        best_snwa_idx = int(np.argmax(snwa_list))
        print(
            f"{ds_name:>10}: best HCSA={hcs_list[best_hcs_idx]:.3f} at epoch {ep_list[best_hcs_idx]} | "
            f"best SNWA={snwa_list[best_snwa_idx]:.3f} at epoch {ep_list[best_snwa_idx]}"
        )

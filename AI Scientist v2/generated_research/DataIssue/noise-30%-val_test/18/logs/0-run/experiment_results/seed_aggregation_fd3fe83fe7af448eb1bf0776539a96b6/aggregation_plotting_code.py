import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load all experiment_data dicts that actually exist on disk
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_bf214b89385e445e9a6d1d1bb079a640_proc_3335811/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_b04c3429a716427cbdf81e7550f4a801_proc_3335813/experiment_data.npy",
    "experiments/2025-08-17_18-48-06_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_4b3b86e4b73f4b8481dfb9168c672ddf_proc_3335812/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------------------------------------------------------------
# 2. Aggregate per-dataset
# ------------------------------------------------------------------
agg = defaultdict(lambda: defaultdict(list))  # agg[dataset][field] -> list

for run in all_experiment_data:
    for dset, info in run.items():
        # losses
        for split in ("train", "val"):
            try:
                agg[dset][f"loss_{split}"].append(np.asarray(info["losses"][split]))
            except Exception:
                pass
        # MCC
        for split in ("train", "val"):
            try:
                agg[dset][f"mcc_{split}"].append(
                    np.asarray(info["metrics"][f"{split}_MCC"])
                )
            except Exception:
                pass
        # predictions / gts
        try:
            agg[dset]["preds"].append(np.asarray(info["predictions"]))
            agg[dset]["gts"].append(np.asarray(info["ground_truth"]))
        except Exception:
            pass

saved_files = []

# ------------------------------------------------------------------
# 3. Create aggregate plots
# ------------------------------------------------------------------
for dset, fields in agg.items():
    # --------------------------------------------------------------
    # Helper to compute mean and stderr for a list of equal-length 1D arrays
    # --------------------------------------------------------------
    def mean_se(arr_list):
        arr_stack = np.stack(arr_list, axis=0)  # (runs, epochs)
        mean = arr_stack.mean(axis=0)
        se = arr_stack.std(axis=0, ddof=1) / np.sqrt(arr_stack.shape[0])
        return mean, se

    # ensure epoch axis uses shortest run length to avoid shape mismatch
    min_epochs = (
        min(a.shape[0] for a in fields.get("loss_train", []))
        if fields.get("loss_train")
        else None
    )
    epochs = np.arange(1, min_epochs + 1) if min_epochs else None

    # =============== 3.1  Loss curves  ============================
    try:
        if epochs is not None:
            mean_train, se_train = mean_se(
                [a[:min_epochs] for a in fields["loss_train"]]
            )
            mean_val, se_val = mean_se([a[:min_epochs] for a in fields["loss_val"]])

            plt.figure()
            plt.plot(epochs, mean_train, label="train mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_train - se_train,
                mean_train + se_train,
                color="tab:blue",
                alpha=0.3,
                label="train ± SE",
            )
            plt.plot(epochs, mean_val, "--", label="val mean", color="tab:orange")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:orange",
                alpha=0.3,
                label="val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset}: Training vs Validation Loss (Aggregated)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_agg_loss_curves.png")
            plt.savefig(fname, dpi=150)
            saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # =============== 3.2  MCC curves  =============================
    try:
        if epochs is not None and "mcc_train" in fields:
            mean_train, se_train = mean_se(
                [a[:min_epochs] for a in fields["mcc_train"]]
            )
            mean_val, se_val = mean_se([a[:min_epochs] for a in fields["mcc_val"]])

            plt.figure()
            plt.plot(epochs, mean_train, label="train mean", color="tab:green")
            plt.fill_between(
                epochs,
                mean_train - se_train,
                mean_train + se_train,
                color="tab:green",
                alpha=0.3,
                label="train ± SE",
            )
            plt.plot(epochs, mean_val, "--", label="val mean", color="tab:red")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:red",
                alpha=0.3,
                label="val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title(f"{dset}: Training vs Validation MCC (Aggregated)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_agg_MCC_curves.png")
            plt.savefig(fname, dpi=150)
            saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated MCC plot for {dset}: {e}")
        plt.close()

    # =============== 3.3  Confusion Matrix ========================
    try:
        if "preds" in fields and "gts" in fields:
            cm = np.zeros((2, 2), dtype=int)
            for preds, gts in zip(fields["preds"], fields["gts"]):
                for g, p in zip(gts, preds):
                    cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title(
                f"{dset} Confusion Matrix (Aggregated)\nLeft: Ground Truth, Right: Predicted"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar()
            fname = os.path.join(working_dir, f"{dset}_agg_conf_matrix.png")
            plt.savefig(fname, dpi=150)
            saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset}: {e}")
        plt.close()

    # =============== 3.4  Numeric summary =========================
    try:
        if "mcc_val" in fields:
            final_vals = [arr[-1] for arr in fields["mcc_val"]]
            mean_final = np.mean(final_vals)
            se_final = np.std(final_vals, ddof=1) / np.sqrt(len(final_vals))
            print(f"{dset} final-epoch val MCC: {mean_final:.3f} ± {se_final:.3f}")
    except Exception as e:
        print(f"Error computing numeric summary for {dset}: {e}")

print("Saved figures:", saved_files)

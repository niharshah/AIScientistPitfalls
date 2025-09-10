import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load every experiment file that actually exists
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_9fd226ab2ef64d9284a33288ed9f6473_proc_1599370/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ab9e2f80b0a440d593b80cc5b87c64dc_proc_1599371/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e9befd8717f14ce4b854392680ee3487_proc_1599369/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            ed = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(ed)
        else:
            print(f"File not found, skipped: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------
# 2) Aggregate per-dataset across runs
# ------------------------------------------------------------------
aggregated = {}  # {dset_name: { 'train_loss': [...], 'val_loss': [...], ...}}
for ed in all_experiment_data:
    for dset_name, d in ed.items():
        agg = aggregated.setdefault(
            dset_name,
            {"train_loss": [], "val_loss": [], "cwa": [], "swa": [], "hmwa": []},
        )
        if d["losses"].get("train"):
            agg["train_loss"].append(np.array(d["losses"]["train"], dtype=np.float32))
        if d["losses"].get("val"):
            agg["val_loss"].append(np.array(d["losses"]["val"], dtype=np.float32))
        if d["metrics"].get("val"):
            cwa = [m.get("cwa", np.nan) for m in d["metrics"]["val"]]
            swa = [m.get("swa", np.nan) for m in d["metrics"]["val"]]
            hmwa = [m.get("hmwa", np.nan) for m in d["metrics"]["val"]]
            agg["cwa"].append(np.array(cwa, dtype=np.float32))
            agg["swa"].append(np.array(swa, dtype=np.float32))
            agg["hmwa"].append(np.array(hmwa, dtype=np.float32))


# ------------------------------------------------------------------
# 3) Helper to stack and align arrays (truncate to shortest length)
# ------------------------------------------------------------------
def stack_and_align(list_of_arrays):
    if not list_of_arrays:
        return None
    min_len = min(arr.size for arr in list_of_arrays)
    if min_len == 0:
        return None
    trimmed = np.stack([arr[:min_len] for arr in list_of_arrays], axis=0)
    return trimmed  # shape: (runs, epochs)


# ------------------------------------------------------------------
# 4) Produce aggregated figures
# ------------------------------------------------------------------
for dset_name, data in aggregated.items():
    epochs = None  # will be determined per plot

    # 4.1 Loss curves with mean ± SEM
    try:
        train_mat = stack_and_align(data["train_loss"])
        val_mat = stack_and_align(data["val_loss"])
        if train_mat is not None and val_mat is not None:
            epochs = np.arange(1, train_mat.shape[1] + 1)
            tr_mean = train_mat.mean(axis=0)
            tr_sem = train_mat.std(axis=0, ddof=1) / np.sqrt(train_mat.shape[0])
            val_mean = val_mat.mean(axis=0)
            val_sem = val_mat.std(axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, tr_mean, color="tab:blue", label="Train Mean")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SEM",
            )
            plt.plot(epochs, val_mean, color="tab:orange", label="Val Mean")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Aggregated Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves_aggregated.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # 4.2 Validation weighted accuracies (CWA/SWA/HMWA) with mean ± SEM
    try:
        cwa_mat = stack_and_align(data["cwa"])
        swa_mat = stack_and_align(data["swa"])
        hmwa_mat = stack_and_align(data["hmwa"])
        if cwa_mat is not None and swa_mat is not None and hmwa_mat is not None:
            epochs = np.arange(1, cwa_mat.shape[1] + 1)

            def mean_sem(mat):  # returns mean, sem
                mean = mat.mean(axis=0)
                sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
                return mean, sem

            cwa_mean, cwa_sem = mean_sem(cwa_mat)
            swa_mean, swa_sem = mean_sem(swa_mat)
            hmwa_mean, hmwa_sem = mean_sem(hmwa_mat)

            plt.figure()
            for mean, sem, label, color in [
                (cwa_mean, cwa_sem, "CWA", "tab:green"),
                (swa_mean, swa_sem, "SWA", "tab:red"),
                (hmwa_mean, hmwa_sem, "HMWA", "tab:purple"),
            ]:
                plt.plot(epochs, mean, label=f"{label} Mean", color=color)
                plt.fill_between(
                    epochs,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.3,
                    label=f"{label} ± SEM",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(f"{dset_name}: Aggregated Validation Weighted Accuracies")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dset_name}_weighted_acc_aggregated.png"
            )
            plt.savefig(fname)
            plt.close()

            # Print final epoch aggregated numbers
            final_idx = -1
            print(
                f"{dset_name} – Final Val "
                f"HMWA: {hmwa_mean[final_idx]:.4f}±{hmwa_sem[final_idx]:.4f}  "
                f"CWA: {cwa_mean[final_idx]:.4f}±{cwa_sem[final_idx]:.4f}  "
                f"SWA: {swa_mean[final_idx]:.4f}±{swa_sem[final_idx]:.4f}"
            )
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset_name}: {e}")
        plt.close()

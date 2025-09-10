import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------- setup ----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data paths provided by the user
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_8f12b6db8d204713b066b31310ee6313_proc_3462840/experiment_data.npy",
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_c829977e7a484edb8c50a0f1941f1f7e_proc_3462838/experiment_data.npy",
    "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_fd266361f77242e19c52d509bebc4631_proc_3462839/experiment_data.npy",
]

# ------------------------- load all runs ------------------------------
all_runs = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(run_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------------------- aggregate on SPR_BENCH ------------------------
dataset_key = "SPR_BENCH"
train_loss_runs, val_loss_runs = [], []
train_f1_runs, val_f1_runs = [], []
test_macroF1_list = []
epochs_aligned = None

for rd in all_runs:
    if dataset_key not in rd:
        continue
    spr = rd[dataset_key]
    if "epochs" not in spr:
        continue
    # Align all runs on the shortest epoch length
    ep_len = len(spr["epochs"])
    if epochs_aligned is None:
        epochs_aligned = spr["epochs"]
    else:
        ep_len = min(ep_len, len(epochs_aligned))
        epochs_aligned = epochs_aligned[:ep_len]
    train_loss_runs.append(np.asarray(spr["losses"]["train"])[:ep_len])
    val_loss_runs.append(np.asarray(spr["losses"]["val"])[:ep_len])
    train_f1_runs.append(np.asarray(spr["metrics"]["train"])[:ep_len])
    val_f1_runs.append(np.asarray(spr["metrics"]["val"])[:ep_len])
    if "test_macroF1" in spr:
        test_macroF1_list.append(float(spr["test_macroF1"]))

n_runs = len(train_loss_runs)
if n_runs == 0:
    print("No valid runs found for dataset", dataset_key)

# --------------------------- print test metric ------------------------
if test_macroF1_list:
    mean_test = np.mean(test_macroF1_list)
    std_test = np.std(test_macroF1_list)
    print(f"Aggregate Test Macro-F1 ({dataset_key}): {mean_test:.4f} ± {std_test:.4f}")


# ------------------------------ plots ---------------------------------
# Helper for mean & standard error
def _mean_se(arr_list):
    stack = np.vstack(arr_list)
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, se


# ------------- Loss curve (mean ± SE) ---------------------------------
try:
    if n_runs > 0:
        tr_mean, tr_se = _mean_se(train_loss_runs)
        va_mean, va_se = _mean_se(val_loss_runs)

        plt.figure()
        plt.plot(epochs_aligned, tr_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs_aligned,
            tr_mean - tr_se,
            tr_mean + tr_se,
            alpha=0.3,
            label="Train SE",
        )
        plt.plot(epochs_aligned, va_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs_aligned, va_mean - va_se, va_mean + va_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{dataset_key}: Training vs Validation Loss (Aggregate over {n_runs} runs)"
        )
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_key}_loss_curve_aggregate.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# ---------- Macro-F1 curve (mean ± SE) --------------------------------
try:
    if n_runs > 0:
        tr_mean, tr_se = _mean_se(train_f1_runs)
        va_mean, va_se = _mean_se(val_f1_runs)

        plt.figure()
        plt.plot(epochs_aligned, tr_mean, label="Train Macro-F1 (mean)")
        plt.fill_between(
            epochs_aligned,
            tr_mean - tr_se,
            tr_mean + tr_se,
            alpha=0.3,
            label="Train SE",
        )
        plt.plot(epochs_aligned, va_mean, label="Val Macro-F1 (mean)")
        plt.fill_between(
            epochs_aligned, va_mean - va_se, va_mean + va_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            f"{dataset_key}: Training vs Validation Macro-F1 (Aggregate over {n_runs} runs)"
        )
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset_key}_macroF1_curve_aggregate.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated Macro-F1 curve: {e}")
    plt.close()

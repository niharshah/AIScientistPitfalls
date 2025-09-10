import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helper ----------
def macro_f1_score(y_true, y_pred):
    labels = np.unique(y_true)
    f1s = []
    for l in labels:
        tp = np.sum((y_pred == l) & (y_true == l))
        fp = np.sum((y_pred == l) & (y_true != l))
        fn = np.sum((y_pred != l) & (y_true == l))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


# ---------- load all experiment files ----------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_687fa5fa40a640419413e780375e00fd_proc_3458422/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_9bc834d383c84bbeaccd0304531b109e_proc_3458424/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_5d26ea6dc30c48239dbc9e634d4ec574_proc_3458423/experiment_data.npy",
]
all_experiment_data = []
try:
    ai_root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(ai_root, p) if ai_root else p
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------- aggregate per-dataset ----------
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

for ds_name in dataset_names:
    # collect per-run metric arrays
    train_loss_runs, val_loss_runs = [], []
    train_f1_runs, val_f1_runs = [], []
    epochs_runs = []
    test_f1_final = []

    for exp in all_experiment_data:
        if ds_name not in exp:
            continue
        ds = exp[ds_name]
        try:
            epochs_runs.append(np.array(ds["epochs"]))
            train_loss_runs.append(np.array(ds["losses"]["train"]))
            val_loss_runs.append(np.array(ds["losses"]["val"]))
            train_f1_runs.append(np.array(ds["metrics"]["train_f1"]))
            val_f1_runs.append(np.array(ds["metrics"]["val_f1"]))

            preds = np.array(ds["predictions"])
            gts = np.array(ds["ground_truth"])
            test_f1_final.append(macro_f1_score(gts, preds))
        except Exception as e:
            print(f"Skipping a run for {ds_name} due to missing keys: {e}")

    # skip if not enough data
    n_runs = len(train_loss_runs)
    if n_runs == 0:
        continue

    # align epochs (truncate to shortest run length)
    min_len = min(map(len, epochs_runs))
    epoch_axis = epochs_runs[0][:min_len]

    def stack_and_crop(list_of_arrays):
        return np.stack([arr[:min_len] for arr in list_of_arrays], axis=0)

    train_loss = stack_and_crop(train_loss_runs)
    val_loss = stack_and_crop(val_loss_runs)
    train_f1 = stack_and_crop(train_f1_runs)
    val_f1 = stack_and_crop(val_f1_runs)

    # ---------- aggregated loss curve ----------
    try:
        plt.figure()
        mean_tr, se_tr = train_loss.mean(0), train_loss.std(0) / np.sqrt(n_runs)
        mean_val, se_val = val_loss.mean(0), val_loss.std(0) / np.sqrt(n_runs)

        plt.plot(epoch_axis, mean_tr, label="Train Mean", color="tab:blue")
        plt.fill_between(
            epoch_axis,
            mean_tr - se_tr,
            mean_tr + se_tr,
            color="tab:blue",
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epoch_axis, mean_val, label="Val Mean", color="tab:orange")
        plt.fill_between(
            epoch_axis,
            mean_val - se_val,
            mean_val + se_val,
            color="tab:orange",
            alpha=0.3,
            label="Val ± SE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Aggregated Training vs Validation Loss (n={n_runs})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ---------- aggregated F1 curve ----------
    try:
        plt.figure()
        mean_tr, se_tr = train_f1.mean(0), train_f1.std(0) / np.sqrt(n_runs)
        mean_val, se_val = val_f1.mean(0), val_f1.std(0) / np.sqrt(n_runs)

        plt.plot(epoch_axis, mean_tr, label="Train Mean", color="tab:green")
        plt.fill_between(
            epoch_axis,
            mean_tr - se_tr,
            mean_tr + se_tr,
            color="tab:green",
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epoch_axis, mean_val, label="Val Mean", color="tab:red")
        plt.fill_between(
            epoch_axis,
            mean_val - se_val,
            mean_val + se_val,
            color="tab:red",
            alpha=0.3,
            label="Val ± SE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name} Aggregated Training vs Validation Macro-F1 (n={n_runs})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_aggregated_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {ds_name}: {e}")
        plt.close()

    # ---------- print final aggregated test metric ----------
    try:
        test_f1_arr = np.array(test_f1_final, dtype=float)
        print(
            f"{ds_name} Final Test Macro-F1: "
            f"{test_f1_arr.mean():.4f} ± {test_f1_arr.std():.4f} "
            f"(n={len(test_f1_arr)})"
        )
    except Exception as e:
        print(f"Error computing aggregated test F1 for {ds_name}: {e}")

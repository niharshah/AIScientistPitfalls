import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Load all experiment_data.npy files that were provided
# -----------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_bf27f61eddc94237a911bdc8d58d4dfa_proc_3475348/experiment_data.npy",
        "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_561bc6e689bd4956adfbd96c7ab0fe6c_proc_3475349/experiment_data.npy",
        "experiments/2025-08-17_23-44-27_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_d9625dc2dea04ba2bf2978443e5a13a2_proc_3475347/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        # We rely on the environment variable if it exists, otherwise use relative path
        root_path = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root_path, experiment_data_path)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -----------------------------------------------------------------------
# 2. Aggregate metrics per dataset across different runs
# -----------------------------------------------------------------------
agg = {}  # {dataset: {'epochs': [[..],[..]], 'train_loss': [[..]], ...}}
for run_dict in all_experiment_data:
    for ds_name, ds in run_dict.items():
        ds_agg = agg.setdefault(
            ds_name,
            {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "train_f1": [],
                "val_f1": [],
            },
        )
        ds_agg["epochs"].append(np.asarray(ds.get("epochs", []), dtype=float))
        ds_agg["train_loss"].append(
            np.asarray(ds.get("losses", {}).get("train", []), dtype=float)
        )
        ds_agg["val_loss"].append(
            np.asarray(ds.get("losses", {}).get("val", []), dtype=float)
        )
        ds_agg["train_f1"].append(
            np.asarray(ds.get("metrics", {}).get("train", []), dtype=float)
        )
        ds_agg["val_f1"].append(
            np.asarray(ds.get("metrics", {}).get("val", []), dtype=float)
        )

# store final val scores for summary plot
final_val_summary = {}  # {ds: (mean, sem)}

# -----------------------------------------------------------------------
# 3. Plot aggregated curves for each dataset
# -----------------------------------------------------------------------
for ds_name, ds_agg in agg.items():
    # Skip if we do not have at least one valid run
    if not ds_agg["epochs"]:
        continue

    # Align sequence lengths (use min length across runs)
    min_len = min(len(e) for e in ds_agg["epochs"] if len(e) > 0)
    if min_len == 0:
        continue  # nothing to plot

    # Create matrices shape (n_runs, min_len)
    def stack_and_truncate(list_of_arr):
        return np.vstack([arr[:min_len] for arr in list_of_arr])

    epochs = ds_agg["epochs"][0][:min_len]  # assume identical epoch values

    train_loss_mtx = stack_and_truncate(ds_agg["train_loss"])
    val_loss_mtx = stack_and_truncate(ds_agg["val_loss"])
    train_f1_mtx = stack_and_truncate(ds_agg["train_f1"])
    val_f1_mtx = stack_and_truncate(ds_agg["val_f1"])

    n_runs = train_loss_mtx.shape[0]
    sem = lambda x: (
        np.std(x, axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(np.mean(x, axis=0))
    )

    # -------------------- Aggregated Loss curve -------------------------
    try:
        plt.figure()
        tr_mean, tr_sem = np.mean(train_loss_mtx, axis=0), sem(train_loss_mtx)
        val_mean, val_sem = np.mean(val_loss_mtx, axis=0), sem(val_loss_mtx)

        plt.plot(epochs, tr_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            alpha=0.3,
            label="Train Loss ± SEM",
        )
        plt.plot(epochs, val_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            label="Val Loss ± SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name}: Aggregated Training vs Validation Loss (Mean ± SEM)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated loss for {ds_name}: {e}")
        plt.close()

    # -------------------- Aggregated Macro-F1 curve --------------------
    try:
        plt.figure()
        tr_mean, tr_sem = np.mean(train_f1_mtx, axis=0), sem(train_f1_mtx)
        val_mean, val_sem = np.mean(val_f1_mtx, axis=0), sem(val_f1_mtx)

        plt.plot(epochs, tr_mean, label="Train Macro-F1 (mean)")
        plt.fill_between(
            epochs,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            alpha=0.3,
            label="Train Macro-F1 ± SEM",
        )
        plt.plot(epochs, val_mean, label="Val Macro-F1 (mean)")
        plt.fill_between(
            epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            label="Val Macro-F1 ± SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name}: Aggregated Training vs Validation Macro-F1 (Mean ± SEM)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_aggregated_macroF1_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated f1 for {ds_name}: {e}")
        plt.close()

    # store final val f1
    final_val_summary[ds_name] = (val_mean[-1], val_sem[-1])

# -----------------------------------------------------------------------
# 4. Cross-dataset comparison bar chart of Final Validation Macro-F1
# -----------------------------------------------------------------------
try:
    if final_val_summary:
        plt.figure()
        names = list(final_val_summary.keys())
        means = [final_val_summary[n][0] for n in names]
        sems = [final_val_summary[n][1] for n in names]

        y_pos = np.arange(len(names))
        plt.barh(y_pos, means, xerr=sems, align="center", alpha=0.7)
        plt.yticks(y_pos, names)
        plt.xlabel("Final Validation Macro-F1 (mean ± SEM)")
        plt.title("Cross-Dataset Comparison of Final Validation Macro-F1")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cross_dataset_final_val_macroF1.png"))
        plt.close()

        # print aggregated numbers
        print("Final Validation Macro-F1 (mean ± SEM):")
        for n, m, s in zip(names, means, sems):
            print(f"  {n}: {m:.4f} ± {s:.4f}")
except Exception as e:
    print(f"Error plotting cross-dataset aggregated comparison: {e}")
    plt.close()

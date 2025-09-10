import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- experiment files to aggregate --------
experiment_data_path_list = [
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_546edabd52824f229314bd6b7b3be332_proc_2678330/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_36155d2e83cb4768991eb1f7e94bbfb6_proc_2678329/experiment_data.npy",
    "experiments/2025-08-14_15-47-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_1ced583751a54325a68a1d07df49decf_proc_2678328/experiment_data.npy",
]

# -------- load data from all runs --------
all_runs = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# -------- merge by dataset --------
datasets = {}
for run in all_runs:
    for dset_name, rec in run.items():
        bucket = datasets.setdefault(
            dset_name,
            {
                "train_loss": [],
                "val_loss": [],
                "train_swa": [],
                "val_swa": [],
                "test_acc": [],
            },
        )
        # losses & metrics
        bucket["train_loss"].append(
            np.asarray(rec["losses"].get("train", []), dtype=float)
        )
        bucket["val_loss"].append(np.asarray(rec["losses"].get("val", []), dtype=float))
        bucket["train_swa"].append(
            np.asarray(rec["metrics"].get("train_swa", []), dtype=float)
        )
        bucket["val_swa"].append(
            np.asarray(rec["metrics"].get("val_swa", []), dtype=float)
        )
        preds = np.asarray(rec.get("predictions", []))
        gts = np.asarray(rec.get("ground_truth", []))
        acc = float(np.mean(preds == gts)) if len(preds) else np.nan
        bucket["test_acc"].append(acc)


# -------- helper to stack to common length --------
def stack_trim(arr_list):
    lens = [len(a) for a in arr_list if len(a)]
    if not lens:
        return None
    min_len = min(lens)
    trimmed = np.stack([a[:min_len] for a in arr_list], axis=0)  # (runs, epochs)
    return trimmed


# -------- plotting --------
for dset_name, rec in datasets.items():
    n_runs = len(rec["train_loss"])
    # Aggregate curves
    train_loss_stack = stack_trim(rec["train_loss"])
    val_loss_stack = stack_trim(rec["val_loss"])
    train_swa_stack = stack_trim(rec["train_swa"])
    val_swa_stack = stack_trim(rec["val_swa"])
    epochs_loss = (
        np.arange(1, train_loss_stack.shape[1] + 1)
        if train_loss_stack is not None
        else None
    )
    epochs_swa = (
        np.arange(1, train_swa_stack.shape[1] + 1)
        if train_swa_stack is not None
        else None
    )

    # ---- 1: aggregated loss curves ----
    try:
        if train_loss_stack is not None and val_loss_stack is not None:
            plt.figure()
            m_tr = train_loss_stack.mean(axis=0)
            m_val = val_loss_stack.mean(axis=0)
            if n_runs > 1:
                se_tr = train_loss_stack.std(axis=0, ddof=1) / np.sqrt(n_runs)
                se_val = val_loss_stack.std(axis=0, ddof=1) / np.sqrt(n_runs)
                plt.fill_between(
                    epochs_loss, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="Train SE"
                )
                plt.fill_between(
                    epochs_loss,
                    m_val - se_val,
                    m_val + se_val,
                    alpha=0.3,
                    label="Val SE",
                )
            plt.plot(epochs_loss, m_tr, label="Train Mean", color="tab:blue")
            plt.plot(epochs_loss, m_val, label="Val Mean", color="tab:orange")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Mean Train vs Val Loss (n={n_runs})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_aggregated_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ---- 2: aggregated SWA curves ----
    try:
        if train_swa_stack is not None and val_swa_stack is not None:
            plt.figure()
            m_tr = train_swa_stack.mean(axis=0)
            m_val = val_swa_stack.mean(axis=0)
            if n_runs > 1:
                se_tr = train_swa_stack.std(axis=0, ddof=1) / np.sqrt(n_runs)
                se_val = val_swa_stack.std(axis=0, ddof=1) / np.sqrt(n_runs)
                plt.fill_between(
                    epochs_swa, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="Train SE"
                )
                plt.fill_between(
                    epochs_swa,
                    m_val - se_val,
                    m_val + se_val,
                    alpha=0.3,
                    label="Val SE",
                )
            plt.plot(epochs_swa, m_tr, label="Train Mean", color="tab:green")
            plt.plot(epochs_swa, m_val, label="Val Mean", color="tab:red")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{dset_name}: Mean Train vs Val SWA (n={n_runs})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_aggregated_swa_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {dset_name}: {e}")
        plt.close()

    # ---- 3: aggregated test accuracy bar ----
    try:
        acc_arr = np.asarray(rec["test_acc"], dtype=float)
        if len(acc_arr):
            mean_acc = np.nanmean(acc_arr)
            se_acc = (
                np.nanstd(acc_arr, ddof=1) / np.sqrt(len(acc_arr))
                if len(acc_arr) > 1
                else 0.0
            )
            plt.figure()
            plt.bar(
                [0], [mean_acc], yerr=[se_acc], capsize=5, color="tab:purple", alpha=0.7
            )
            plt.ylim(0, 1)
            plt.xticks([0], ["Accuracy"])
            plt.title(f"{dset_name}: Test Accuracy Mean ± SE (n={n_runs})")
            fname = os.path.join(
                working_dir, f"{dset_name}_aggregated_test_accuracy.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset_name}: {e}")
        plt.close()

    # ---- console summary ----
    try:
        best_val_swa_per_run = [
            np.nanmax(v) if len(v) else np.nan for v in rec["val_swa"]
        ]
        mean_best_val_swa = np.nanmean(best_val_swa_per_run)
        se_best_val_swa = (
            np.nanstd(best_val_swa_per_run, ddof=1) / np.sqrt(n_runs)
            if n_runs > 1
            else 0.0
        )
        print(
            f"{dset_name}: best_val_SWA={mean_best_val_swa:.4f} ± {se_best_val_swa:.4f}, "
            f"test_acc={mean_acc:.4f} ± {se_acc:.4f}"
        )
    except Exception as e:
        print(f"Error computing summary for {dset_name}: {e}")

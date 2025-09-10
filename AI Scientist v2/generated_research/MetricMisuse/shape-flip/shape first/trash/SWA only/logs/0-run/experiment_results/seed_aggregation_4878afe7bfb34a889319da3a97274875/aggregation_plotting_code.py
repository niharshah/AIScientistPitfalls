import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# Load every experiment_data.npy that was listed
experiment_data_path_list = [
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_9353283ca7d545a485a6f93fa657e215_proc_2605921/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_647371cb525d44828c5ed80f5fb73616_proc_2605923/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_fb624b67b304425b9292578488552f23_proc_2605922/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---------------------------------------------------------------
def stack_metric(runs, key_chain):
    """Collect metric along key_chain and return list of 1-D arrays (one per run)"""
    vals = []
    for run in runs:
        cur = run
        try:
            for k in key_chain:
                cur = cur[k]
            if cur:  # non-empty
                vals.append(np.asarray(cur, dtype=float))
        except Exception:
            continue
    return vals


def pad_and_stack(list_of_arrays):
    """truncate to min length, then stack"""
    if not list_of_arrays:
        return None
    min_len = min(len(a) for a in list_of_arrays)
    clipped = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return clipped


def class_hist(lst, n_cls=2):
    h = [0] * n_cls
    for x in lst:
        if 0 <= x < n_cls:
            h[x] += 1
    return np.array(h, dtype=float)


# ---------------------------------------------------------------
# Organise runs by dataset
dataset_runs = {}
for run in all_experiment_data:
    for model_name, ds_dict in run.items():
        for ds_name, rec in ds_dict.items():
            dataset_runs.setdefault(ds_name, []).append(rec)

# ---------------------------------------------------------------
for ds_name, runs in dataset_runs.items():
    n_runs = len(runs)
    if n_runs == 0:
        continue

    # ---------- Aggregated Loss Curves --------------------------------
    try:
        train_lists = stack_metric(runs, ["losses", "train"])
        val_lists = stack_metric(runs, ["losses", "val"])
        if train_lists and val_lists:
            train_mat = pad_and_stack(train_lists)
            val_mat = pad_and_stack(val_lists)
            epochs = np.arange(train_mat.shape[1])

            plt.figure()
            # Train
            m = train_mat.mean(axis=0)
            se = train_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Train mean", color="tab:blue")
            plt.fill_between(epochs, m - se, m + se, color="tab:blue", alpha=0.2)

            # Val
            m = val_mat.mean(axis=0)
            se = val_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Val mean", color="tab:orange")
            plt.fill_between(epochs, m - se, m + se, color="tab:orange", alpha=0.2)

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"Aggregated Loss Curves (mean ± SE)\nDataset: {ds_name} | Runs: {n_runs}"
            )
            plt.legend()
            fname = f"{ds_name}_aggregated_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ---------- Aggregated Accuracy Curves ----------------------------
    try:
        tr_acc_lists = stack_metric(runs, ["metrics", "train"])
        if tr_acc_lists:
            tr_acc_lists = [[m["acc"] for m in run_list] for run_list in tr_acc_lists]
        va_acc_lists = stack_metric(runs, ["metrics", "val"])
        if va_acc_lists:
            va_acc_lists = [[m["acc"] for m in run_list] for run_list in va_acc_lists]

        if tr_acc_lists and va_acc_lists:
            tr_mat = pad_and_stack([np.asarray(l, dtype=float) for l in tr_acc_lists])
            va_mat = pad_and_stack([np.asarray(l, dtype=float) for l in va_acc_lists])
            epochs = np.arange(tr_mat.shape[1])

            plt.figure()
            # Train
            m = tr_mat.mean(axis=0)
            se = tr_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Train mean", color="tab:green")
            plt.fill_between(epochs, m - se, m + se, color="tab:green", alpha=0.2)
            # Val
            m = va_mat.mean(axis=0)
            se = va_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Val mean", color="tab:red")
            plt.fill_between(epochs, m - se, m + se, color="tab:red", alpha=0.2)

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"Aggregated Accuracy Curves (mean ± SE)\nDataset: {ds_name} | Runs: {n_runs}"
            )
            plt.legend()
            fname = f"{ds_name}_aggregated_accuracy_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # ---------- Aggregated SWA Curves ---------------------------------
    try:
        tr_swa_lists = stack_metric(runs, ["metrics", "train"])
        if tr_swa_lists:
            tr_swa_lists = [[m["swa"] for m in run_list] for run_list in tr_swa_lists]
        va_swa_lists = stack_metric(runs, ["metrics", "val"])
        if va_swa_lists:
            va_swa_lists = [[m["swa"] for m in run_list] for run_list in va_swa_lists]

        if tr_swa_lists and va_swa_lists:
            tr_mat = pad_and_stack([np.asarray(l, dtype=float) for l in tr_swa_lists])
            va_mat = pad_and_stack([np.asarray(l, dtype=float) for l in va_swa_lists])
            epochs = np.arange(tr_mat.shape[1])

            plt.figure()
            # Train
            m = tr_mat.mean(axis=0)
            se = tr_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Train mean", color="tab:purple")
            plt.fill_between(epochs, m - se, m + se, color="tab:purple", alpha=0.2)
            # Val
            m = va_mat.mean(axis=0)
            se = va_mat.std(axis=0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, m, label="Val mean", color="tab:brown")
            plt.fill_between(epochs, m - se, m + se, color="tab:brown", alpha=0.2)

            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(
                f"Aggregated SWA Curves (mean ± SE)\nDataset: {ds_name} | Runs: {n_runs}"
            )
            plt.legend()
            fname = f"{ds_name}_aggregated_swa_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot for {ds_name}: {e}")
        plt.close()

    # ---------- Aggregated Test Distribution --------------------------
    try:
        pred_hists = []
        gt_hists = []
        for rec in runs:
            preds = rec.get("predictions", [])
            gts = rec.get("ground_truth", [])
            if preds and gts:
                pred_hists.append(class_hist(preds))
                gt_hists.append(class_hist(gts))
        if pred_hists and gt_hists:
            pred_mat = np.vstack(pred_hists)
            gt_mat = np.vstack(gt_hists)
            pred_mean = pred_mat.mean(axis=0)
            gt_mean = gt_mat.mean(axis=0)
            x = np.arange(len(pred_mean))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_mean, width, label="Ground Truth mean")
            plt.bar(x + width / 2, pred_mean, width, label="Predictions mean")
            plt.xlabel("Class")
            plt.ylabel("Average Count")
            plt.title(
                f"Aggregated Test Distribution\nDataset: {ds_name} | Runs: {n_runs}"
            )
            plt.legend()
            fname = f"{ds_name}_aggregated_test_distribution.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated distribution plot for {ds_name}: {e}")
        plt.close()

    # ---------- Print Aggregated Test Metrics -------------------------
    try:
        metrics_list = [rec.get("metrics", {}).get("test", {}) for rec in runs]
        if metrics_list:
            # collect numeric keys
            keys = {k for m in metrics_list for k in m.keys()}
            for k in keys:
                vals = [m[k] for m in metrics_list if k in m]
                if vals:
                    vals = np.asarray(vals, dtype=float)
                    mean = vals.mean()
                    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    print(
                        f"{ds_name} | Test {k}: {mean:.4f} ± {se:.4f} (n={len(vals)})"
                    )
    except Exception as e:
        print(f"Error aggregating test metrics for {ds_name}: {e}")

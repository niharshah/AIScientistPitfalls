import matplotlib.pyplot as plt
import numpy as np
import os

# working directory -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# helper to aggregate arrays of unequal length (truncate to min length)
def stack_and_trim(list_of_1d_arrays):
    if len(list_of_1d_arrays) == 0:
        return None
    min_len = min([len(a) for a in list_of_1d_arrays])
    trimmed = np.stack([a[:min_len] for a in list_of_1d_arrays], axis=0)
    return trimmed  # shape (n_runs, min_len)


# -----------------------------------------------------------------------------
# 1. Load every experiment_data dict
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_d4c9db252f844ac78db418307ef65820_proc_3027375/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_e3f510ef01504da8821ead5b89234ed3_proc_3027374/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_59ed195897034fec92ce4e9d1705440c_proc_3027376/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(d)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# -----------------------------------------------------------------------------
# 2. Collect every dataset that appears in at least one run
dataset_names = set()
for d in all_experiment_data:
    dataset_names.update(d.keys())

# -----------------------------------------------------------------------------
# 3. Iterate over datasets and build plots with aggregation
for ds_name in dataset_names:
    # Gather per-run records for this dataset
    per_run_recs = [d[ds_name] for d in all_experiment_data if ds_name in d]
    n_runs = len(per_run_recs)
    if n_runs == 0:
        continue  # nothing to aggregate

    # ======================= Plot A: Pre-train contrastive loss ===============
    try:
        pre_losses = [
            np.asarray(r["losses"]["pretrain"])
            for r in per_run_recs
            if "pretrain" in r["losses"]
        ]
        stacked = stack_and_trim(pre_losses)
        if stacked is not None:
            mean = stacked.mean(axis=0)
            se = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            epochs = np.arange(1, len(mean) + 1)
            plt.figure()
            plt.plot(epochs, mean, color="C0", label="Mean")
            plt.fill_between(
                epochs, mean - se, mean + se, color="C0", alpha=0.3, label="±SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Contrastive Loss")
            plt.title(
                f"{ds_name}: Contrastive Pre-training Loss (Mean ± SE, n={n_runs})"
            )
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_agg_pretrain_loss.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated pretrain loss plot for {ds_name}: {e}")
        plt.close()

    # ======================= Plot B: Fine-tune train/val loss ================
    try:
        train_losses = [
            np.asarray(r["losses"]["train"])
            for r in per_run_recs
            if "train" in r["losses"]
        ]
        val_losses = [
            np.asarray(r["losses"]["val"]) for r in per_run_recs if "val" in r["losses"]
        ]
        s_train = stack_and_trim(train_losses)
        s_val = stack_and_trim(val_losses)
        if s_train is not None and s_val is not None:
            m_train, se_train = s_train.mean(0), s_train.std(0, ddof=1) / np.sqrt(
                s_train.shape[0]
            )
            m_val, se_val = s_val.mean(0), s_val.std(0, ddof=1) / np.sqrt(
                s_val.shape[0]
            )
            epochs = np.arange(1, len(m_train) + 1)
            plt.figure()
            plt.plot(epochs, m_train, "--", color="C1", label="Train Mean")
            plt.fill_between(
                epochs,
                m_train - se_train,
                m_train + se_train,
                color="C1",
                alpha=0.25,
                label="Train ±SE",
            )
            plt.plot(epochs, m_val, "-", color="C2", label="Val Mean")
            plt.fill_between(
                epochs,
                m_val - se_val,
                m_val + se_val,
                color="C2",
                alpha=0.25,
                label="Val ±SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy Loss")
            plt.title(f"{ds_name}: Fine-tuning Loss (Mean ± SE, n={n_runs})")
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_agg_train_val_loss.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated train/val loss plot for {ds_name}: {e}")
        plt.close()

    # ======================= Plot C: Validation metrics ======================
    try:
        val_accs = [
            np.asarray(r["metrics"]["val_acc"])
            for r in per_run_recs
            if "val_acc" in r["metrics"]
        ]
        val_acas = [
            np.asarray(r["metrics"]["val_aca"])
            for r in per_run_recs
            if "val_aca" in r["metrics"]
        ]
        s_acc = stack_and_trim(val_accs)
        s_aca = stack_and_trim(val_acas)
        if s_acc is not None and s_aca is not None:
            m_acc, se_acc = s_acc.mean(0), s_acc.std(0, ddof=1) / np.sqrt(
                s_acc.shape[0]
            )
            m_aca, se_aca = s_aca.mean(0), s_aca.std(0, ddof=1) / np.sqrt(
                s_aca.shape[0]
            )
            epochs = np.arange(1, len(m_acc) + 1)
            plt.figure()
            plt.plot(epochs, m_acc, color="C3", label="Val Acc Mean")
            plt.fill_between(
                epochs,
                m_acc - se_acc,
                m_acc + se_acc,
                color="C3",
                alpha=0.25,
                label="Acc ±SE",
            )
            plt.plot(epochs, m_aca, color="C4", label="Val ACA Mean")
            plt.fill_between(
                epochs,
                m_aca - se_aca,
                m_aca + se_aca,
                color="C4",
                alpha=0.25,
                label="ACA ±SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{ds_name}: Validation Metrics (Mean ± SE, n={n_runs})")
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_agg_val_metrics.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metrics plot for {ds_name}: {e}")
        plt.close()

    # ======================= Plot D: Test metrics summary ====================
    try:
        metric_names = ["acc", "swa", "cwa", "aca"]
        collected = {k: [] for k in metric_names}
        for r in per_run_recs:
            for k in metric_names:
                if k in r["test"]:
                    collected[k].append(r["test"][k])
        means = []
        stderrs = []
        labels_present = []
        for k in metric_names:
            vals = collected[k]
            if len(vals) > 0:
                labels_present.append(k)
                vals_arr = np.asarray(vals)
                means.append(vals_arr.mean())
                stderrs.append(vals_arr.std(ddof=1) / np.sqrt(len(vals_arr)))
        if len(means):
            x = np.arange(len(means))
            plt.figure()
            plt.bar(
                x,
                means,
                yerr=stderrs,
                capsize=5,
                color="skyblue",
                alpha=0.8,
                label="Mean ± SE",
            )
            plt.xticks(x, labels_present)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(
                f"{ds_name}: Test Metrics Summary (Mean ± SE, n={n_runs})\nLeft→Right: "
                + ", ".join(labels_present)
            )
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_agg_test_metrics.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics plot for {ds_name}: {e}")
        plt.close()

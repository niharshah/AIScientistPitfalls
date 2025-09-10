import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list all experiment_data.npy files ----------
experiment_data_path_list = [
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_12890bd9d48540318fbfaf3545328e0e_proc_3104090/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_e11c1b7ba9ba45a49ac3767fbd07f18c_proc_3104089/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f075a46ca04c41688727b04ec4ffde06_proc_3104091/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
try:
    for path in experiment_data_path_list:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp_data = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---------- aggregate helpers ----------
def stack_and_trim(list_of_1d_arrays):
    min_len = min(len(a) for a in list_of_1d_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_1d_arrays], axis=0)
    mean = trimmed.mean(axis=0)
    stderr = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
    return mean, stderr, min_len


# ---------- collect per-dataset data across runs ----------
datasets = {}
for run_data in all_experiment_data:
    for dset, info in run_data.items():
        entry = datasets.setdefault(
            dset, {"train_losses": [], "val_losses": [], "metrics": {}}
        )
        entry["train_losses"].append(np.asarray(info["losses"]["train"]))
        entry["val_losses"].append(np.asarray(info["losses"]["val"]))
        for m_name, m_vals in info["metrics"].items():
            entry["metrics"].setdefault(m_name, []).append(np.asarray(m_vals))

# ---------- plotting ----------
best_summary = {}
for dset, coll in datasets.items():
    # ----- aggregate loss curves -----
    try:
        train_mean, train_se, n_epochs = stack_and_trim(coll["train_losses"])
        val_mean, val_se, _ = stack_and_trim(coll["val_losses"])
        epochs = np.arange(1, n_epochs + 1)
        plt.figure()
        plt.plot(epochs, train_mean, label="Train (mean)")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ±SE",
        )
        plt.plot(epochs, val_mean, label="Val (mean)")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ±SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{dset} – Mean Loss w/ Standard Error (N={len(coll['train_losses'])})"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_agg_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # ----- aggregate primary metric curves (take first metric key) -----
    try:
        metric_name = next(iter(coll["metrics"]))
        metric_runs = coll["metrics"][metric_name]
        m_mean, m_se, n_epochs = stack_and_trim(metric_runs)
        epochs = np.arange(1, n_epochs + 1)
        plt.figure()
        plt.plot(epochs, m_mean, label=f"{metric_name} (mean)")
        plt.fill_between(epochs, m_mean - m_se, m_mean + m_se, alpha=0.3, label="±SE")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{dset} – Mean {metric_name} w/ Standard Error")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_agg_{metric_name}_curve.png"))
        plt.close()

        # store best values per run for summary
        best_vals_per_run = [np.max(arr[:n_epochs]) for arr in metric_runs]
        best_summary[dset] = (
            np.mean(best_vals_per_run),
            np.std(best_vals_per_run, ddof=1) / np.sqrt(len(best_vals_per_run)),
        )
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dset}: {e}")
        plt.close()

# ---------- bar chart of best validation metric across datasets ----------
try:
    if len(best_summary) > 0:
        names, stats = zip(*best_summary.items())
        means = [s[0] for s in stats]
        ses = [s[1] for s in stats]
        plt.figure()
        plt.bar(names, means, yerr=ses, capsize=5, color="skyblue")
        plt.ylabel("Best Validation Metric (mean ± SE)")
        plt.title("Across-run Mean Best Validation Metric per Dataset")
        plt.savefig(os.path.join(working_dir, "datasets_best_val_metric_mean_se.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison bar plot: {e}")
    plt.close()

# ---------- textual summary ----------
for dset, (m, se) in best_summary.items():
    print(f"{dset}: best validation metric = {m:.4f} ± {se:.4f}")

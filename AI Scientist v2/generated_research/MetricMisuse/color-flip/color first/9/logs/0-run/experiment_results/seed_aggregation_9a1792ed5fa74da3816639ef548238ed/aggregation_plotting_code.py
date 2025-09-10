import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list of experiment result files ----------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_af9aaf87cb934e499ebe2529f271496d_proc_1723214/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2329043f0d264819923ed990f30fd49d_proc_1723215/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_be2fd1459597455582959a2f2e779858_proc_1723216/experiment_data.npy",
]

# ---------- load all experiment dicts ----------
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# early exit if nothing to plot
if not all_experiment_data:
    quit()


# ---------- aggregation helpers ----------
def aggregate_curves(runs, key_chain):
    """
    Collect curves (epoch, value) from all runs for the provided key_chain
    (e.g. ["losses","train"]) and return:
        epochs_sorted, mean_values, se_values
    Missing points in some runs are ignored for that epoch.
    """
    epoch_dict = {}
    for run in runs:
        d = run
        try:
            for k in key_chain:
                d = d[k]
        except KeyError:
            continue
        for ep, val in d:
            epoch_dict.setdefault(ep, []).append(val)

    if not epoch_dict:
        return [], [], []
    epochs_sorted = sorted(epoch_dict.keys())
    means, ses = [], []
    for ep in epochs_sorted:
        vals = np.array(epoch_dict[ep], dtype=float)
        means.append(np.mean(vals))
        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    return np.array(epochs_sorted), np.array(means), np.array(ses)


def simple_accuracy(gt, pr):
    gt = np.asarray(gt)
    pr = np.asarray(pr)
    return float(np.mean(gt == pr)) if gt.size else 0.0


# ---------- assume all runs share the same dataset name ----------
dataset_name = next(iter(all_experiment_data[0]))
runs_for_dataset = [
    {dataset_name: r[dataset_name]}[dataset_name] for r in all_experiment_data
]

# ---------- PLOT 1: aggregated losses ----------
try:
    e_train, m_train, se_train = aggregate_curves(runs_for_dataset, ["losses", "train"])
    e_val, m_val, se_val = aggregate_curves(runs_for_dataset, ["losses", "val"])
    if e_train.size:
        plt.figure()
        plt.plot(e_train, m_train, label="Train Loss (mean)")
        plt.fill_between(
            e_train, m_train - se_train, m_train + se_train, alpha=0.3, label="Train SE"
        )
        if e_val.size:
            plt.plot(e_val, m_val, label="Val Loss (mean)")
            plt.fill_between(
                e_val, m_val - se_val, m_val + se_val, alpha=0.3, label="Val SE"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        subtitle = (
            "Train vs Val Loss (mean ± SE)" if e_val.size else "Train Loss (mean ± SE)"
        )
        plt.title(f"{dataset_name} Aggregated Loss Curve\n{subtitle}")
        plt.legend()
        fname = f"{dataset_name.lower()}_aggregated_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ---------- PLOT 2: aggregated validation metrics ----------
try:
    metrics = {
        "CWA": ["metrics", "val"],
        "SWA": ["metrics", "val"],
        "DWHS": ["metrics", "val"],
    }
    # collect each metric separately
    epochs_cwa, mean_cwa, se_cwa = aggregate_curves(
        runs_for_dataset, ["metrics", "val"]
    )  # we will parse later
    # The structure of val metrics appears to be (epoch, cwa, swa, dwhs)
    # So manually collect each
    epoch_dict = {}
    for run in runs_for_dataset:
        for epoch, cwa, swa, dwhs in run.get("metrics", {}).get("val", []):
            epoch_dict.setdefault(epoch, {"cwa": [], "swa": [], "dwhs": []})
            epoch_dict[epoch]["cwa"].append(cwa)
            epoch_dict[epoch]["swa"].append(swa)
            epoch_dict[epoch]["dwhs"].append(dwhs)
    if epoch_dict:
        epochs_sorted = sorted(epoch_dict.keys())

        def calc(arrs):
            arrs = np.array(arrs, float)
            return arrs.mean(), (
                arrs.std(ddof=1) / np.sqrt(len(arrs))
                if len(arrs) > 1
                else (arrs.mean(), 0.0)
            )

        m_cwa, se_cwa = zip(*[calc(epoch_dict[e]["cwa"]) for e in epochs_sorted])
        m_swa, se_swa = zip(*[calc(epoch_dict[e]["swa"]) for e in epochs_sorted])
        m_dwhs, se_dwhs = zip(*[calc(epoch_dict[e]["dwhs"]) for e in epochs_sorted])
        m_cwa, se_cwa = np.array(m_cwa), np.array(se_cwa)
        m_swa, se_swa = np.array(m_swa), np.array(se_swa)
        m_dwhs, se_dwhs = np.array(m_dwhs), np.array(se_dwhs)

        plt.figure()
        plt.plot(epochs_sorted, m_cwa, label="CWA (mean)")
        plt.fill_between(
            epochs_sorted, m_cwa - se_cwa, m_cwa + se_cwa, alpha=0.2, label="CWA SE"
        )
        plt.plot(epochs_sorted, m_swa, label="SWA (mean)")
        plt.fill_between(
            epochs_sorted, m_swa - se_swa, m_swa + se_swa, alpha=0.2, label="SWA SE"
        )
        plt.plot(epochs_sorted, m_dwhs, label="DWHS (mean)")
        plt.fill_between(
            epochs_sorted,
            m_dwhs - se_dwhs,
            m_dwhs + se_dwhs,
            alpha=0.2,
            label="DWHS SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dataset_name} Validation Metrics\nMean ± SE over Runs")
        plt.legend()
        fname = f"{dataset_name.lower()}_aggregated_val_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metric plot: {e}")
    plt.close()

# ---------- final test accuracies ----------
try:
    accuracies = []
    for run in runs_for_dataset:
        preds = np.asarray(run.get("predictions", []))
        gts = np.asarray(run.get("ground_truth", []))
        if preds.size and gts.size:
            accuracies.append(simple_accuracy(gts, preds))
    if accuracies:
        acc_mean = np.mean(accuracies)
        acc_se = (
            np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
            if len(accuracies) > 1
            else 0.0
        )
        print(f"Aggregated Test Accuracy (mean ± SE): {acc_mean:.4f} ± {acc_se:.4f}")
        # optional bar plot
        try:
            plt.figure()
            plt.bar([0], [acc_mean], yerr=[acc_se], color="skyblue", capsize=5)
            plt.xticks([0], ["Accuracy"])
            plt.ylim(0, 1)
            plt.title(
                f"{dataset_name} Test Accuracy\nMean ± SE over {len(accuracies)} runs"
            )
            fname = f"{dataset_name.lower()}_aggregated_test_accuracy.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy bar plot: {e}")
            plt.close()
except Exception as e:
    print(f"Error computing aggregated test accuracy: {e}")

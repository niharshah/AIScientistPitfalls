import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------- load all experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f0918925d4e949ebaac2496e3f8016ac_proc_310214/experiment_data.npy",
        "experiments/2025-07-27_23-49-14_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_80af0b151e51497581fda7e5f2498d90_proc_310213/experiment_data.npy",
        "None/experiment_data.npy",  # may fail
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        try:
            full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
            ed = np.load(full, allow_pickle=True).item()
            all_experiment_data.append(ed)
        except Exception as e:
            print(f"Error loading {p}: {e}")
except Exception as e:
    print(f"Unexpected error when building experiment list: {e}")
    all_experiment_data = []


# --------------------------------------------------- helper
def collect(metric_path):
    """Return list of np.array from every run (if present)"""
    out = []
    for d in all_experiment_data:
        spr = d.get("SPR_BENCH", {})
        cur = spr
        for k in metric_path:
            if cur is None:
                break
            cur = cur.get(k, None)
        if cur is not None and len(cur) > 0:
            out.append(np.asarray(cur, dtype=float))
    return out


def pad_to_same_length(arrays):
    if not arrays:
        return np.array([])
    max_len = max(len(a) for a in arrays)
    padded = []
    for a in arrays:
        if len(a) < max_len:
            pad = np.full(max_len - len(a), np.nan)
            padded.append(np.concatenate([a, pad]))
        else:
            padded.append(a)
    return np.vstack(padded)


# --------------------------------------------------- Plot 1: Accuracy with mean ± SE
try:
    train_runs = collect(["metrics", "train_acc"])
    val_runs = collect(["metrics", "val_acc"])
    if train_runs and val_runs:
        train_mat = pad_to_same_length(train_runs)
        val_mat = pad_to_same_length(val_runs)
        epochs = np.arange(1, train_mat.shape[1] + 1)

        train_mean = np.nanmean(train_mat, axis=0)
        val_mean = np.nanmean(val_mat, axis=0)
        train_se = np.nanstd(train_mat, axis=0) / np.sqrt(train_mat.shape[0])
        val_se = np.nanstd(val_mat, axis=0) / np.sqrt(val_mat.shape[0])

        plt.figure()
        plt.plot(epochs, train_mean, label="Train Mean")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ±1 SE",
        )
        plt.plot(epochs, val_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ±1 SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy per Epoch\nMean ± Standard Error over Runs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve_mean_se.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating mean accuracy plot: {e}")
    plt.close()

# --------------------------------------------------- Plot 2: Loss with mean ± SE
try:
    train_runs = collect(["losses", "train"])
    val_runs = collect(["losses", "val"])
    if train_runs and val_runs:
        train_mat = pad_to_same_length(train_runs)
        val_mat = pad_to_same_length(val_runs)
        epochs = np.arange(1, train_mat.shape[1] + 1)

        train_mean = np.nanmean(train_mat, axis=0)
        val_mean = np.nanmean(val_mat, axis=0)
        train_se = np.nanstd(train_mat, axis=0) / np.sqrt(train_mat.shape[0])
        val_se = np.nanstd(val_mat, axis=0) / np.sqrt(val_mat.shape[0])

        plt.figure()
        plt.plot(epochs, train_mean, label="Train Mean")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ±1 SE",
        )
        plt.plot(epochs, val_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ±1 SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss per Epoch\nMean ± Standard Error over Runs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_mean_se.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()

# --------------------------------------------------- Plot 3: Final Test Metrics mean ± SE
try:
    metric_names = ["overall_acc", "SWA", "CWA", "ZSRTA"]
    collected = {m: [] for m in metric_names}

    for d in all_experiment_data:
        spr = d.get("SPR_BENCH", {})
        for m in metric_names:
            v = spr.get("metrics", {}).get(m, None)
            if v is None:
                continue
            # ZSRTA may be a list; take last entry
            if isinstance(v, (list, tuple)):
                v = v[-1] if len(v) else None
            if v is not None:
                collected[m].append(float(v))

    final_means, final_ses, labels = [], [], []
    for m in metric_names:
        vals = collected[m]
        if vals:
            labels.append(m.upper() if m != "overall_acc" else "Overall Acc")
            mean = np.mean(vals)
            se = np.std(vals) / np.sqrt(len(vals))
            final_means.append(mean)
            final_ses.append(se)

    if final_means:
        plt.figure()
        x = np.arange(len(final_means))
        plt.bar(x, final_means, yerr=final_ses, capsize=5)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.xticks(x, labels)
        plt.title("SPR_BENCH Final Test Metrics\nMean ± Standard Error over Runs")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_mean_se.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating aggregated final metrics plot: {e}")
    plt.close()

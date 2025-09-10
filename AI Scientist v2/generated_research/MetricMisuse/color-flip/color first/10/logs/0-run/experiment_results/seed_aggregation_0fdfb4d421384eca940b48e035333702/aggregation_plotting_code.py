import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load multiple experiment_data.npy ----------
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_265e9e2517404e44983f4da199dac005_proc_1763776/experiment_data.npy",
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_708ee68fe6494d05a656f6fa4b403ef1_proc_1763778/experiment_data.npy",
        "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_036033e1ff944e399c3f4c77790bf97d_proc_1763777/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# Utility: collect a list of np.arrays (one per run) for a given metric
def collect(metric_key, bench_name="SPR_BENCH"):
    arrs = []
    for ed in all_experiment_data:
        bench = ed.get(bench_name, {})
        metrics = bench.get("metrics", {})
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            arrs.append(np.array(metrics[metric_key]))
    return arrs


def trim_and_stack(arr_list):
    if not arr_list:
        return np.array([]), np.array([])
    min_len = min(map(len, arr_list))
    trimmed = np.array([a[:min_len] for a in arr_list])
    mean = trimmed.mean(0)
    sem = trimmed.std(0, ddof=1) / np.sqrt(trimmed.shape[0])
    return mean, sem


# ---------- aggregate ----------
train_loss_m, train_loss_sem = trim_and_stack(collect("train_loss"))
val_loss_m, val_loss_sem = trim_and_stack(collect("val_loss"))
val_CWA_m, val_CWA_sem = trim_and_stack(collect("val_CWA"))
val_SWA_m, val_SWA_sem = trim_and_stack(collect("val_SWA"))
val_CWA2_m, val_CWA2_sem = trim_and_stack(collect("val_CWA2"))

epochs = np.arange(1, len(train_loss_m) + 1)

# ---------- plots ----------
# 1) Aggregated loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss_m, label="Train Loss (mean)")
    plt.fill_between(
        epochs,
        train_loss_m - train_loss_sem,
        train_loss_m + train_loss_sem,
        alpha=0.3,
        label="Train Loss ± SEM",
    )
    plt.plot(epochs, val_loss_m, label="Val Loss (mean)")
    plt.fill_between(
        epochs,
        val_loss_m - val_loss_sem,
        val_loss_m + val_loss_sem,
        alpha=0.3,
        label="Val Loss ± SEM",
    )
    plt.title("SPR_BENCH Aggregated Loss Curves\nLeft: Mean ± SEM across runs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# 2) Aggregated weighted accuracy curves
try:
    plt.figure()
    for m, s, name, color in [
        (val_CWA_m, val_CWA_sem, "CWA", "tab:blue"),
        (val_SWA_m, val_SWA_sem, "SWA", "tab:orange"),
        (val_CWA2_m, val_CWA2_sem, "CWA2", "tab:green"),
    ]:
        if len(m) == 0:  # skip missing metric
            continue
        plt.plot(epochs, m, label=f"{name} (mean)", color=color)
        plt.fill_between(
            epochs, m - s, m + s, alpha=0.3, color=color, label=f"{name} ± SEM"
        )
    plt.title("SPR_BENCH Aggregated Validation Weighted Accuracies\nRight: Mean ± SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_weighted_acc_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# 3) Final epoch bar chart with error bars
try:
    finals = {
        "CWA": collect("val_CWA"),
        "SWA": collect("val_SWA"),
        "CWA2": collect("val_CWA2"),
    }
    labels, means, sems = [], [], []
    for k, arr_list in finals.items():
        if not arr_list:
            continue
        finals_per_run = [a[-1] for a in arr_list]
        labels.append(k)
        means.append(np.mean(finals_per_run))
        sems.append(np.std(finals_per_run, ddof=1) / np.sqrt(len(finals_per_run)))
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=5)
    plt.xticks(x, labels)
    plt.title("SPR_BENCH Final Epoch Weighted Accuracies\nMean ± SEM across runs")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_final_weighted_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final bar plot: {e}")
    plt.close()

# ---------- print summary ----------
for lbl, m, s in zip(labels, means, sems):
    print(f"{lbl}: {m:.4f} ± {s:.4f}")

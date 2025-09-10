import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- loader
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_f9a648788cfa4b15b92ad96618423f40_proc_3198476/experiment_data.npy",
    "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_067b512095b34d878d2fb30b7f5ebbdf_proc_3198477/experiment_data.npy",
    "None/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"path '{full_path}' does not exist")
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    raise SystemExit("No experiment data could be loaded.")

# ---------------------------------------------------------------- dataset key
ds_name = "SPR_BENCH"


# ------------------------------- helper to collect metric across experiments
def collect_curves(key_chain, cast=float):
    curves = []
    for exp in all_experiment_data:
        val = exp.get(ds_name, {})
        for k in key_chain:
            val = val.get(k, {})
        if isinstance(val, (list, np.ndarray)) and len(val):
            curves.append(np.array(val, dtype=float))
    return curves


def collect_scalars(scalar_fn):
    vals = []
    for exp in all_experiment_data:
        ds = exp.get(ds_name, {})
        try:
            vals.append(float(scalar_fn(ds)))
        except Exception:
            continue
    return np.array(vals, dtype=float)


# ------------------------------------------------------------------- curves
train_loss_curves = collect_curves(["losses", "train"])
val_loss_curves = collect_curves(["metrics", "val_loss"])
train_acc_curves = collect_curves(["metrics", "train_acc"])
val_acc_curves = collect_curves(["metrics", "val_acc"])


# ------------------------------------------------------------ scalar metrics
def _test_acc(ds):
    p = np.array(ds.get("predictions", []))
    g = np.array(ds.get("ground_truth", []))
    return (p == g).mean() if p.size and g.size else np.nan


def _fidelity(ds):
    p = np.array(ds.get("predictions", []))
    r = np.array(ds.get("rule_preds", []))
    return (p == r).mean() if p.size and r.size else np.nan


def _fagm(ds):
    ta = _test_acc(ds)
    fi = _fidelity(ds)
    return np.sqrt(ta * fi) if np.isfinite(ta * fi) else np.nan


test_acc_vals = collect_scalars(_test_acc)
fidelity_vals = collect_scalars(_fidelity)
fagm_vals = collect_scalars(_fagm)


# ------------------------------- generic routine to compute mean & sem curves
def mean_sem(curves):
    if not curves:
        return None, None, None
    max_len = max(map(len, curves))
    mat = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    epochs = np.arange(1, len(mean) + 1)
    return epochs, mean, sem


# ------------------------------------------------------------------ PLOT 1
try:
    ep_tr, m_tr, se_tr = mean_sem(train_loss_curves)
    ep_val, m_val, se_val = mean_sem(val_loss_curves)
    if m_tr is not None or m_val is not None:
        plt.figure()
        if m_tr is not None:
            plt.plot(ep_tr, m_tr, label="Train Loss (mean)")
            plt.fill_between(
                ep_tr, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="Train SEM"
            )
        if m_val is not None:
            plt.plot(ep_val, m_val, label="Val Loss (mean)")
            plt.fill_between(
                ep_val, m_val - se_val, m_val + se_val, alpha=0.3, label="Val SEM"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Aggregated Loss Curves\n(Mean ± SEM over runs)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png"))
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------ PLOT 2
try:
    ep_tr, m_tr, se_tr = mean_sem(train_acc_curves)
    ep_val, m_val, se_val = mean_sem(val_acc_curves)
    if m_tr is not None or m_val is not None:
        plt.figure()
        if m_tr is not None:
            plt.plot(ep_tr, m_tr, label="Train Acc (mean)")
            plt.fill_between(
                ep_tr, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="Train SEM"
            )
        if m_val is not None:
            plt.plot(ep_val, m_val, label="Val Acc (mean)")
            plt.fill_between(
                ep_val, m_val - se_val, m_val + se_val, alpha=0.3, label="Val SEM"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title(f"{ds_name} Aggregated Accuracy Curves\n(Mean ± SEM over runs)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_accuracy_curves.png"))
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------ PLOT 3
try:
    metrics = [test_acc_vals, fidelity_vals, fagm_vals]
    labels = ["Test Acc", "Fidelity", "FAGM"]
    means = [np.nanmean(m) if m.size else np.nan for m in metrics]
    sems = [
        np.nanstd(m, ddof=1) / np.sqrt(len(m)) if m.size > 1 else 0 for m in metrics
    ]
    if not all(np.isnan(means)):
        x = np.arange(len(labels))
        plt.figure()
        plt.bar(
            x,
            means,
            yerr=sems,
            capsize=5,
            color=["tab:blue", "tab:orange", "tab:green"],
        )
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        for i, (mn, se) in enumerate(zip(means, sems)):
            if np.isfinite(mn):
                plt.text(i, mn + 0.02, f"{mn:.2f}±{se:.2f}", ha="center")
        plt.title(
            f"{ds_name} Summary Metrics\nMean ± SEM over {len(test_acc_vals)} runs"
        )
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_summary_metrics.png"))
except Exception as e:
    print(f"Error creating aggregated summary plot: {e}")
finally:
    plt.close()

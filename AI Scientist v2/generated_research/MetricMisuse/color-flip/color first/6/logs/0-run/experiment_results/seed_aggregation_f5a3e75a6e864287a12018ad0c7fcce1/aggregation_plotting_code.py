import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Load every experiment result listed by the system
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_96787b605d02401ca0cf7d008a31d596_proc_1664069/experiment_data.npy",
        "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_93de3f196a174e8bb3565e42edfa1039_proc_1664071/experiment_data.npy",
        "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e71d7dc926e54620b6e2f3930751ca35_proc_1664070/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------
def get_beta_values(all_data):
    betas = set()
    for ed in all_data:
        for b in ed.get("adam_beta2", {}):
            betas.add(b)
    return sorted(betas, key=lambda x: float(x))


beta2_values = get_beta_values(all_experiment_data)


# ------------------------------------------------------------
# Helper to aggregate curves across runs
def aggregate_curves(all_data, beta, split="train"):
    # returns epochs, mean, stderr
    run_losses = []
    for ed in all_data:
        try:
            curve = ed["adam_beta2"][beta]["SPR_BENCH"]["losses"][split]
            epochs, losses = zip(*curve)
            run_losses.append(np.array(losses, dtype=float))
        except Exception:
            continue
    if not run_losses:
        return None, None, None
    run_losses = np.array(run_losses)  # shape (runs, epochs)
    mean = run_losses.mean(axis=0)
    stderr = run_losses.std(axis=0, ddof=1) / np.sqrt(run_losses.shape[0])
    return np.array(epochs, dtype=int), mean, stderr


# ------------------------------------------------------------
# 1-4.  Mean loss curves with standard-error bands (≤4 betas)
for beta in beta2_values[:4]:
    try:
        tr_epochs, tr_mean, tr_se = aggregate_curves(all_experiment_data, beta, "train")
        val_epochs, val_mean, val_se = aggregate_curves(
            all_experiment_data, beta, "val"
        )
        if tr_epochs is None or val_epochs is None:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(tr_epochs, tr_mean, label="Train (mean)", color="tab:blue")
        plt.fill_between(
            tr_epochs,
            tr_mean - tr_se,
            tr_mean + tr_se,
            color="tab:blue",
            alpha=0.3,
            label="Train (SE)",
        )
        plt.plot(val_epochs, val_mean, label="Val (mean)", color="tab:orange")
        plt.fill_between(
            val_epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="tab:orange",
            alpha=0.3,
            label="Val (SE)",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss (Mean ± SE)  β₂={beta}\nLeft: Train, Right: Val")
        plt.legend()
        fname = f"mean_loss_curve_SPR_BENCH_beta2_{beta}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating mean loss plot for β2={beta}: {e}")
        plt.close()

# ------------------------------------------------------------
# 5. Overlay of mean val-loss for every β2
try:
    plt.figure(figsize=(6, 4))
    for beta in beta2_values:
        ep, mean, se = aggregate_curves(all_experiment_data, beta, "val")
        if ep is None:
            continue
        plt.plot(ep, mean, label=f"β₂={beta}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross-Entropy Loss")
    plt.title("SPR_BENCH Validation Loss (Mean over runs)\nAll β₂ settings")
    plt.legend()
    fname = "overlay_val_loss_SPR_BENCH_all_beta2.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating overlay plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 6. Summary bar chart of final validation metrics with error bars
try:
    metrics = ["CWA", "SWA", "EWA"]
    mean_vals = {m: [] for m in metrics}
    se_vals = {m: [] for m in metrics}

    for beta in beta2_values:
        for m in metrics:
            metric_runs = []
            for ed in all_experiment_data:
                try:
                    metr_list = ed["adam_beta2"][beta]["SPR_BENCH"]["metrics"]["val"]
                    _, last_dict = metr_list[-1]
                    metric_runs.append(float(last_dict[m]))
                except Exception:
                    continue
            if metric_runs:
                metric_runs = np.array(metric_runs)
                mean_vals[m].append(metric_runs.mean())
                se_vals[m].append(metric_runs.std(ddof=1) / np.sqrt(metric_runs.size))
            else:
                mean_vals[m].append(np.nan)
                se_vals[m].append(np.nan)

    x = np.arange(len(beta2_values))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, mean_vals[m], width, yerr=se_vals[m], capsize=3, label=m)

    plt.xticks(x + width, beta2_values)
    plt.ylabel("Score")
    plt.title("SPR_BENCH Final Validation Metrics  (Mean ± SE)\nBars: CWA, SWA, EWA")
    plt.legend()
    fname = "val_metric_summary_SPR_BENCH_mean_se.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating summary metric plot: {e}")
    plt.close()

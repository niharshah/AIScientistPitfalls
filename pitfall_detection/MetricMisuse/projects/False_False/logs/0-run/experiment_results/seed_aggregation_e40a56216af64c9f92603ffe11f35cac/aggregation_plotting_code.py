import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Paths supplied by the prompt
experiment_data_path_list = [
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_b70b1a3c505c490cab3db45a5e6d0e0f_proc_2964458/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_bae519d796f84228818cfeee12c15f90_proc_2964457/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_fd93af9c72ea46e6b9c1ae364c3c0a89_proc_2964456/experiment_data.npy",
]

all_exp = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_exp.append(ed)
    if len(all_exp) == 0:
        raise RuntimeError("No experiment files loaded")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_exp = []

# ------------------------------------------------------------------
if all_exp:
    # Assume same dropout and shape across runs
    rates = np.array(all_exp[0]["dropout_rate"]["SPR_BENCH"]["rates"])
    n_rates = len(rates)
    n_runs = len(all_exp)

    # Pre-allocate lists for stacking
    train_f1_runs, val_f1_runs = [], []
    train_ls_runs, val_ls_runs = [], []
    swa_runs, cwa_runs = [], []

    for ed in all_exp:
        exp = ed["dropout_rate"]["SPR_BENCH"]
        train_f1_runs.append(np.array(exp["metrics"]["train_macroF1"]))
        val_f1_runs.append(np.array(exp["metrics"]["val_macroF1"]))
        train_ls_runs.append(np.array(exp["losses"]["train"]))
        val_ls_runs.append(np.array(exp["losses"]["val"]))
        swa_runs.append(np.array(exp["swa"]))
        cwa_runs.append(np.array(exp["cwa"]))

    train_f1 = np.stack(train_f1_runs, axis=0)  # shape (R, D, E)
    val_f1 = np.stack(val_f1_runs, axis=0)
    train_ls = np.stack(train_ls_runs, axis=0)
    val_ls = np.stack(val_ls_runs, axis=0)
    swa = np.stack(swa_runs, axis=0)  # shape (R, D)
    cwa = np.stack(cwa_runs, axis=0)

    epochs = np.arange(1, train_f1.shape[2] + 1)
    sem = lambda x, axis=0: np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])

    # 1) Aggregated Macro-F1 curves
    try:
        plt.figure()
        for d in range(n_rates):
            mean_val = val_f1[:, d, :].mean(0)
            err_val = sem(val_f1[:, d, :], axis=0)
            plt.plot(epochs, mean_val, label=f"val d={rates[d]:.2f}")
            plt.fill_between(epochs, mean_val - err_val, mean_val + err_val, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Validation Macro-F1 (Mean ± SEM)\nDropout Sweep")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, "SPR_BENCH_agg_val_macroF1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Macro-F1 plot: {e}")
        plt.close()

    # 2) Aggregated Loss curves
    try:
        plt.figure()
        for d in range(n_rates):
            mean_val = val_ls[:, d, :].mean(0)
            err_val = sem(val_ls[:, d, :], axis=0)
            plt.plot(epochs, mean_val, label=f"val d={rates[d]:.2f}")
            plt.fill_between(epochs, mean_val - err_val, mean_val + err_val, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Validation Loss (Mean ± SEM)\nDropout Sweep")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, "SPR_BENCH_agg_val_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Loss plot: {e}")
        plt.close()

    # 3) Final Val Macro-F1 vs Dropout (bar + error)
    try:
        final_val = val_f1[:, :, -1]  # shape (R, D)
        mean_final = final_val.mean(0)
        err_final = sem(final_val, axis=0)
        plt.figure()
        plt.bar(rates, mean_final, width=0.05, yerr=err_final, capsize=5)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Val Macro-F1")
        plt.title("SPR_BENCH Final Validation Macro-F1\nMean ± SEM across runs")
        fname = os.path.join(working_dir, "SPR_BENCH_agg_final_valF1_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Val-F1 bar plot: {e}")
        plt.close()

    # 4) SWA vs Dropout
    try:
        mean_swa = swa.mean(0)
        err_swa = sem(swa, axis=0)
        plt.figure()
        plt.errorbar(rates, mean_swa, yerr=err_swa, marker="o", capsize=5)
        plt.xlabel("Dropout Rate")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH Shape-Weighted Accuracy\nMean ± SEM across runs")
        fname = os.path.join(working_dir, "SPR_BENCH_agg_SWA_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SWA plot: {e}")
        plt.close()

    # 5) CWA vs Dropout
    try:
        mean_cwa = cwa.mean(0)
        err_cwa = sem(cwa, axis=0)
        plt.figure()
        plt.errorbar(
            rates, mean_cwa, yerr=err_cwa, marker="o", capsize=5, color="green"
        )
        plt.xlabel("Dropout Rate")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH Color-Weighted Accuracy\nMean ± SEM across runs")
        fname = os.path.join(working_dir, "SPR_BENCH_agg_CWA_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CWA plot: {e}")
        plt.close()

    # -------- Evaluation summary --------
    best_idx = np.argmax(mean_final)
    print(
        f"(Aggregated) Best dropout rate: {rates[best_idx]:.2f} "
        f"with mean final Val Macro-F1={mean_final[best_idx]:.4f} "
        f"± {err_final[best_idx]:.4f} (SEM)"
    )
else:
    print("No experiment data available for aggregation.")

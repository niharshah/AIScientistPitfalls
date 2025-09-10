import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------- #
# basic set-up
# --------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------- #
# load every run that belongs to this sweep
# --------------------------------------------------------------- #
experiment_data_path_list = [
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_c6014dd2ef4847b69a21ddaeffc094a2_proc_1513230/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_ae472dc3d6dd413e8b698f9d76cfa600_proc_1513231/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6309830c61404f61bf8b53ac86e02551_proc_1513228/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# --------------------------------------------------------------- #
# helpers to collect SPR data
# --------------------------------------------------------------- #
spr_runs = [ed.get("SPR", {}) for ed in all_experiment_data if "SPR" in ed]
if not spr_runs:
    print("No SPR runs found – nothing to plot")
else:
    # assume identical epoch list across runs
    epochs = spr_runs[0].get("epochs", [])
    n_ep = len(epochs)

    # ------------------------------------------------------------------ #
    # 1) aggregate best-lr TRAIN/VAL LOSS curves
    # ------------------------------------------------------------------ #
    try:
        train_mat, val_mat = [], []
        for run in spr_runs:
            lr_vals = run.get("lr_values", [])
            best_lr = run.get("best_lr", None)
            if best_lr not in lr_vals:
                continue
            idx = lr_vals.index(best_lr)
            tr = np.asarray(run["losses"]["train"][idx][:n_ep])
            va = np.asarray(run["losses"]["val"][idx][:n_ep])
            train_mat.append(tr)
            val_mat.append(va)

        if train_mat and val_mat:
            train_mat = np.vstack(train_mat)
            val_mat = np.vstack(val_mat)

            tr_mean, tr_se = train_mat.mean(0), train_mat.std(0) / np.sqrt(
                train_mat.shape[0]
            )
            va_mean, va_se = val_mat.mean(0), val_mat.std(0) / np.sqrt(val_mat.shape[0])

            plt.figure()
            plt.plot(epochs, tr_mean, label="Train – mean")
            plt.fill_between(
                epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3, label="Train ±1 SE"
            )
            plt.plot(epochs, va_mean, label="Val – mean")
            plt.fill_between(
                epochs, va_mean - va_se, va_mean + va_se, alpha=0.3, label="Val ±1 SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR Dataset – Aggregated Train/Val Loss (best lr of each run)")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_aggregated_loss_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 2) aggregate best-lr HM curves
    # ------------------------------------------------------------------ #
    try:
        tr_mat, va_mat = [], []
        for run in spr_runs:
            lr_vals = run.get("lr_values", [])
            best_lr = run.get("best_lr", None)
            if best_lr not in lr_vals:
                continue
            idx = lr_vals.index(best_lr)

            tr_hm = np.asarray([m["HM"] for m in run["metrics"]["train"][idx]])[:n_ep]
            va_hm = np.asarray([m["HM"] for m in run["metrics"]["val"][idx]])[:n_ep]
            tr_mat.append(tr_hm)
            va_mat.append(va_hm)

        if tr_mat and va_mat:
            tr_mat, va_mat = np.vstack(tr_mat), np.vstack(va_mat)
            tr_mean, tr_se = tr_mat.mean(0), tr_mat.std(0) / np.sqrt(tr_mat.shape[0])
            va_mean, va_se = va_mat.mean(0), va_mat.std(0) / np.sqrt(va_mat.shape[0])

            plt.figure()
            plt.plot(epochs, tr_mean, label="Train HM – mean")
            plt.fill_between(
                epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3, label="Train ±1 SE"
            )
            plt.plot(epochs, va_mean, label="Val HM – mean")
            plt.fill_between(
                epochs, va_mean - va_se, va_mean + va_se, alpha=0.3, label="Val ±1 SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("HM")
            plt.title("SPR Dataset – Aggregated Harmonic-Mean (best lr of each run)")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_aggregated_HM_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated HM plot: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 3) aggregated bar chart of FINAL VAL-HM vs LR
    # ------------------------------------------------------------------ #
    try:
        # collect all lr values seen across runs
        lr_set = set()
        for run in spr_runs:
            lr_set.update(run.get("lr_values", []))
        lr_list = sorted(lr_set)

        means, ses = [], []
        for lr in lr_list:
            vals = []
            for run in spr_runs:
                if lr in run.get("lr_values", []):
                    idx = run["lr_values"].index(lr)
                    vals.append(run["metrics"]["val"][idx][-1]["HM"])
            if vals:
                vals = np.asarray(vals)
                means.append(vals.mean())
                ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                ses.append(np.nan)

        x = np.arange(len(lr_list))
        plt.figure()
        plt.bar(x, means, yerr=ses, capsize=4)
        plt.xticks(x, [str(lr) for lr in lr_list], rotation=45)
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Validation HM")
        plt.title("SPR Dataset – Aggregated Final Val HM per Learning Rate")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_final_valHM_vs_lr_aggregated.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated LR sweep HM bar chart: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 4) print summary of best-lr final HM across runs
    # ------------------------------------------------------------------ #
    finals = []
    for run in spr_runs:
        lr_vals = run.get("lr_values", [])
        best_lr = run.get("best_lr", None)
        if best_lr in lr_vals:
            idx = lr_vals.index(best_lr)
            finals.append(run["metrics"]["val"][idx][-1]["HM"])
    if finals:
        finals = np.asarray(finals)
        print(
            f"Best-lr Final Val HM: mean={finals.mean():.4f} ± {finals.std(ddof=1)/np.sqrt(len(finals)):.4f} (SE) from {len(finals)} runs"
        )

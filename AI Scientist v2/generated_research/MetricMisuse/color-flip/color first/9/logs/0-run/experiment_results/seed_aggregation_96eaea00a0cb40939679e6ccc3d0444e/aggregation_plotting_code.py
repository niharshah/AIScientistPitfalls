import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths given by the user (relative to $AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_aea74893041549abb5b9d80c2f4565cf_proc_1727702/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_efc669f8a4a9484ea64f318b02fd8ab8_proc_1727703/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_a1225ce7f56348368386e61f50a9ca6d_proc_1727704/experiment_data.npy",
]

# ---------- load all runs ----------
all_runs = []
for p in experiment_data_path_list:
    try:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), p), allow_pickle=True
        ).item()
        # keep only runs that actually contain the target dataset
        if "SPR_BENCH" in ed:
            all_runs.append(ed["SPR_BENCH"])
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_runs:
    print("No experiment data could be loaded – nothing to plot.")
else:
    # ---------- helper to aggregate metrics ----------
    def aggregate_losses(runs, split_key):
        # returns {lr: {epoch: [values over runs]}}
        out = {}
        for run in runs:
            for lr, ep, loss in run["losses"][split_key]:
                out.setdefault(lr, {}).setdefault(ep, []).append(loss)
        return out

    def aggregate_metrics(runs, split_key):
        # returns {lr: {epoch: [(cwa,swa,hwa) over runs]}}
        out = {}
        for run in runs:
            for lr, ep, cwa, swa, hwa in run["metrics"][split_key]:
                out.setdefault(lr, {}).setdefault(ep, []).append((cwa, swa, hwa))
        return out

    tr_loss = aggregate_losses(all_runs, "train")
    val_loss = aggregate_losses(all_runs, "val")
    val_metrics = aggregate_metrics(all_runs, "val")

    # collect test results (might be missing)
    test_res = {}
    for run in all_runs:
        if "test" in run["metrics"]:
            lr, cwa, swa, hwa = run["metrics"]["test"]
            test_res.setdefault(lr, []).append((cwa, swa, hwa))

    # ---------- common info ----------
    all_epochs = [ep for lr in tr_loss for ep in tr_loss[lr]]
    max_ep = max(all_epochs) if all_epochs else 0
    stride = max(1, int(np.ceil(max_ep / 5)))  # sample at most 5 epochs

    # ---------- 1) aggregated loss curves ----------
    try:
        plt.figure()
        for lr in sorted(tr_loss):
            eps_sorted = sorted(tr_loss[lr])
            sel = eps_sorted[::stride] + (
                [eps_sorted[-1]] if eps_sorted[-1] not in eps_sorted[::stride] else []
            )
            # training mean & sem
            tr_means = [np.mean(tr_loss[lr][e]) for e in sel]
            tr_sems = [
                np.std(tr_loss[lr][e], ddof=1) / np.sqrt(len(tr_loss[lr][e]))
                for e in sel
            ]
            plt.errorbar(
                sel, tr_means, yerr=tr_sems, fmt="-o", label=f"train lr={lr}", capsize=3
            )
            # validation mean & sem
            val_means = [np.mean(val_loss[lr][e]) for e in sel]
            val_sems = [
                np.std(val_loss[lr][e], ddof=1) / np.sqrt(len(val_loss[lr][e]))
                for e in sel
            ]
            plt.errorbar(
                sel,
                val_means,
                yerr=val_sems,
                fmt="--x",
                label=f"val lr={lr}",
                capsize=3,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss (mean ± SEM over runs)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ---------- 2) aggregated HWA curves ----------
    try:
        plt.figure()
        for lr in sorted(val_metrics):
            eps_sorted = sorted(val_metrics[lr])
            sel = eps_sorted[::stride] + (
                [eps_sorted[-1]] if eps_sorted[-1] not in eps_sorted[::stride] else []
            )
            means = []
            sems = []
            for e in sel:
                hwas = [t[2] for t in val_metrics[lr][e]]
                means.append(np.mean(hwas))
                sems.append(np.std(hwas, ddof=1) / np.sqrt(len(hwas)))
            plt.errorbar(sel, means, yerr=sems, fmt="-o", label=f"lr={lr}", capsize=3)
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy (mean ± SEM)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA plot: {e}")
        plt.close()

    # ---------- 3) CWA vs SWA scatter (final epoch mean ± SEM) ----------
    try:
        plt.figure()
        for lr in sorted(val_metrics):
            last_ep = max(val_metrics[lr])
            cwas = [t[0] for t in val_metrics[lr][last_ep]]
            swas = [t[1] for t in val_metrics[lr][last_ep]]
            plt.errorbar(
                np.mean(cwas),
                np.mean(swas),
                xerr=np.std(cwas, ddof=1) / np.sqrt(len(cwas)),
                yerr=np.std(swas, ddof=1) / np.sqrt(len(swas)),
                fmt="o",
                label=f"lr={lr}",
                capsize=3,
            )
            plt.text(np.mean(cwas), np.mean(swas), f"{lr:.0e}")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Final Epoch CWA vs SWA (mean ± SEM)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_cwa_swa_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CWA/SWA scatter: {e}")
        plt.close()

    # ---------- 4) Test HWA bar chart ----------
    try:
        plt.figure()
        if not test_res:  # synthesise from last validation epoch if test missing
            for lr in val_metrics:
                last_ep = max(val_metrics[lr])
                test_res[lr] = val_metrics[lr][last_ep]
        lr_list = sorted(test_res)
        means = []
        sems = []
        for lr in lr_list:
            hwas = [t[2] for t in test_res[lr]]
            means.append(np.mean(hwas))
            sems.append(np.std(hwas, ddof=1) / np.sqrt(len(hwas)))
        x_pos = np.arange(len(lr_list))
        plt.bar(
            x_pos,
            means,
            yerr=sems,
            capsize=5,
            tick_label=[f"{lr:.0e}" for lr in lr_list],
        )
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Test Harmonic Weighted Accuracy by LR (mean ± SEM)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_test_hwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test HWA bar chart: {e}")
        plt.close()

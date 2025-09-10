import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------- #
# 0. House-keeping                                              #
# ------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------- #
# 1. Collect all experiment_data dicts                          #
# ------------------------------------------------------------- #
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_7fa860ff844641b0b832bc3b11375f18_proc_1610519/experiment_data.npy",
    "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_01eb6bc3aacc47bbae11603610e21a22_proc_1610520/experiment_data.npy",
    "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_64dcafe62d9a4e1b93d4ee080cb86ff0_proc_1610517/experiment_data.npy",
]

all_exp = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_exp.append(np.load(full_p, allow_pickle=True).item())
    except Exception as e:
        print(f"Error loading {p}: {e}")


def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# ------------------------------------------------------------- #
# 2. Aggregate per-dataset information                          #
# ------------------------------------------------------------- #
datasets = {}
for run in all_exp:
    for dname, dct in run.items():
        datasets.setdefault(dname, []).append(dct)

# ------------------------------------------------------------- #
# 3. Iterate over datasets and create aggregated plots          #
# ------------------------------------------------------------- #
for dname, run_list in datasets.items():
    num_runs = len(run_list)

    # ========== a) Mean ± SEM loss curves ==========
    try:
        tr_losses, val_losses = [], []
        for dct in run_list:
            tr_epochs = np.array(unpack(dct["losses"]["train"], 0))
            tr_vals = np.array(unpack(dct["losses"]["train"], 1))
            val_epochs = np.array(unpack(dct["losses"]["val"], 0))
            val_vals = np.array(unpack(dct["losses"]["val"], 1))

            # keep only common length
            L = min(len(tr_vals), len(val_vals))
            tr_losses.append(tr_vals[:L])
            val_losses.append(val_vals[:L])
            epochs_common = tr_epochs[:L]  # assume aligned across runs

        tr_losses = np.vstack(tr_losses)
        val_losses = np.vstack(val_losses)

        tr_mean, tr_sem = tr_losses.mean(axis=0), tr_losses.std(
            axis=0, ddof=1
        ) / np.sqrt(num_runs)
        val_mean, val_sem = val_losses.mean(axis=0), val_losses.std(
            axis=0, ddof=1
        ) / np.sqrt(num_runs)

        plt.figure()
        plt.plot(epochs_common, tr_mean, "--", label="Train μ")
        plt.fill_between(
            epochs_common,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            color="steelblue",
            alpha=0.3,
            label="Train ±SEM",
        )
        plt.plot(epochs_common, val_mean, "-", label="Validation μ", color="orange")
        plt.fill_between(
            epochs_common,
            val_mean - val_sem,
            val_mean + val_sem,
            color="orange",
            alpha=0.3,
            label="Val ±SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{dname}: Aggregated Train vs Val Loss\n(N={num_runs} runs)")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, f"{dname}_agg_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ========== b) Mean ± SEM metric curves (4 panels) ==========
    try:
        metric_names = ["CWA", "SWA", "HCSA", "SNWA"]
        plt.figure(figsize=(8, 6))
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs = axs.flatten()

        for m_idx, m_name in enumerate(metric_names):
            curves = []
            ep_common = None
            for dct in run_list:
                metrics_val = dct["metrics"]["val"]
                if not metrics_val:
                    continue
                ep = np.array(unpack(metrics_val, 0))
                values = np.array(
                    unpack(metrics_val, m_idx + 1)
                )  # +1: first entry is epochs
                if ep_common is None:
                    ep_common = ep[:]
                L = min(len(values), len(ep_common))
                curves.append(values[:L])
                ep_common = ep_common[:L]
            if not curves:
                continue
            curves = np.vstack(curves)
            mean, sem = curves.mean(axis=0), curves.std(axis=0, ddof=1) / np.sqrt(
                curves.shape[0]
            )
            ax = axs[m_idx]
            ax.plot(ep_common, mean, "-o", ms=3, label="μ")
            ax.fill_between(ep_common, mean - sem, mean + sem, alpha=0.3, label="±SEM")
            ax.set_xlabel("Epoch")
            ax.set_title(m_name)
            ax.legend(fontsize=6)

        fig.suptitle(
            f"{dname}: Aggregated Validation Metrics (N={num_runs})\nLeft-Top→Right-Bottom: CWA, SWA, HCSA, SNWA"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, f"{dname}_agg_val_metrics.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dname}: {e}")
        plt.close()

    # ========== c) Dev vs Test accuracy bar with error bars ==========
    try:
        dev_accs, test_accs = [], []
        for dct in run_list:
            for split in ["dev", "test"]:
                preds = np.array(dct["predictions"].get(split, []))
                gts = np.array(dct["ground_truth"].get(split, []))
                acc = (preds == gts).mean() if preds.size else np.nan
                dct.setdefault("acc", {})[split] = acc
            dev_accs.append(dct["acc"]["dev"])
            test_accs.append(dct["acc"]["test"])

        dev_accs, test_accs = np.array(dev_accs), np.array(test_accs)
        means = [np.nanmean(dev_accs), np.nanmean(test_accs)]
        sems = [
            np.nanstd(dev_accs, ddof=1) / np.sqrt(len(dev_accs)),
            np.nanstd(test_accs, ddof=1) / np.sqrt(len(test_accs)),
        ]

        plt.figure()
        plt.bar(
            ["Dev", "Test"], means, yerr=sems, color=["steelblue", "orange"], capsize=4
        )
        plt.ylabel("Accuracy")
        plt.title(f"{dname}: Dev vs Test Accuracy (μ ± SEM)\n(N={num_runs})")
        fname = os.path.join(working_dir, f"{dname}_agg_dev_vs_test_accuracy.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated accuracy chart for {dname}: {e}")
        plt.close()

# ------------------------------------------------------------- #
# 4. Print text summary                                         #
# ------------------------------------------------------------- #
for dname, run_list in datasets.items():
    dev_accs = [r["acc"]["dev"] for r in run_list if "acc" in r and "dev" in r["acc"]]
    test_accs = [
        r["acc"]["test"] for r in run_list if "acc" in r and "test" in r["acc"]
    ]
    if dev_accs and test_accs:
        print(
            f"{dname}: Dev μ={np.mean(dev_accs):.3f} (±{np.std(dev_accs, ddof=1)/np.sqrt(len(dev_accs)):.3f}), "
            f"Test μ={np.mean(test_accs):.3f} (±{np.std(test_accs, ddof=1)/np.sqrt(len(test_accs)):.3f})"
        )

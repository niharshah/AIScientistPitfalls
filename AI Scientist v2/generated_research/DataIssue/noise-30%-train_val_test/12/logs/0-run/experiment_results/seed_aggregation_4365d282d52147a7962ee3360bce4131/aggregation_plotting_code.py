import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- SETUP --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Absolute (or relative) paths provided by the system
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_09a6622b5dbe4715bcda6e9c1790017d_proc_3475576/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_3295b437ac884ef3a9d4cc14cc84daff_proc_3475575/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_0bbf9d272beb487f8079e9f61693d39f_proc_3475578/experiment_data.npy",
]

# -------------------- LOAD & MERGE RUNS --------------------
aggregated = {}  # aggregated[abl][dataset]['batch_size'][bs]['runs'] -> list[run_dict]
try:
    for p in experiment_data_path_list:
        try:
            exp = np.load(
                os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
            ).item()
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue
        for abl_name, abl_dict in exp.items():
            a_entry = aggregated.setdefault(abl_name, {})
            for dset, dset_dict in abl_dict.items():
                d_entry = a_entry.setdefault(dset, {})
                for bs, run in dset_dict.get("batch_size", {}).items():
                    bs_entry = d_entry.setdefault("batch_size", {}).setdefault(bs, {})
                    bs_entry.setdefault("runs", []).append(run)
except Exception as e:
    print(f"Unexpected aggregation error: {e}")


# -------------------- HELPER --------------------
def _to_np(lst):
    return np.asarray(lst, dtype=float)


def _compress_epochs(epochs, max_points=200):
    if len(epochs) <= max_points:
        return np.arange(len(epochs))
    idx = np.linspace(0, len(epochs) - 1, max_points, dtype=int)
    return idx


# -------------------- PLOTS PER MODEL-DATASET --------------------
for abl, abl_dict in aggregated.items():
    for dset, dset_dict in abl_dict.items():
        for bs, bs_dict in dset_dict["batch_size"].items():
            runs = bs_dict["runs"]
            if not runs:
                continue
            # stack data
            epochs = runs[0]["epochs"]
            idx_keep = _compress_epochs(epochs)
            epochs_np = _to_np(epochs)[idx_keep]
            train_losses = np.stack(
                [_to_np(r["losses"]["train"])[idx_keep] for r in runs]
            )
            val_losses = np.stack([_to_np(r["losses"]["val"])[idx_keep] for r in runs])
            val_f1s = np.stack([_to_np(r["metrics"]["val_f1"])[idx_keep] for r in runs])

            n_runs = train_losses.shape[0]
            train_mu, train_se = train_losses.mean(0), train_losses.std(
                0, ddof=1
            ) / np.sqrt(n_runs)
            val_mu, val_se = val_losses.mean(0), val_losses.std(0, ddof=1) / np.sqrt(
                n_runs
            )
            f1_mu, f1_se = val_f1s.mean(0), val_f1s.std(0, ddof=1) / np.sqrt(n_runs)

            # --------- LOSS CURVES ---------
            try:
                plt.figure(figsize=(6, 4))
                plt.fill_between(
                    epochs_np, train_mu - train_se, train_mu + train_se, alpha=0.2
                )
                plt.plot(epochs_np, train_mu, label="Train Loss")
                plt.fill_between(epochs_np, val_mu - val_se, val_mu + val_se, alpha=0.2)
                plt.plot(epochs_np, val_mu, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{abl} on {dset} (bs={bs})\nLeft: Mean ± SE Loss Curves")
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{dset}_{abl}_bs{bs}_loss_mean_se.png"
                )
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Error plotting loss curves for {abl} bs={bs}: {e}")
                plt.close()

            # --------- VAL F1 CURVES ---------
            try:
                plt.figure(figsize=(6, 4))
                plt.fill_between(epochs_np, f1_mu - f1_se, f1_mu + f1_se, alpha=0.2)
                plt.plot(epochs_np, f1_mu, label="Val Macro-F1")
                plt.xlabel("Epoch")
                plt.ylabel("Macro-F1")
                plt.title(f"{abl} on {dset} (bs={bs})\nRight: Mean ± SE Val Macro-F1")
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{dset}_{abl}_bs{bs}_valF1_mean_se.png"
                )
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Error plotting F1 curves for {abl} bs={bs}: {e}")
                plt.close()

# -------------------- BAR CHART OF BEST VAL F1 --------------------
try:
    # gather best-per-run then aggregate
    bar_dict = {}  # {(abl,bs): list_of_best_f1}
    for abl, abl_dict in aggregated.items():
        for dset, dset_dict in abl_dict.items():
            for bs, bs_dict in dset_dict["batch_size"].items():
                runs = bs_dict["runs"]
                bests = [max(_to_np(r["metrics"]["val_f1"])) for r in runs]
                bar_dict.setdefault((dset, abl, bs), []).extend(bests)

    # make one figure per dataset to avoid clutter
    for dset in sorted({k[0] for k in bar_dict}):
        entries = {k: v for k, v in bar_dict.items() if k[0] == dset}
        models = sorted({k[1] for k in entries})
        bss = sorted({k[2] for k in entries})
        x = np.arange(len(bss))
        width = 0.8 / len(models) if models else 0.4
        plt.figure(figsize=(6, 4))
        for i, model in enumerate(models):
            means = []
            ses = []
            for bs in bss:
                vals = entries.get((dset, model, bs), [])
                if vals:
                    means.append(np.mean(vals))
                    ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                else:
                    means.append(0.0)
                    ses.append(0.0)
            pos = x + i * width
            plt.bar(pos, means, width=width, yerr=ses, capsize=3, label=model)
        plt.xticks(x + width * (len(models) - 1) / 2, [str(bs) for bs in bss])
        plt.xlabel("Batch Size")
        plt.ylabel("Best Val Macro-F1 (mean ± SE)")
        plt.title(f"{dset}: Best Validation F1 across runs")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, f"{dset}_best_valF1_bar_mean_se.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar chart: {e}")
    plt.close()

# -------------------- PRINT SUMMARY TABLE --------------------
print("\nAggregated Best Val Macro-F1 (mean ± SE)")
for (dset, abl, bs), vals in sorted(bar_dict.items()):
    vals_np = np.asarray(vals, dtype=float)
    mu, se = vals_np.mean(), vals_np.std(ddof=1) / np.sqrt(len(vals_np))
    print(f"{dset:10s} | {abl:12s} | bs={bs:3d} | {mu:.4f} ± {se:.4f}")

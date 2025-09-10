import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load all experiment files -------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_4c18efd9865f47549429129321cc8481_proc_1604392/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_24f40d6683344c36a08ffb2fb2d9711e_proc_1604394/experiment_data.npy",
    "experiments/2025-08-31_02-26-44_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_5e69b6f84aca46d2908181f75d4c3e14_proc_1604395/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_p):
            all_experiment_data.append(np.load(full_p, allow_pickle=True).item())
        else:
            print(f"File not found: {full_p}")
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data found, nothing to plot.")
    exit()

# ------------------------------------------------------------------
# collect per-tag lists of metrics across repeats ------------------
tags = set()
for ed in all_experiment_data:
    tags.update(ed.keys())
tags = sorted(tags)

train_loss_mean, train_loss_se = {}, {}
val_loss_mean, val_loss_se = {}, {}
val_hmwa_mean, val_hmwa_se = {}, {}
test_hmwa_mean, test_hmwa_se = {}, {}
epochs_dict = {}

for tag in tags:
    # gather arrays across runs
    train_curves, val_curves, hmwa_curves, test_vals = [], [], [], []
    for ed in all_experiment_data:
        if tag not in ed:
            continue
        spr = ed[tag]["SPR_BENCH"]
        train_curves.append(np.asarray(spr["losses"]["train"], dtype=float))
        val_curves.append(np.asarray(spr["losses"]["val"], dtype=float))
        hmwa_curves.append(
            np.asarray([m["hmwa"] for m in spr["metrics"]["val"]], dtype=float)
        )
        test_vals.append(float(spr["metrics"]["test"]["hmwa"]))
    if not train_curves:  # tag not present in any run
        continue

    # align lengths to shortest run
    min_len = min(map(len, train_curves))
    train_curves = np.stack([c[:min_len] for c in train_curves])
    val_curves = np.stack([c[:min_len] for c in val_curves])
    hmwa_curves = np.stack([c[:min_len] for c in hmwa_curves])

    train_loss_mean[tag] = train_curves.mean(0)
    train_loss_se[tag] = train_curves.std(0, ddof=1) / np.sqrt(train_curves.shape[0])

    val_loss_mean[tag] = val_curves.mean(0)
    val_loss_se[tag] = val_curves.std(0, ddof=1) / np.sqrt(val_curves.shape[0])

    val_hmwa_mean[tag] = hmwa_curves.mean(0)
    val_hmwa_se[tag] = hmwa_curves.std(0, ddof=1) / np.sqrt(hmwa_curves.shape[0])

    test_vals = np.asarray(test_vals)
    test_hmwa_mean[tag] = test_vals.mean()
    if len(test_vals) > 1:
        test_hmwa_se[tag] = test_vals.std(ddof=1) / np.sqrt(len(test_vals))
    else:
        test_hmwa_se[tag] = 0.0

    epochs_dict[tag] = list(range(1, min_len + 1))

# ------------------------------------------------------------------
# PLOT 1: mean ± SE loss curves ------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for tag in tags:
        if tag not in train_loss_mean:
            continue
        e = epochs_dict[tag]
        axes[0].plot(e, train_loss_mean[tag], label=tag)
        axes[0].fill_between(
            e,
            train_loss_mean[tag] - train_loss_se[tag],
            train_loss_mean[tag] + train_loss_se[tag],
            alpha=0.3,
        )
        axes[1].plot(e, val_loss_mean[tag], label=tag)
        axes[1].fill_between(
            e,
            val_loss_mean[tag] - val_loss_se[tag],
            val_loss_mean[tag] + val_loss_se[tag],
            alpha=0.3,
        )
    axes[0].set_title("Train Loss (mean ± SE)")
    axes[1].set_title("Validation Loss (mean ± SE)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy")
        ax.legend()
    fig.suptitle(
        "SPR_BENCH Loss Curves Aggregated Across Repeats\nLeft: Train   Right: Validation"
    )
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_aggregated.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# PLOT 2: mean ± SE validation HMWA -------------------------------
try:
    plt.figure(figsize=(6, 4))
    for tag in tags:
        if tag not in val_hmwa_mean:
            continue
        e = epochs_dict[tag]
        plt.plot(e, val_hmwa_mean[tag], label=tag)
        plt.fill_between(
            e,
            val_hmwa_mean[tag] - val_hmwa_se[tag],
            val_hmwa_mean[tag] + val_hmwa_se[tag],
            alpha=0.3,
        )
    plt.title("SPR_BENCH Validation HMWA (mean ± SE) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HMWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HMWA_aggregated.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HMWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# PLOT 3: Test HMWA bar with SE -----------------------------------
try:
    plt.figure(figsize=(6, 4))
    names = [t for t in tags if t in test_hmwa_mean]
    means = [test_hmwa_mean[t] for t in names]
    ses = [test_hmwa_se[t] for t in names]
    x = np.arange(len(names))
    plt.bar(x, means, yerr=ses, capsize=5, color="skyblue")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("HMWA")
    plt.title("SPR_BENCH Test HMWA (mean ± SE) by Hidden Dimension")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_HMWA_bar_aggregated.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test HMWA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final aggregated test metrics ------------------------------
print("\nAggregated Test-set performance (mean ± SE):")
for tag in tags:
    if tag not in test_hmwa_mean:
        continue
    # retrieve other metrics if present in first run for completeness
    first_run = next(ed for ed in all_experiment_data if tag in ed)
    met_single = first_run[tag]["SPR_BENCH"]["metrics"]["test"]
    print(
        f"{tag}: "
        f"CWA={met_single['cwa']:.4f} (single-run), "
        f"SWA={met_single['swa']:.4f} (single-run), "
        f"HMWA={test_hmwa_mean[tag]:.4f} ± {test_hmwa_se[tag]:.4f}"
    )

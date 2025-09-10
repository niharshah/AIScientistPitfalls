import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ------------------------------------------------------------
# Basic set-up
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# 1. Load every experiment_data.npy that actually exists
# ------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_1c137cd264324631bc91007deff6ec91_proc_3102750/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_5344a009089349d195086124489d4ce6_proc_3102749/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8df95724fe944b3785cc88f81c674582_proc_3102751/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment files could be loaded – exiting.")
    exit()

# ------------------------------------------------------------
# 2. Aggregate across runs
# ------------------------------------------------------------
# We build nested dictionaries of the form:
# losses_agg[hs]['train'][epoch] = [v1, v2, ...]  (all runs)
losses_agg = {}
hwa_agg = {}
final_hwa_vals = {}

for exp in all_experiment_data:
    hidden_dict = exp.get("hidden_size", {})
    for hs, result in hidden_dict.items():
        rec = result.get("SPR_BENCH", {})
        tr_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        hwa_seq = [(e, h) for e, _, _, h in rec.get("metrics", {}).get("val", [])]

        # Initialise dicts
        for container, key in [
            (losses_agg, hs),
            (hwa_agg, hs),
            (final_hwa_vals, hs),
        ]:
            if key not in container:
                if container is losses_agg:
                    container[key] = {"train": {}, "val": {}}
                elif container is hwa_agg:
                    container[key] = {}
                else:
                    container[key] = []

        # Train loss
        for e, v in tr_loss:
            losses_agg[hs]["train"].setdefault(e, []).append(v)
        # Val loss
        for e, v in val_loss:
            losses_agg[hs]["val"].setdefault(e, []).append(v)
        # HWA
        for e, v in hwa_seq:
            hwa_agg[hs].setdefault(e, []).append(v)
        # Final epoch HWA
        if hwa_seq:
            final_hwa_vals[hs].append(hwa_seq[-1][1])


# ------------------------------------------------------------
# 3. Helper to turn epoch->list into sorted arrays of mean/sem
# ------------------------------------------------------------
def epoch_dict_to_arrays(d):
    epochs = sorted(d.keys())
    mean_arr = np.array([np.mean(d[e]) for e in epochs])
    sem_arr = np.array(
        [np.std(d[e], ddof=1) / sqrt(len(d[e])) if len(d[e]) > 1 else 0 for e in epochs]
    )
    return np.array(epochs), mean_arr, sem_arr


# ------------------------------------------------------------
# 4. Plot aggregated TRAIN & VAL loss curves
# ------------------------------------------------------------
try:
    plt.figure()
    for hs in sorted(losses_agg.keys()):
        # Train
        ep_t, m_t, s_t = epoch_dict_to_arrays(losses_agg[hs]["train"])
        plt.plot(ep_t, m_t, label=f"train hs={hs}")
        plt.fill_between(ep_t, m_t - s_t, m_t + s_t, alpha=0.25)

        # Val
        ep_v, m_v, s_v = epoch_dict_to_arrays(losses_agg[hs]["val"])
        plt.plot(ep_v, m_v, linestyle="--", label=f"val hs={hs}")
        plt.fill_between(ep_v, m_v - s_v, m_v + s_v, alpha=0.25)

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH (aggregated): Training vs Validation Loss\nMeans ± 1 SEM")
    plt.legend()
    fname = os.path.join(
        working_dir, "SPR_BENCH_loss_curves_hidden_sizes_aggregated.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ------------------------------------------------------------
# 5. Plot aggregated HWA curves
# ------------------------------------------------------------
try:
    plt.figure()
    for hs in sorted(hwa_agg.keys()):
        ep, m, s = epoch_dict_to_arrays(hwa_agg[hs])
        plt.plot(ep, m, label=f"hs={hs}")
        plt.fill_between(ep, m - s, m + s, alpha=0.25)

    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH (aggregated): HWA Curves\nMeans ± 1 SEM")
    plt.legend()
    fname = os.path.join(
        working_dir, "SPR_BENCH_hwa_curves_hidden_sizes_aggregated.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA curves: {e}")
    plt.close()

# ------------------------------------------------------------
# 6. Bar chart of final-epoch HWA with error bars
# ------------------------------------------------------------
try:
    plt.figure()
    hs_sorted = sorted(final_hwa_vals.keys())
    means = [np.mean(final_hwa_vals[h]) for h in hs_sorted]
    sems = [
        (
            np.std(final_hwa_vals[h], ddof=1) / sqrt(len(final_hwa_vals[h]))
            if len(final_hwa_vals[h]) > 1
            else 0
        )
        for h in hs_sorted
    ]
    x_pos = np.arange(len(hs_sorted))
    plt.bar(x_pos, means, yerr=sems, capsize=5, color="skyblue")
    plt.xticks(x_pos, [str(h) for h in hs_sorted])
    plt.xlabel("Hidden Size")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final HWA by Hidden Size\nMeans ± 1 SEM over runs")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar_aggregated.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final HWA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------
# 7. Console summary
# ------------------------------------------------------------
print("Aggregated Final-epoch HWA per hidden size (mean ± SEM):")
for hs in hs_sorted:
    m = np.mean(final_hwa_vals[hs])
    s = (
        np.std(final_hwa_vals[hs], ddof=1) / sqrt(len(final_hwa_vals[hs]))
        if len(final_hwa_vals[hs]) > 1
        else 0
    )
    print(f"  hidden={hs:>3}: {m:.4f} ± {s:.4f}")

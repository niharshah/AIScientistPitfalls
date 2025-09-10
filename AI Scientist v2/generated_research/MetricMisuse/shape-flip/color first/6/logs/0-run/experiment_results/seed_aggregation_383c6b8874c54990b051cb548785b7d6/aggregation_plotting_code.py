import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ALL experiment data -----------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_eb770f9faf4e441fb17dfb78889a5153_proc_1490561/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_8c1e7392314448968448e948c16bacb2_proc_1490562/experiment_data.npy",
    "experiments/2025-08-30_20-55-34_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_63cc822c90fe4fb592d5ce5b5bac42b9_proc_1490560/experiment_data.npy",
]
all_experiment_data = []
for rel_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – nothing to plot.")
    exit()

dataset_type = all_experiment_data[0].get("dataset_type", "unknown")

# ---------- aggregate metrics -------------------------------------------------
test_scores = {}  # bs -> list of scores
losses_train = {}  # bs -> list of np.arrays
losses_val = {}

for exp in all_experiment_data:
    for key, subd in exp.get("batch_size", {}).items():
        try:
            bs_int = int(key.split("_")[-1])
            test_scores.setdefault(bs_int, []).append(subd["metrics"]["test_compwa"])
            losses_train.setdefault(bs_int, []).append(
                np.asarray(subd["losses"]["train"])
            )
            losses_val.setdefault(bs_int, []).append(np.asarray(subd["losses"]["val"]))
        except Exception as e:
            print(f"Error aggregating data for key {key}: {e}")

# ---------- bar plot: mean ± SE test CompWA ----------------------------------
try:
    plt.figure()
    bs_sorted = sorted(test_scores.keys())
    means = [np.mean(test_scores[bs]) for bs in bs_sorted]
    ses = [
        np.std(test_scores[bs], ddof=1) / np.sqrt(len(test_scores[bs]))
        for bs in bs_sorted
    ]
    plt.bar(
        [str(b) for b in bs_sorted],
        means,
        yerr=ses,
        capsize=5,
        alpha=0.7,
        label="Mean ± SE",
    )
    plt.title(f"Test CompWA (Mean ± SE) by Batch Size\nDataset: {dataset_type}")
    plt.xlabel("Batch Size")
    plt.ylabel("Test CompWA")
    plt.legend()
    fname = f"test_compwa_agg_{dataset_type}.png"
    plt.savefig(os.path.join(working_dir, fname))
except Exception as e:
    print(f"Error creating aggregated CompWA bar plot: {e}")
finally:
    plt.close()

# ---------- aggregated loss curves (max 5 batch sizes) ------------------------
max_plots = 5
for i, bs in enumerate(bs_sorted[:max_plots]):
    try:
        tr_runs = losses_train.get(bs, [])
        va_runs = losses_val.get(bs, [])
        if not tr_runs or not va_runs:
            continue

        # Align epochs by shortest run length
        min_len = min([len(x) for x in tr_runs])
        tr_stack = np.vstack([r[:min_len] for r in tr_runs])
        va_stack = np.vstack([r[:min_len] for r in va_runs])
        epochs = np.arange(min_len)

        tr_mean, tr_se = tr_stack.mean(axis=0), tr_stack.std(axis=0, ddof=1) / np.sqrt(
            tr_stack.shape[0]
        )
        va_mean, va_se = va_stack.mean(axis=0), va_stack.std(axis=0, ddof=1) / np.sqrt(
            va_stack.shape[0]
        )

        plt.figure()
        plt.plot(epochs, tr_mean, label="Train Mean", color="tab:blue")
        plt.fill_between(
            epochs,
            tr_mean - tr_se,
            tr_mean + tr_se,
            color="tab:blue",
            alpha=0.2,
            label="Train ±SE",
        )
        plt.plot(epochs, va_mean, label="Val Mean", color="tab:orange")
        plt.fill_between(
            epochs,
            va_mean - va_se,
            va_mean + va_se,
            color="tab:orange",
            alpha=0.2,
            label="Val ±SE",
        )
        plt.title(
            f"Aggregated Loss Curve (Mean ± SE)\nDataset: {dataset_type}, BS={bs}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"loss_curve_agg_{dataset_type}_bs{bs}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating aggregated loss plot for bs={bs}: {e}")
    finally:
        plt.close()

# ---------- print aggregated evaluation metrics ------------------------------
print("\n=== Aggregated Test CompWA by Batch Size ===")
for bs in bs_sorted:
    m = np.mean(test_scores[bs])
    se = np.std(test_scores[bs], ddof=1) / np.sqrt(len(test_scores[bs]))
    print(f"  bs={bs:>3}: {m:.4f} ± {se:.4f}")
best_bs = bs_sorted[int(np.argmax([np.mean(test_scores[bs]) for bs in bs_sorted]))]
print(f"\nBest batch size (highest mean CompWA): {best_bs}")

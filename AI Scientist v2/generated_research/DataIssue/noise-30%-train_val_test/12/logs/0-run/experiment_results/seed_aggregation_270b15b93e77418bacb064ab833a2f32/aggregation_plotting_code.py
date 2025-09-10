import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- experiment data paths ----
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_453209b24abb42808713ed775a2b67bb_proc_3462723/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_c8e35197058b4b9e80a5dbd6ebbafbce_proc_3462725/experiment_data.npy",
    "experiments/2025-08-17_23-44-14_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_80826c57a81041a9acc89b3c314fcafc_proc_3462724/experiment_data.npy",
]

# ---- load all runs ----
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        p_full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(p_full, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ---- aggregate by batch size ----
per_bs = {}  # bs -> dict of lists
for run in all_experiment_data:
    batch_dict = run.get("batch_size", {})
    for bs, stats in batch_dict.items():
        bs = int(bs)
        if bs not in per_bs:
            per_bs[bs] = {
                "epochs": [],
                "tr_loss": [],
                "val_loss": [],
                "val_f1": [],
                "best_f1": [],
            }
        per_bs[bs]["epochs"].append(np.array(stats["epochs"]))
        per_bs[bs]["tr_loss"].append(np.array(stats["losses"]["train"]))
        per_bs[bs]["val_loss"].append(np.array(stats["losses"]["val"]))
        per_bs[bs]["val_f1"].append(np.array(stats["metrics"]["val_f1"]))
        per_bs[bs]["best_f1"].append(np.array(stats["metrics"]["val_f1"]).max())


# helper to align epochs across runs (use intersection to keep consistency)
def align_and_stack(list_of_arrays, ref_epochs=None):
    """
    Intersect epochs of all arrays (assuming same ordering) and stack values.
    Returns epochs, stacked_values (runs x epochs)
    """
    if ref_epochs is None:
        ref_epochs = list_of_arrays[0]
    common = set(ref_epochs)
    for arr in list_of_arrays[1:]:
        common &= set(arr)
    common = sorted(list(common))
    aligned = []
    for arr in list_of_arrays:
        # build mask of indices where epoch in common
        idx = [np.where(arr == e)[0][0] for e in common]
        aligned.append(arr[idx])
    return np.array(common), np.stack(aligned, axis=0)


# ---- compute mean & sem ----
agg = {}
for bs, d in per_bs.items():
    epochs, _ = align_and_stack(d["epochs"])
    tr_stack = align_and_stack(d["tr_loss"], ref_epochs=epochs)[1]
    val_loss_stack = align_and_stack(d["val_loss"], ref_epochs=epochs)[1]
    val_f1_stack = align_and_stack(d["val_f1"], ref_epochs=epochs)[1]

    agg[bs] = {
        "epochs": epochs,
        "tr_mean": tr_stack.mean(0),
        "tr_sem": tr_stack.std(0, ddof=1) / np.sqrt(tr_stack.shape[0]),
        "val_loss_mean": val_loss_stack.mean(0),
        "val_loss_sem": val_loss_stack.std(0, ddof=1)
        / np.sqrt(val_loss_stack.shape[0]),
        "val_f1_mean": val_f1_stack.mean(0),
        "val_f1_sem": val_f1_stack.std(0, ddof=1) / np.sqrt(val_f1_stack.shape[0]),
        "best_f1_mean": np.mean(d["best_f1"]),
        "best_f1_sem": np.std(d["best_f1"], ddof=1) / np.sqrt(len(d["best_f1"])),
    }

# ---- 1: train loss mean ± SEM ----
try:
    plt.figure()
    for bs, a in agg.items():
        plt.plot(a["epochs"], a["tr_mean"], label=f"bs={bs} mean")
        plt.fill_between(
            a["epochs"],
            a["tr_mean"] - a["tr_sem"],
            a["tr_mean"] + a["tr_sem"],
            alpha=0.3,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training Loss: Mean ± SEM across runs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_train_loss_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training-loss plot: {e}")
    plt.close()

# ---- 2: validation loss mean ± SEM ----
try:
    plt.figure()
    for bs, a in agg.items():
        plt.plot(a["epochs"], a["val_loss_mean"], label=f"bs={bs} mean")
        plt.fill_between(
            a["epochs"],
            a["val_loss_mean"] - a["val_loss_sem"],
            a["val_loss_mean"] + a["val_loss_sem"],
            alpha=0.3,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Validation Loss: Mean ± SEM across runs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_loss_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation-loss plot: {e}")
    plt.close()

# ---- 3: validation macro-F1 mean ± SEM ----
try:
    plt.figure()
    for bs, a in agg.items():
        plt.plot(a["epochs"], a["val_f1_mean"], label=f"bs={bs} mean")
        plt.fill_between(
            a["epochs"],
            a["val_f1_mean"] - a["val_f1_sem"],
            a["val_f1_mean"] + a["val_f1_sem"],
            alpha=0.3,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Validation Macro-F1: Mean ± SEM across runs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_f1_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation-F1 plot: {e}")
    plt.close()

# ---- 4: best macro-F1 bar: mean ± SEM ----
try:
    plt.figure()
    bs_vals = sorted(agg.keys())
    means = [agg[bs]["best_f1_mean"] for bs in bs_vals]
    sems = [agg[bs]["best_f1_sem"] for bs in bs_vals]
    plt.bar(range(len(bs_vals)), means, yerr=sems, capsize=5, tick_label=bs_vals)
    plt.xlabel("Batch Size")
    plt.ylabel("Best Macro-F1 (mean ± SEM)")
    plt.title("SPR_BENCH Best Validation Macro-F1 by Batch Size")
    plt.savefig(os.path.join(working_dir, "spr_best_f1_bar_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated best-F1 bar plot: {e}")
    plt.close()

# ---- numeric summary ----
print("====== Best Validation Macro-F1 (mean ± SEM) ======")
for bs in sorted(agg.keys()):
    print(
        f"Batch size {bs:>3}: {agg[bs]['best_f1_mean']:.4f} ± {agg[bs]['best_f1_sem']:.4f}"
    )

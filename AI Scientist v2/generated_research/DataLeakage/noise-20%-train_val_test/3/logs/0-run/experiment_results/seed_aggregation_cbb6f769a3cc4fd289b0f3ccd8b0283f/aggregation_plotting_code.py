import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

# ------------------ paths & data ------------------
import warnings

warnings.filterwarnings("ignore")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy files supplied in the prompt
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_92b0f331e7114b58ac03e5ecdf82f766_proc_3162418/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_ccc6960c81604462989021f3bfa7fbb3_proc_3162419/experiment_data.npy",
    "experiments/2025-08-17_00-44-36_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_82724947d1c84e249beadaf7964419f4_proc_3162420/experiment_data.npy",
]

all_runs = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(run_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs = []

# ------------- aggregate per dataset -------------
agg = {}  # {ds: {metric: [runs]}}
for run in all_runs:
    for ds_name, ed in run.items():
        m = ed.get("metrics", {})
        if ds_name not in agg:
            agg[ds_name] = {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "val_f1": [],
                "final_f1": [],
            }
        agg[ds_name]["epochs"].append(np.array(ed.get("epochs", [])))
        agg[ds_name]["train_loss"].append(np.array(m.get("train_loss", [])))
        agg[ds_name]["val_loss"].append(np.array(m.get("val_loss", [])))
        agg[ds_name]["val_f1"].append(np.array(m.get("val_f1", [])))

        # final macro-F1 from predictions / gts if present
        preds = np.array(ed.get("predictions", []))
        gts = np.array(ed.get("ground_truth", []))
        if preds.size and gts.size:
            try:
                agg[ds_name]["final_f1"].append(f1_score(gts, preds, average="macro"))
            except Exception:
                pass


# -------- helper to compute mean & sem --------
def mean_sem(arr_list):
    """
    arr_list: list of 1-D np arrays (possibly different lengths)
    Returns trimmed mean, sem, epochs
    """
    if len(arr_list) == 0:
        return None, None
    # align to shortest length so every index has full statistics
    min_len = min([len(a) for a in arr_list])
    stack = np.stack([a[:min_len] for a in arr_list], axis=0)
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, sem


# ------------ create plots per dataset ------------
final_f1_summary = {}  # mean±sem per dataset
for ds, content in agg.items():
    epochs_list = content["epochs"]
    if not epochs_list:
        continue
    min_len = min([len(e) for e in epochs_list])
    epochs = epochs_list[0][:min_len]  # assume same epochs across runs

    # --- aggregate curves ---
    train_mean, train_sem = mean_sem(content["train_loss"])
    val_mean, val_sem = mean_sem(content["val_loss"])
    f1_mean, f1_sem = mean_sem(content["val_f1"])

    # --- 1. Train/Val loss ---
    try:
        if train_mean is not None and val_mean is not None:
            plt.figure()
            plt.plot(epochs, train_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.3
            )
            plt.plot(epochs, val_mean, label="Val Loss (mean)")
            plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.3)
            plt.title(f"{ds} Mean Loss Curves ± SEM\nLeft: Train, Right: Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_mean_sem_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # --- 2. Validation F1 ---
    try:
        if f1_mean is not None:
            plt.figure()
            plt.plot(epochs, f1_mean, marker="o", label="Val Macro-F1 (mean)")
            plt.fill_between(
                epochs, f1_mean - f1_sem, f1_mean + f1_sem, alpha=0.3, label="± SEM"
            )
            plt.title(f"{ds} Validation Macro-F1 ± SEM Across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_mean_sem_val_f1_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {ds}: {e}")
        plt.close()

    # --- collect final F1 statistics ---
    if content["final_f1"]:
        vals = np.array(content["final_f1"])
        final_f1_summary[ds] = (vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)))
        print(
            f"{ds} Final Test Macro-F1: {vals.mean():.4f} ± {final_f1_summary[ds][1]:.4f}"
        )

# --- 3. comparison bar plot if >1 dataset ---
if len(final_f1_summary) > 1:
    try:
        plt.figure()
        names = list(final_f1_summary.keys())
        means = [final_f1_summary[n][0] for n in names]
        sems = [final_f1_summary[n][1] for n in names]
        plt.bar(names, means, yerr=sems, capsize=5)
        plt.ylim(0, 1)
        plt.title("Final Macro-F1 Comparison Across Datasets (mean ± SEM)")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        fname = os.path.join(working_dir, "datasets_f1_mean_sem_comparison.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated comparison plot: {e}")
        plt.close()

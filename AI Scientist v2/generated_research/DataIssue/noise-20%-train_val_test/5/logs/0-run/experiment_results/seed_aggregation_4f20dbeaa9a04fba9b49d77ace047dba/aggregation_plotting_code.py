import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Collect and load all experiment files
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_6664cf88eb484fdc9e88e1b323928261_proc_3161121/experiment_data.npy",
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_4ce44c2f67ed459b85d3a2afd1e774a4_proc_3161122/experiment_data.npy",
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_61c9f6b8df30427bb9a405d36db69b89_proc_3161120/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------ #
# Aggregate metrics
dataset_name = "SPR_BENCH"
agg = {}  # {nhead: {"train_acc": [...], "val_acc": [...], ... , "test_acc": []}}
for exp in all_experiment_data:
    try:
        results = exp["nhead_tuning"][dataset_name]["results"]
    except Exception:
        continue
    for nhead, data in results.items():
        if nhead not in agg:
            agg[nhead] = {
                "train_acc": [],
                "val_acc": [],
                "train_loss": [],
                "val_loss": [],
                "test_acc": [],
                "preds": [],
                "gts": [],
            }
        # Metrics (variable length – trim later)
        agg[nhead]["train_acc"].append(np.asarray(data["metrics"]["train_acc"]))
        agg[nhead]["val_acc"].append(np.asarray(data["metrics"]["val_acc"]))
        agg[nhead]["train_loss"].append(np.asarray(data["losses"]["train_loss"]))
        agg[nhead]["val_loss"].append(np.asarray(data["losses"]["val_loss"]))
        agg[nhead]["test_acc"].append(data["test_acc"])
        agg[nhead]["preds"].append(np.asarray(data.get("predictions", [])))
        agg[nhead]["gts"].append(np.asarray(data.get("ground_truth", [])))


# ------------------------------------------------------------------ #
# Helper to compute mean & stderr given a list of 1-D arrays
def mean_stderr(arr_list):
    if not arr_list:
        return None, None
    # equalize length by truncating to shortest run
    min_len = min(a.shape[0] for a in arr_list)
    arr_stack = np.stack([a[:min_len] for a in arr_list], axis=0)
    mean = arr_stack.mean(axis=0)
    se = arr_stack.std(axis=0, ddof=1) / np.sqrt(arr_stack.shape[0])
    return mean, se


# ------------------------------------------------------------------ #
# 1. Accuracy curves (mean ± SE)
try:
    plt.figure()
    for nhead, d in agg.items():
        m_train, se_train = mean_stderr(d["train_acc"])
        m_val, se_val = mean_stderr(d["val_acc"])
        if m_train is None or m_val is None:
            continue
        epochs = np.arange(1, len(m_train) + 1)
        plt.plot(epochs, m_train, label=f"Train μ nhead={nhead}")
        plt.fill_between(epochs, m_train - se_train, m_train + se_train, alpha=0.2)
        plt.plot(epochs, m_val, linestyle="--", label=f"Val μ nhead={nhead}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} Accuracy (mean ± SE across runs)")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_agg_accuracy_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2. Loss curves (mean ± SE)
try:
    plt.figure()
    for nhead, d in agg.items():
        m_train, se_train = mean_stderr(d["train_loss"])
        m_val, se_val = mean_stderr(d["val_loss"])
        if m_train is None or m_val is None:
            continue
        epochs = np.arange(1, len(m_train) + 1)
        plt.plot(epochs, m_train, label=f"Train μ nhead={nhead}")
        plt.fill_between(epochs, m_train - se_train, m_train + se_train, alpha=0.2)
        plt.plot(epochs, m_val, linestyle="--", label=f"Val μ nhead={nhead}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} Loss (mean ± SE across runs)")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_agg_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3. Test accuracy bar chart (mean ± SE)
try:
    nheads = sorted(agg.keys())
    means = []
    ses = []
    for n in nheads:
        vals = np.asarray(agg[n]["test_acc"])
        means.append(vals.mean())
        ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    x = np.arange(len(nheads))
    plt.figure()
    plt.bar(x, means, yerr=ses, color="skyblue", capsize=5)
    plt.xticks(x, nheads)
    plt.xlabel("n-head")
    plt.ylabel("Test Accuracy")
    plt.title(f"{dataset_name} Test Accuracy (mean ± SE)")
    fname = os.path.join(working_dir, f"{dataset_name}_agg_test_accuracy.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4. Confusion matrix for best mean-accuracy nhead
try:
    if nheads:
        best_idx = int(np.argmax(means))
        best_nhead = nheads[best_idx]
        # Sum confusion matrices across runs
        all_preds = agg[best_nhead]["preds"]
        all_gts = agg[best_nhead]["gts"]
        if all_preds and all_gts and len(all_preds) == len(all_gts):
            num_classes = len(np.unique(np.concatenate(all_gts)))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p_arr, g_arr in zip(all_preds, all_gts):
                for p, g in zip(p_arr, g_arr):
                    cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dataset_name} Confusion Matrix (best μ nhead={best_nhead})")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fname = os.path.join(working_dir, f"{dataset_name}_agg_confusion_best.png")
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
except Exception as e:
    print(f"Error creating aggregated confusion matrix: {e}")
    plt.close()

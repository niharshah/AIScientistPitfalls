import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_2a72eb39cf984b288880ffac60f1d335_proc_2983631/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_21043688b5e94657b15cc4254182eb65_proc_2983633/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_59c2d3fd98414f9ea7e92b256549bc3a_proc_2983630/experiment_data.npy",
]


# ---------- helpers ----------
def save_and_close(fig, fname):
    fig.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def stack_and_trim(list_of_lists):
    """Stack runs by trimming to the shortest length."""
    if not list_of_lists:
        return np.empty((0, 0))
    min_len = min(len(l) for l in list_of_lists)
    if min_len == 0:
        return np.empty((0, 0))
    arr = np.stack([np.array(l[:min_len], dtype=float) for l in list_of_lists])
    return arr  # shape (n_runs, min_len)


# ---------- load all runs ----------
all_runs = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        spr = exp_data.get("SPR_BENCH", {})
        if spr:
            all_runs.append(spr)
    except Exception as e:
        print(f"Error loading {p}: {e}")

n_runs = len(all_runs)
if n_runs == 0:
    print("No SPR_BENCH data found in any run.")
else:
    print(f"Loaded {n_runs} SPR_BENCH runs.")

# ========== 1. aggregated loss curves ==========
try:
    train_losses = [run.get("losses", {}).get("train", []) for run in all_runs]
    val_losses = [run.get("losses", {}).get("val", []) for run in all_runs]
    tr_mat = stack_and_trim(train_losses)
    va_mat = stack_and_trim(val_losses)

    if tr_mat.size and va_mat.size:
        epochs = np.arange(1, tr_mat.shape[1] + 1)
        tr_mean, tr_se = tr_mat.mean(0), tr_mat.std(0, ddof=1) / np.sqrt(n_runs)
        va_mean, va_se = va_mat.mean(0), va_mat.std(0, ddof=1) / np.sqrt(n_runs)

        fig = plt.figure()
        plt.plot(epochs, tr_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3, label="Train SE"
        )
        plt.plot(epochs, va_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs, va_mean - va_se, va_mean + va_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH Aggregated Loss Curves\n(mean ± standard error, N={})".format(
                n_runs
            )
        )
        plt.legend()
        save_and_close(fig, "SPR_BENCH_aggregated_loss_curves.png")

        # print final numbers
        print("Final epoch Train Loss: {:.4f} ± {:.4f}".format(tr_mean[-1], tr_se[-1]))
        print("Final epoch Val   Loss: {:.4f} ± {:.4f}".format(va_mean[-1], va_se[-1]))
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ========== 2. aggregated weighted accuracy curves ==========
try:
    swa_runs = [
        [m["swa"] for m in run.get("metrics", {}).get("val", [])] for run in all_runs
    ]
    cwa_runs = [
        [m["cwa"] for m in run.get("metrics", {}).get("val", [])] for run in all_runs
    ]
    hwa_runs = [
        [m["hwa"] for m in run.get("metrics", {}).get("val", [])] for run in all_runs
    ]

    swa_mat = stack_and_trim(swa_runs)
    cwa_mat = stack_and_trim(cwa_runs)
    hwa_mat = stack_and_trim(hwa_runs)

    if swa_mat.size and cwa_mat.size and hwa_mat.size:
        epochs = np.arange(1, swa_mat.shape[1] + 1)

        fig = plt.figure()
        for mat, name, color in zip(
            [swa_mat, cwa_mat, hwa_mat],
            ["SWA", "CWA", "HWA"],
            ["tab:blue", "tab:orange", "tab:green"],
        ):
            mean = mat.mean(0)
            se = mat.std(0, ddof=1) / np.sqrt(n_runs)
            plt.plot(epochs, mean, label=f"{name} (mean)", color=color)
            plt.fill_between(
                epochs, mean - se, mean + se, alpha=0.3, color=color, label=f"{name} SE"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH Aggregated Weighted Accuracies\n(mean ± standard error, N={})".format(
                n_runs
            )
        )
        plt.legend()
        save_and_close(fig, "SPR_BENCH_aggregated_weighted_accuracy_curves.png")

        # print final numbers
        for mat, name in zip([swa_mat, cwa_mat, hwa_mat], ["SWA", "CWA", "HWA"]):
            mean, se = mat.mean(0)[-1], (mat.std(0, ddof=1) / np.sqrt(n_runs))[-1]
            print(f"Final epoch {name}: {mean:.4f} ± {se:.4f}")
except Exception as e:
    print(f"Error creating aggregated weighted accuracy curves: {e}")
    plt.close()

# ========== 3. aggregated confusion matrix ==========
try:
    preds_all, trues_all = [], []
    for run in all_runs:
        preds_all.append(np.array(run.get("predictions", []), dtype=int))
        trues_all.append(np.array(run.get("ground_truth", []), dtype=int))

    if preds_all and all(p.size for p in preds_all):
        preds = np.concatenate(preds_all)
        trues = np.concatenate(trues_all)
        num_labels = int(max(preds.max(), trues.max())) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1

        fig = plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH Aggregated Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
        )
        save_and_close(fig, "SPR_BENCH_aggregated_confusion_matrix.png")
except Exception as e:
    print(f"Error creating aggregated confusion matrix: {e}")
    plt.close()

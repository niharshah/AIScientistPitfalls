import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list of experiment data paths ----------
experiment_data_path_list = [
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_330aaa5945ac4d9cb26eb8bdb71762e1_proc_2751360/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_1574596ada924d74bfa78ff1df2da356_proc_2751359/experiment_data.npy",
    "experiments/2025-08-14_21-45-52_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_5cfd2658f9344333b9eb44188df243e8_proc_2751358/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---------- helper function ----------
def collect_metric(run, dataset, key_chain):
    """Safely fetch a metric list from nested dicts given a key chain."""
    d = run.get(dataset, {})
    for k in key_chain:
        d = d.get(k, {})
    # final d should be a list-like
    return np.asarray(d if isinstance(d, (list, tuple, np.ndarray)) else [])


# ---------- aggregate and plot ----------
datasets = set()
for run in all_experiment_data:
    datasets.update(run.keys())

for dataset in datasets:
    # prepare containers
    train_acc_runs, val_acc_runs, train_loss_runs, val_ura_runs, test_acc_runs = (
        [],
        [],
        [],
        [],
        [],
    )

    # -------- gather per run --------
    for run in all_experiment_data:
        # metrics might not exist in this run
        tr_acc = collect_metric(run, dataset, ["metrics", "train_acc"])
        va_acc = collect_metric(run, dataset, ["metrics", "val_acc"])
        tr_loss = collect_metric(run, dataset, ["losses", "train"])
        va_ura = collect_metric(run, dataset, ["metrics", "val_ura"])
        preds = collect_metric(run, dataset, ["predictions"])
        gts = collect_metric(run, dataset, ["ground_truth"])
        # keep only if non-empty
        if tr_acc.size:
            train_acc_runs.append(tr_acc)
        if va_acc.size:
            val_acc_runs.append(va_acc)
        if tr_loss.size:
            train_loss_runs.append(tr_loss)
        if va_ura.size:
            val_ura_runs.append(va_ura)
        if preds.size and gts.size:
            test_acc_runs.append((preds == gts).mean())

    # helper to stack, truncate to min len, compute mean & se
    def mean_se(arr_list):
        if not arr_list:
            return None, None
        min_len = min(len(a) for a in arr_list)
        arr = np.stack([a[:min_len] for a in arr_list], axis=0)
        mean = arr.mean(0)
        se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        epochs = np.arange(1, min_len + 1)
        return epochs, mean, se

    # ---------- plot aggregated accuracy ----------
    try:
        res1 = mean_se(train_acc_runs)
        res2 = mean_se(val_acc_runs)
        if res1 and res2:
            epochs, tr_mean, tr_se = res1
            _, va_mean, va_se = res2
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train Acc (mean)")
            plt.fill_between(epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3)
            plt.plot(epochs, va_mean, label="Val Acc (mean)")
            plt.fill_between(epochs, va_mean - va_se, va_mean + va_se, alpha=0.3)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dataset} – Train vs Val Accuracy (Mean ± SE)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_accuracy_agg.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dataset}: {e}")
        plt.close()

    # ---------- plot aggregated loss ----------
    try:
        res = mean_se(train_loss_runs)
        if res:
            epochs, loss_mean, loss_se = res
            plt.figure()
            plt.plot(epochs, loss_mean, label="Train Loss (mean)")
            plt.fill_between(
                epochs, loss_mean - loss_se, loss_mean + loss_se, alpha=0.3
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} – Training Loss (Mean ± SE)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_loss_agg.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset}: {e}")
        plt.close()

    # ---------- plot aggregated URA ----------
    try:
        res = mean_se(val_ura_runs)
        if res:
            epochs, ura_mean, ura_se = res
            plt.figure()
            plt.plot(epochs, ura_mean, label="Val URA (mean)", color="green")
            plt.fill_between(
                epochs, ura_mean - ura_se, ura_mean + ura_se, alpha=0.3, color="green"
            )
            plt.xlabel("Epoch")
            plt.ylabel("URA")
            plt.title(f"{dataset} – Validation URA (Mean ± SE)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_ura_agg.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated URA plot for {dataset}: {e}")
        plt.close()

    # ---------- confusion matrices for first <=5 runs ----------
    shown = 0
    for idx, run in enumerate(all_experiment_data):
        if shown >= 5:
            break
        preds = collect_metric(run, dataset, ["predictions"])
        gts = collect_metric(run, dataset, ["ground_truth"])
        if preds.size and gts.size:
            try:
                cm = np.zeros((2, 2), dtype=int)
                for p, t in zip(preds, gts):
                    cm[int(t), int(p)] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
                plt.xticks([0, 1], ["Pred 0", "Pred 1"])
                plt.yticks([0, 1], ["True 0", "True 1"])
                plt.title(f"{dataset} – Confusion Matrix (Run {idx})")
                plt.savefig(
                    os.path.join(working_dir, f"{dataset}_confusion_run{idx}.png")
                )
                plt.close()
                shown += 1
            except Exception as e:
                print(f"Error creating confusion matrix for {dataset} run {idx}: {e}")
                plt.close()

    # ---------- print aggregated test accuracy ----------
    if test_acc_runs:
        test_acc_mean = np.mean(test_acc_runs)
        test_acc_se = np.std(test_acc_runs, ddof=1) / np.sqrt(len(test_acc_runs))
        print(
            f"{dataset}: Test Accuracy mean±SE = {test_acc_mean:.3f} ± {test_acc_se:.3f} over {len(test_acc_runs)} runs"
        )

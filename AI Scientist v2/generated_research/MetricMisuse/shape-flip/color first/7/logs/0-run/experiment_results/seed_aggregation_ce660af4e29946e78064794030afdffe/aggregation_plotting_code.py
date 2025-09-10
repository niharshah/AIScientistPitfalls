import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data from all runs ----------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_027e97336a844547b26608389b9422e3_proc_1490728/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_39abd2fe51ed486ebadc9d0f7362ba1a_proc_1490729/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_8c10a2e54e714b828a6017f7e8722e5b_proc_1490727/experiment_data.npy",
]

dataset_runs = {}  # {dataset_name : [ds1, ds2, ...]}
valid_paths = []
for rel_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        experiment_data = np.load(full_path, allow_pickle=True).item()
        dataset_name = list(experiment_data["num_epochs"].keys())[0]
        ds = experiment_data["num_epochs"][dataset_name]
        dataset_runs.setdefault(dataset_name, []).append(ds)
        valid_paths.append(full_path)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")


# ---------- helper to stack a metric ----------
def stack_metric(run_list, getter):
    """getter must take (ds, idx) and return scalar for epoch idx"""
    min_len = min(len(ds["losses"]["train"]) for ds in run_list)
    stacked = np.zeros((len(run_list), min_len))
    for r, ds in enumerate(run_list):
        for i in range(min_len):
            stacked[r, i] = getter(ds, i)
    epochs = np.arange(1, min_len + 1)
    mean = stacked.mean(axis=0)
    se = (
        stacked.std(axis=0, ddof=1) / np.sqrt(len(run_list))
        if len(run_list) > 1
        else np.zeros_like(mean)
    )
    return epochs, mean, se


# ---------- plotting ----------
for dataset_name, runs in dataset_runs.items():
    # ----- loss -----
    try:
        epochs, train_mean, train_se = stack_metric(
            runs, lambda d, i: d["losses"]["train"][i]
        )
        _, val_mean, val_se = stack_metric(runs, lambda d, i: d["losses"]["val"][i])
        plt.figure()
        plt.plot(epochs, train_mean, label="Train Mean")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, val_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ± SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} Loss Curves (Mean ± SE)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_aggregated_loss_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset_name}: {e}")
        plt.close()

    # ----- accuracy -----
    try:
        epochs, train_mean, train_se = stack_metric(
            runs, lambda d, i: d["metrics"]["train"][i]["acc"]
        )
        _, val_mean, val_se = stack_metric(
            runs, lambda d, i: d["metrics"]["val"][i]["acc"]
        )
        plt.figure()
        plt.plot(epochs, train_mean, label="Train Mean")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, val_mean, label="Val Mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="Val ± SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name} Accuracy Curves (Mean ± SE)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_aggregated_accuracy_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dataset_name}: {e}")
        plt.close()

    # ----- CoWA -----
    try:
        epochs, val_mean, val_se = stack_metric(
            runs, lambda d, i: d["metrics"]["val"][i]["cowa"]
        )
        plt.figure()
        plt.plot(epochs, val_mean, label="Val CoWA Mean")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            alpha=0.3,
            label="Val CoWA ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{dataset_name} CoWA Curves (Mean ± SE)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_aggregated_cowa_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CoWA plot for {dataset_name}: {e}")
        plt.close()

# ---------- aggregated final test metrics ----------
try:
    test_accs, test_cowas = [], []

    def count_color_variety(sequence: str) -> int:
        return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

    def count_shape_variety(sequence: str) -> int:
        return len(set(tok[0] for tok in sequence.strip().split() if tok))

    def complexity_weight(seq: str) -> int:
        return count_color_variety(seq) + count_shape_variety(seq)

    for runs in dataset_runs.values():
        for ds in runs:
            preds = np.array(ds["predictions"])
            gts = np.array(ds["ground_truth"])
            seqs = np.array(ds["sequences"])
            acc = (preds == gts).mean()
            weights = np.array([complexity_weight(s) for s in seqs])
            cowa = (weights * (preds == gts)).sum() / weights.sum()
            test_accs.append(acc)
            test_cowas.append(cowa)

    test_accs = np.array(test_accs)
    test_cowas = np.array(test_cowas)
    acc_mean, acc_se = test_accs.mean(), (
        test_accs.std(ddof=1) / np.sqrt(len(test_accs)) if len(test_accs) > 1 else (0.0)
    )
    cowa_mean, cowa_se = test_cowas.mean(), (
        test_cowas.std(ddof=1) / np.sqrt(len(test_cowas))
        if len(test_cowas) > 1
        else (0.0)
    )
    print(
        f"Test Accuracy: {acc_mean:.3f} ± {acc_se:.3f} | Test CoWA: {cowa_mean:.3f} ± {cowa_se:.3f}"
    )
except Exception as e:
    print(f"Error computing aggregated evaluation metrics: {e}")

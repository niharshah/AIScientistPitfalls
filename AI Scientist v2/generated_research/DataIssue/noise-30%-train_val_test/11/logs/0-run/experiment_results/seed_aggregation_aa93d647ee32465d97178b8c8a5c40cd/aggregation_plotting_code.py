import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- paths / load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Absolute paths are constructed as  <AI_SCIENTIST_ROOT>/<experiment_data_path>
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_c3d9890bc0b4464e904f312c6a555af9_proc_3475997/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_a7ee26ae858c4d6b898b58164243d8a7_proc_3475996/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_2ab5aeac35c045e6b19785c5ab4e50e7_proc_3475999/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# --------------------------------------------------------------------- helper to aggregate
def aggregate_runs(model_name, dataset_name):
    """Return dict with aggregated arrays for train/val loss and f1 plus test f1 list"""
    train_loss_runs, val_loss_runs = [], []
    train_f1_runs, val_f1_runs, test_f1_scores = [], [], []

    for exp in all_experiment_data:
        try:
            rec = exp[model_name][dataset_name]
            epochs = rec.get("epochs", [])
            min_len = len(epochs)  # used later
            tl = np.array(rec["losses"].get("train", []), dtype=float)
            vl = np.array(rec["losses"].get("val", []), dtype=float)
            tf = np.array(rec["metrics"].get("train_macro_f1", []), dtype=float)
            vf = np.array(rec["metrics"].get("val_macro_f1", []), dtype=float)

            # ensure all run arrays are same length -> truncate to shortest
            min_len = min(len(tl), len(vl), len(tf), len(vf))
            if min_len == 0:
                continue
            train_loss_runs.append(tl[:min_len])
            val_loss_runs.append(vl[:min_len])
            train_f1_runs.append(tf[:min_len])
            val_f1_runs.append(vf[:min_len])
            test_f1_scores.append(float(rec.get("test_macro_f1", np.nan)))
        except Exception as e:
            print(f"Skipping a run due to error: {e}")

    if len(train_loss_runs) == 0:
        return None  # nothing to aggregate

    train_loss_arr = np.stack(train_loss_runs)
    val_loss_arr = np.stack(val_loss_runs)
    train_f1_arr = np.stack(train_f1_runs)
    val_f1_arr = np.stack(val_f1_runs)

    out = {
        "epochs": np.arange(train_loss_arr.shape[1]),
        "train_loss_mean": train_loss_arr.mean(0),
        "val_loss_mean": val_loss_arr.mean(0),
        "train_loss_se": train_loss_arr.std(0, ddof=1)
        / np.sqrt(train_loss_arr.shape[0]),
        "val_loss_se": val_loss_arr.std(0, ddof=1) / np.sqrt(val_loss_arr.shape[0]),
        "train_f1_mean": train_f1_arr.mean(0),
        "val_f1_mean": val_f1_arr.mean(0),
        "train_f1_se": train_f1_arr.std(0, ddof=1) / np.sqrt(train_f1_arr.shape[0]),
        "val_f1_se": val_f1_arr.std(0, ddof=1) / np.sqrt(val_f1_arr.shape[0]),
        "test_f1_scores": np.array(test_f1_scores, dtype=float),
    }
    return out


# we aggregate only for the dataset/model that exists in example
model_name = "NoPadMask_Transformer"
dataset_name = "SPR_BENCH"
agg = aggregate_runs(model_name, dataset_name)

if agg is None:
    print("No data available to plot.")
else:
    epochs = agg["epochs"]

    # ----------------------------------------------------------------- plot 1: aggregated loss
    try:
        plt.figure()
        plt.plot(epochs, agg["train_loss_mean"], label="Train mean")
        plt.fill_between(
            epochs,
            agg["train_loss_mean"] - agg["train_loss_se"],
            agg["train_loss_mean"] + agg["train_loss_se"],
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, agg["val_loss_mean"], label="Validation mean")
        plt.fill_between(
            epochs,
            agg["val_loss_mean"] - agg["val_loss_se"],
            agg["val_loss_mean"] + agg["val_loss_se"],
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{dataset_name} – Aggregated Loss Curves\nMean ± Standard Error over {len(agg['test_f1_scores'])} runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ----------------------------------------------------------------- plot 2: aggregated macro-F1
    try:
        plt.figure()
        plt.plot(epochs, agg["train_f1_mean"], label="Train mean")
        plt.fill_between(
            epochs,
            agg["train_f1_mean"] - agg["train_f1_se"],
            agg["train_f1_mean"] + agg["train_f1_se"],
            alpha=0.3,
            label="Train ± SE",
        )
        plt.plot(epochs, agg["val_f1_mean"], label="Validation mean")
        plt.fill_between(
            epochs,
            agg["val_f1_mean"] - agg["val_f1_se"],
            agg["val_f1_mean"] + agg["val_f1_se"],
            alpha=0.3,
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            f"{dataset_name} – Aggregated Macro-F1 Curves\nMean ± Standard Error over {len(agg['test_f1_scores'])} runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_agg_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot: {e}")
        plt.close()

    # ----------------------------------------------------------------- plot 3: test macro-F1 summary
    try:
        means = agg["test_f1_scores"].mean()
        se = agg["test_f1_scores"].std(ddof=1) / np.sqrt(len(agg["test_f1_scores"]))
        plt.figure()
        plt.bar([0], [means], yerr=[se], capsize=5, color="skyblue")
        plt.xticks([0], [dataset_name])
        plt.ylabel("Test Macro-F1")
        plt.title(
            f"{dataset_name} – Test Macro-F1\nMean ± SE across {len(agg['test_f1_scores'])} runs"
        )
        fname = os.path.join(working_dir, f"{dataset_name}_agg_test_macro_f1.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 summary plot: {e}")
        plt.close()

    # ----------------------------------------------------------------- console output
    test_mean = agg["test_f1_scores"].mean()
    test_se = agg["test_f1_scores"].std(ddof=1) / np.sqrt(len(agg["test_f1_scores"]))
    print(f"Aggregated Test Macro-F1: {test_mean:.4f} ± {test_se:.4f}")

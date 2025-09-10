import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load every experiment_data.npy that is available
# -------------------------------------------------------------------------
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_52d97c4e2ffe4d4fa1aa0de433d2c460_proc_3108301/experiment_data.npy",
    "experiments/2025-08-16_01-26-03_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_9045c0c70fc4496da84d8b317a5cdb9f_proc_3108302/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# -------------------------------------------------------------------------
# 2. Re-organise data: aggregated_data[ds_name] = { field : [run1, run2, ...] }
# -------------------------------------------------------------------------
aggregated_data = {}
for exp in all_experiment_data:
    for ds_name, ds_dict in exp.items():
        bucket = aggregated_data.setdefault(
            ds_name,
            {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "metrics": [],  # list of list-of-dict (one per run)
            },
        )
        bucket["epochs"].append(np.array(ds_dict.get("epochs", [])))
        bucket["train_loss"].append(
            np.array(ds_dict.get("losses", {}).get("train", []), dtype=float)
        )
        bucket["val_loss"].append(
            np.array(ds_dict.get("losses", {}).get("val", []), dtype=float)
        )
        bucket["metrics"].append(ds_dict.get("metrics", {}).get("val", []))

# -------------------------------------------------------------------------
# 3. Plot aggregated curves per dataset
# -------------------------------------------------------------------------
for ds_name, ds_dict in aggregated_data.items():
    # Align lengths to the shortest run so that we can stack safely
    n_runs = len(ds_dict["epochs"])
    if n_runs == 0:
        continue
    min_len = min(len(x) for x in ds_dict["epochs"])
    epochs = ds_dict["epochs"][0][:min_len]  # assume same epoch numbers

    train_loss_runs = np.vstack([x[:min_len] for x in ds_dict["train_loss"]])
    val_loss_runs = np.vstack([x[:min_len] for x in ds_dict["val_loss"]])

    # ---- Plot 1: aggregated loss ----
    try:
        plt.figure()
        # mean + stderr
        mean_train = train_loss_runs.mean(axis=0)
        se_train = train_loss_runs.std(axis=0, ddof=1) / np.sqrt(n_runs)
        mean_val = val_loss_runs.mean(axis=0)
        se_val = val_loss_runs.std(axis=0, ddof=1) / np.sqrt(n_runs)

        plt.plot(epochs, mean_train, "--", label="Train (mean)")
        plt.fill_between(
            epochs,
            mean_train - se_train,
            mean_train + se_train,
            alpha=0.2,
            label="Train SE",
        )
        plt.plot(epochs, mean_val, "-", label="Validation (mean)")
        plt.fill_between(
            epochs, mean_val - se_val, mean_val + se_val, alpha=0.2, label="Val SE"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"{ds_name} Aggregated Loss Curves\nMean with Standard Error over {n_runs} runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_aggregated_loss.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ---- Plot 2: aggregated validation metrics ----
    # First, collect metric arrays (some runs may lack them)
    try:
        metric_keys = ["SWA", "CWA", "SCAA"]
        metric_stacks = {k: [] for k in metric_keys}

        for run_metrics in ds_dict["metrics"]:
            if not run_metrics:
                continue
            # ensure same length
            run_metrics = run_metrics[:min_len]
            for k in metric_keys:
                metric_stacks[k].append(
                    np.array([m.get(k, np.nan) for m in run_metrics])
                )

        if any(len(v) for v in metric_stacks.values()):
            plt.figure()
            for k, stack in metric_stacks.items():
                if len(stack) == 0:
                    continue
                stack = np.vstack(stack)
                mean_k = np.nanmean(stack, axis=0)
                se_k = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                plt.plot(epochs, mean_k, label=f"{k} (mean)")
                plt.fill_between(
                    epochs, mean_k - se_k, mean_k + se_k, alpha=0.2, label=f"{k} SE"
                )

            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(
                f"{ds_name} Aggregated Validation Metrics\nMean ± SE over {n_runs} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_aggregated_val_metrics.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            raise ValueError("No metric arrays available for aggregation")
    except Exception as e:
        print(f"Error creating aggregated metric plot for {ds_name}: {e}")
        plt.close()

    # ---- Console summary of best mean metric ----
    try:
        best_metrics = {}
        for k, stack in metric_stacks.items():
            if len(stack):
                mean_curve = np.nanmean(np.vstack(stack), axis=0)
                best_metrics[k] = np.nanmax(mean_curve)
        if best_metrics:
            summary = ", ".join([f"{k}={v:.3f}" for k, v in best_metrics.items()])
            print(f"{ds_name} – best mean validation metrics: {summary}")
    except Exception as e:
        print(f"Error computing console summary for {ds_name}: {e}")

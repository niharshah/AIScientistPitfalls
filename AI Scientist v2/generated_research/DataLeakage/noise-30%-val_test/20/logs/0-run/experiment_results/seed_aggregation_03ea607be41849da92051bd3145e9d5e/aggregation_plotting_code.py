import matplotlib.pyplot as plt
import numpy as np
import os

# Basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- Load all experiments --------
experiment_data_path_list = [
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_23f1241f23654851b8188d3b076b39f3_proc_3440937/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_8d6d8b4b6b1449dab6d8b19b89e31971_proc_3440935/experiment_data.npy",
    "experiments/2025-08-17_22-28-20_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_842d6db6ed614f61a671e12a8c84ec82_proc_3440936/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -------- Aggregate data per dataset --------
aggregated = {}
for run in all_experiment_data:
    for dset, ddata in run.items():
        agg = aggregated.setdefault(
            dset, {"losses": {"train": [], "val": []}, "metrics": {"val": []}}
        )
        # collect curves
        losses = ddata.get("losses", {})
        metrics = ddata.get("metrics", {})
        if "train" in losses:
            agg["losses"]["train"].append(np.asarray(losses["train"]))
        if "val" in losses:
            agg["losses"]["val"].append(np.asarray(losses["val"]))
        if "val" in metrics:
            agg["metrics"]["val"].append(np.asarray(metrics["val"]))


# -------- Helper to compute mean & SE --------
def mean_se(arr_list):
    # trim to shortest length
    if not arr_list:
        return None, None, None
    min_len = min(len(a) for a in arr_list)
    arr = np.stack([a[:min_len] for a in arr_list], axis=0)
    mean = arr.mean(axis=0)
    se = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    epochs = np.arange(1, len(mean) + 1)
    return epochs, mean, se


# -------- Plot per dataset --------
for dset, ddata in aggregated.items():
    # 1) Aggregated loss curves
    try:
        epochs_tr, mean_tr, se_tr = mean_se(ddata["losses"]["train"])
        epochs_val, mean_val, se_val = mean_se(ddata["losses"]["val"])
        if mean_tr is not None or mean_val is not None:
            plt.figure()
            if mean_tr is not None:
                plt.plot(
                    epochs_tr, mean_tr, label="Train Loss (mean)", color="tab:blue"
                )
                plt.fill_between(
                    epochs_tr,
                    mean_tr - se_tr,
                    mean_tr + se_tr,
                    alpha=0.3,
                    color="tab:blue",
                    label="Train SE",
                )
            if mean_val is not None:
                plt.plot(
                    epochs_val, mean_val, label="Val Loss (mean)", color="tab:orange"
                )
                plt.fill_between(
                    epochs_val,
                    mean_val - se_val,
                    mean_val + se_val,
                    alpha=0.3,
                    color="tab:orange",
                    label="Val SE",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} Aggregated Loss Curves\nMean ± SE across runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # 2) Aggregated Macro-F1 curves
    try:
        epochs_f1, mean_f1, se_f1 = mean_se(ddata["metrics"]["val"])
        if mean_f1 is not None:
            plt.figure()
            plt.plot(epochs_f1, mean_f1, label="Val Macro-F1 (mean)", color="tab:green")
            plt.fill_between(
                epochs_f1,
                mean_f1 - se_f1,
                mean_f1 + se_f1,
                alpha=0.3,
                color="tab:green",
                label="Val SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dset} Aggregated Validation Macro-F1\nMean ± SE across runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_macroF1_curve_agg.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated Macro-F1 plot for {dset}: {e}")
        plt.close()

    # 3) Final Macro-F1 bar plot
    try:
        finals = [m[-1] for m in ddata["metrics"]["val"] if len(m)]
        if finals:
            runs = np.arange(len(finals))
            mean_final = np.mean(finals)
            se_final = (
                np.std(finals, ddof=1) / np.sqrt(len(finals))
                if len(finals) > 1
                else 0.0
            )
            plt.figure()
            plt.bar(
                runs, finals, color="tab:purple", alpha=0.7, label="Individual runs"
            )
            plt.hlines(
                mean_final,
                -0.5,
                len(runs) - 0.5,
                color="black",
                linestyle="--",
                label=f"Mean ± SE ({mean_final:.3f}±{se_final:.3f})",
            )
            plt.fill_between(
                [-0.5, len(runs) - 0.5],
                mean_final - se_final,
                mean_final + se_final,
                color="gray",
                alpha=0.2,
            )
            plt.xlabel("Run index")
            plt.ylabel("Final Macro-F1")
            plt.title(f"{dset} Final Validation Macro-F1 Across Runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_final_macroF1_bar.png")
            plt.savefig(fname)
            plt.close()
            print(
                f"{dset}: Aggregated final Macro-F1 = {mean_final:.4f} ± {se_final:.4f}"
            )
    except Exception as e:
        print(f"Error creating final Macro-F1 bar plot for {dset}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
# Absolute paths are resolved via AI_SCIENTIST_ROOT
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_5248448e90484b89b3dfd35fb4810f71_proc_3331035/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_f0f0d90379624347a5edd10be9d6689d_proc_3331034/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_a0774ba2f08842f08cbf2e4d38bb4683_proc_3331033/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_p = os.path.join(root, p)
        data = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------------------------------------------------------#
# Aggregate by dataset name
agg = {}
for run_data in all_experiment_data:
    for dname, dct in run_data.items():
        entry = agg.setdefault(
            dname,
            {
                "losses_tr": [],
                "losses_val": [],
                "f1_tr": [],
                "f1_val": [],
                "test_f1": [],
                "test_mcc": [],
            },
        )
        try:
            entry["losses_tr"].append(np.array(dct["losses"]["train"]))
            entry["losses_val"].append(np.array(dct["losses"]["val"]))
            entry["f1_tr"].append(np.array(dct["metrics"]["train"]))
            entry["f1_val"].append(np.array(dct["metrics"]["val"]))

            preds = np.array(dct["predictions"][0]).flatten()
            gts = np.array(dct["ground_truth"][0]).flatten()
            entry["test_f1"].append(f1_score(gts, preds, average="macro"))
            entry["test_mcc"].append(matthews_corrcoef(gts, preds))
        except Exception as e:
            print(f"Skipped run for {dname} due to missing keys: {e}")

# ------------------------------------------------------------------#
for dname, dct in agg.items():
    n_runs = len(dct["losses_tr"])
    if n_runs == 0:
        continue

    # Align epoch lengths to the shortest run
    min_len_loss = min(len(x) for x in dct["losses_tr"])
    min_len_f1 = min(len(x) for x in dct["f1_tr"])
    epochs_loss = np.arange(min_len_loss)
    epochs_f1 = np.arange(min_len_f1)

    # Stack & compute statistics
    def mean_se(arr_list, trim_len):
        arr = np.stack([a[:trim_len] for a in arr_list], axis=0)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    loss_tr_mu, loss_tr_se = mean_se(dct["losses_tr"], min_len_loss)
    loss_val_mu, loss_val_se = mean_se(dct["losses_val"], min_len_loss)
    f1_tr_mu, f1_tr_se = mean_se(dct["f1_tr"], min_len_f1)
    f1_val_mu, f1_val_se = mean_se(dct["f1_val"], min_len_f1)

    # --------------------- Aggregated Loss curves ----------------------------#
    try:
        plt.figure()
        plt.plot(epochs_loss, loss_tr_mu, label="Train Mean", color="steelblue")
        plt.fill_between(
            epochs_loss,
            loss_tr_mu - loss_tr_se,
            loss_tr_mu + loss_tr_se,
            alpha=0.3,
            color="steelblue",
            label="Train ± SE",
        )
        plt.plot(epochs_loss, loss_val_mu, label="Val Mean", color="orange")
        plt.fill_between(
            epochs_loss,
            loss_val_mu - loss_val_se,
            loss_val_mu + loss_val_se,
            alpha=0.3,
            color="orange",
            label="Val ± SE",
        )
        plt.title(f"{dname} Aggregated Loss Curves\nMean ± SE over {n_runs} runs")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_aggregated_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # -------------------- Aggregated F1 curves -------------------------------#
    try:
        plt.figure()
        plt.plot(epochs_f1, f1_tr_mu, label="Train Mean", color="steelblue")
        plt.fill_between(
            epochs_f1,
            f1_tr_mu - f1_tr_se,
            f1_tr_mu + f1_tr_se,
            alpha=0.3,
            color="steelblue",
            label="Train ± SE",
        )
        plt.plot(epochs_f1, f1_val_mu, label="Val Mean", color="orange")
        plt.fill_between(
            epochs_f1,
            f1_val_mu - f1_val_se,
            f1_val_mu + f1_val_se,
            alpha=0.3,
            color="orange",
            label="Val ± SE",
        )
        plt.title(f"{dname} Aggregated Macro-F1 Curves\nMean ± SE over {n_runs} runs")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_aggregated_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 plot for {dname}: {e}")
        plt.close()

    # ----------------- Aggregated Test metrics bar chart ---------------------#
    try:
        test_f1_arr = np.array(dct["test_f1"])
        test_mcc_arr = np.array(dct["test_mcc"])
        metrics_mean = [test_f1_arr.mean(), test_mcc_arr.mean()]
        metrics_se = [
            test_f1_arr.std(ddof=1) / np.sqrt(n_runs),
            test_mcc_arr.std(ddof=1) / np.sqrt(n_runs),
        ]

        plt.figure()
        bars = plt.bar(
            ["Macro-F1", "MCC"],
            metrics_mean,
            yerr=metrics_se,
            capsize=5,
            color=["steelblue", "orange"],
        )
        plt.ylim(0, 1)
        plt.title(f"{dname} Test Metrics (Mean ± SE)\nAggregated over {n_runs} runs")
        for i, v in enumerate(metrics_mean):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(
            working_dir, f"{dname.lower()}_aggregated_test_metrics.png"
        )
        plt.savefig(fname)
        plt.close()

        print(
            f"{dname} | Test Macro-F1: {metrics_mean[0]:.4f}±{metrics_se[0]:.4f} | "
            f"Test MCC: {metrics_mean[1]:.4f}±{metrics_se[1]:.4f}"
        )
    except Exception as e:
        print(f"Error creating aggregated test metrics for {dname}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------- #
# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------- #
# Load all experiment files that were provided
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_539ca82a3866414cab4bd8ff6d8ff6bb_proc_3166753/experiment_data.npy",
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_d2bcc9e630bb43de8231e19d2ecd426f_proc_3166756/experiment_data.npy",
        "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_02d7219236554d15982daadaf051d464_proc_3166755/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        try:
            d = np.load(
                os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
            ).item()
            all_experiment_data.append(d)
        except Exception as e:
            print(f"Error loading {p}: {e}")
except Exception as e:
    print(f"Error building experiment path list: {e}")
    all_experiment_data = []

# --------------------------------------- #
# Aggregate results across runs
# Structure: aggregated[dataset_name][nhead] -> list_of_run_dicts
aggregated = {}
for exp in all_experiment_data:
    try:
        for tuning_key in exp:
            # Expect "nhead_tuning"
            for dataset_name, dataset_dict in exp[tuning_key].items():
                res = dataset_dict.get("results", {})
                if dataset_name not in aggregated:
                    aggregated[dataset_name] = {}
                for nhead, run_data in res.items():
                    aggregated[dataset_name].setdefault(nhead, []).append(run_data)
    except Exception as e:
        print(f"Aggregation error: {e}")


# Helper to compute mean and stderr for list of 1-D arrays that could have variable lengths
def _stack_with_padding(arr_list, fill_val=np.nan):
    max_len = max(len(a) for a in arr_list)
    stacked = np.full((len(arr_list), max_len), fill_val, dtype=float)
    for i, a in enumerate(arr_list):
        stacked[i, : len(a)] = a
    return stacked


# --------------------------------------- #
# Generate plots for every dataset
for dataset_name, nhead_dict in aggregated.items():

    # 1) Accuracy curves with std-error
    try:
        plt.figure()
        for nhead, run_list in nhead_dict.items():
            # gather train & val accuracies for this nhead
            train_runs = [
                _stack_with_padding([r["metrics"]["train_acc"]])
                for r in run_list
                if "metrics" in r
            ]
            val_runs = [
                _stack_with_padding([r["metrics"]["val_acc"]])
                for r in run_list
                if "metrics" in r
            ]
            if not train_runs or not val_runs:
                continue
            train_stack = _stack_with_padding(
                [r["metrics"]["train_acc"] for r in run_list]
            )
            val_stack = _stack_with_padding([r["metrics"]["val_acc"] for r in run_list])

            epochs = np.arange(1, train_stack.shape[1] + 1)
            # compute mean and stderr ignoring NaNs
            train_mean = np.nanmean(train_stack, axis=0)
            val_mean = np.nanmean(val_stack, axis=0)
            train_se = np.nanstd(train_stack, axis=0) / np.sqrt(train_stack.shape[0])
            val_se = np.nanstd(val_stack, axis=0) / np.sqrt(val_stack.shape[0])

            plt.plot(epochs, train_mean, label=f"Train μ nhead={nhead}")
            plt.fill_between(
                epochs, train_mean - train_se, train_mean + train_se, alpha=0.2
            )

            plt.plot(epochs, val_mean, linestyle="--", label=f"Val μ nhead={nhead}")
            plt.fill_between(epochs, val_mean - val_se, val_mean + val_se, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name} Accuracy (Mean ± SE across runs)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_agg_accuracy_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dataset_name}: {e}")
        plt.close()

    # 2) Loss curves with std-error
    try:
        plt.figure()
        for nhead, run_list in nhead_dict.items():
            train_stack = _stack_with_padding(
                [r["losses"]["train_loss"] for r in run_list if "losses" in r]
            )
            val_stack = _stack_with_padding(
                [r["losses"]["val_loss"] for r in run_list if "losses" in r]
            )
            if train_stack.size == 0 or val_stack.size == 0:
                continue
            epochs = np.arange(1, train_stack.shape[1] + 1)
            train_m = np.nanmean(train_stack, axis=0)
            val_m = np.nanmean(val_stack, axis=0)
            train_se = np.nanstd(train_stack, axis=0) / np.sqrt(train_stack.shape[0])
            val_se = np.nanstd(val_stack, axis=0) / np.sqrt(val_stack.shape[0])

            plt.plot(epochs, train_m, label=f"Train μ nhead={nhead}")
            plt.fill_between(epochs, train_m - train_se, train_m + train_se, alpha=0.2)

            plt.plot(epochs, val_m, linestyle="--", label=f"Val μ nhead={nhead}")
            plt.fill_between(epochs, val_m - val_se, val_m + val_se, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} Loss (Mean ± SE across runs)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_agg_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dataset_name}: {e}")
        plt.close()

    # 3) Final test accuracy bar chart with error bars
    try:
        plt.figure()
        nheads = []
        means = []
        ses = []
        for nhead, run_list in nhead_dict.items():
            test_accs = [r["test_acc"] for r in run_list if "test_acc" in r]
            if not test_accs:
                continue
            nheads.append(nhead)
            means.append(np.mean(test_accs))
            ses.append(np.std(test_accs) / np.sqrt(len(test_accs)))
        if nheads:
            x = np.arange(len(nheads))
            plt.bar(x, means, yerr=ses, capsize=5, color="skyblue")
            plt.xticks(x, nheads)
            plt.xlabel("n-head")
            plt.ylabel("Test Accuracy")
            plt.title(f"{dataset_name} Test Accuracy (Mean ± SE)")
            fname = os.path.join(working_dir, f"{dataset_name}_agg_test_accuracy.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test accuracy bar for {dataset_name}: {e}")
        plt.close()

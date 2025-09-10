import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load all experiment_data dicts
# ------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9e43cfc83f5e4caebf3287444cb52c61_proc_3216350/experiment_data.npy",
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_79fe77dace8f4c74af759cb8012bc2aa_proc_3216347/experiment_data.npy",
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9ca88d7f7c7946599ef964200b88dd44_proc_3216348/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_p, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ------------------------------------------------------------------
# gather metrics for every dataset present
# ------------------------------------------------------------------
datasets = {}  # {dataset_name: {metric_name: list_of_arrays_or_scalars}}
for run_data in all_experiment_data:
    for model_key in run_data:
        for dset_key, ed in run_data[model_key].items():
            m = ed.get("metrics", {})
            losses = ed.get("losses", {})
            ds_dict = datasets.setdefault(
                dset_key,
                {
                    "train_acc": [],
                    "val_acc": [],
                    "train_loss": [],
                    "val_loss": [],
                    "val_rfs": [],
                    "test_acc": [],
                    "test_rfs": [],
                },
            )
            # curves
            for k in ["train_acc", "val_acc", "val_rfs"]:
                if k in m:
                    ds_dict[k].append(np.asarray(m[k]))
            if "train" in losses:
                ds_dict["train_loss"].append(np.asarray(losses["train"]))
            if "val_loss" in m:
                ds_dict["val_loss"].append(np.asarray(m["val_loss"]))
            # final metrics
            if "test_acc" in ed:
                ds_dict["test_acc"].append(ed["test_acc"])
            if "test_rfs" in ed:
                ds_dict["test_rfs"].append(ed["test_rfs"])


# ------------------------------------------------------------------
# helper to compute mean and se with length alignment
# ------------------------------------------------------------------
def mean_se(list_of_arrays):
    if len(list_of_arrays) == 0:
        return None, None
    min_len = min(len(a) for a in list_of_arrays)
    arr = np.stack([a[:min_len] for a in list_of_arrays])  # shape: (runs, epochs)
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    epochs = np.arange(1, min_len + 1)
    return epochs, (mean, se)


# ------------------------------------------------------------------
# plotting per dataset (max 4 distinct figures)
# ------------------------------------------------------------------
for dset, vals in datasets.items():
    # 1. Accuracy ----------------------------------------------------
    try:
        ep_train, (mean_train, se_train) = mean_se(vals["train_acc"])
        ep_val, (mean_val, se_val) = mean_se(vals["val_acc"])
        if ep_train is not None and ep_val is not None:
            plt.figure()
            plt.plot(ep_train, mean_train, label="Train Acc (mean)")
            plt.fill_between(
                ep_train,
                mean_train - se_train,
                mean_train + se_train,
                alpha=0.3,
                label="Train ±SE",
            )
            plt.plot(ep_val, mean_val, label="Val Acc (mean)")
            plt.fill_between(
                ep_val, mean_val - se_val, mean_val + se_val, alpha=0.3, label="Val ±SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset}: Training vs Validation Accuracy (Aggregated)")
            plt.legend()
            out = os.path.join(working_dir, f"{dset}_aggregated_accuracy.png")
            plt.savefig(out)
            print(f"Saved {out}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset}: {e}")
        plt.close()

    # 2. Loss --------------------------------------------------------
    try:
        ep_tr_loss, (mean_tr_loss, se_tr_loss) = mean_se(vals["train_loss"])
        ep_val_loss, (mean_val_loss, se_val_loss) = mean_se(vals["val_loss"])
        if ep_tr_loss is not None and ep_val_loss is not None:
            plt.figure()
            plt.plot(ep_tr_loss, mean_tr_loss, label="Train Loss (mean)")
            plt.fill_between(
                ep_tr_loss,
                mean_tr_loss - se_tr_loss,
                mean_tr_loss + se_tr_loss,
                alpha=0.3,
                label="Train ±SE",
            )
            plt.plot(ep_val_loss, mean_val_loss, label="Val Loss (mean)")
            plt.fill_between(
                ep_val_loss,
                mean_val_loss - se_val_loss,
                mean_val_loss + se_val_loss,
                alpha=0.3,
                label="Val ±SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset}: Training vs Validation Loss (Aggregated)")
            plt.legend()
            out = os.path.join(working_dir, f"{dset}_aggregated_loss.png")
            plt.savefig(out)
            print(f"Saved {out}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # 3. Rule-Faithfulness ------------------------------------------
    try:
        ep_rfs, (mean_rfs, se_rfs) = mean_se(vals["val_rfs"])
        if ep_rfs is not None:
            plt.figure()
            plt.plot(ep_rfs, mean_rfs, marker="o", label="Val RFS (mean)")
            plt.fill_between(
                ep_rfs, mean_rfs - se_rfs, mean_rfs + se_rfs, alpha=0.3, label="Val ±SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("RFS")
            plt.title(f"{dset}: Validation Rule-Faithfulness Score (Aggregated)")
            plt.legend()
            out = os.path.join(working_dir, f"{dset}_aggregated_rfs.png")
            plt.savefig(out)
            print(f"Saved {out}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated RFS plot for {dset}: {e}")
        plt.close()

    # 4. Final test metrics -----------------------------------------
    try:
        tacc = np.asarray(vals["test_acc"])
        trfs = np.asarray(vals["test_rfs"])
        if tacc.size and trfs.size:
            means = [tacc.mean(), trfs.mean()]
            ses = [
                tacc.std(ddof=1) / np.sqrt(len(tacc)),
                trfs.std(ddof=1) / np.sqrt(len(trfs)),
            ]
            labels = ["Test Acc", "Test RFS"]
            plt.figure()
            x = np.arange(len(labels))
            plt.bar(x, means, yerr=ses, capsize=5, color=["steelblue", "orange"])
            plt.ylim(0, 1)
            plt.xticks(x, labels)
            plt.title(f"{dset}: Final Test Metrics (Aggregated)")
            for i, v in enumerate(means):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            out = os.path.join(working_dir, f"{dset}_aggregated_test_metrics.png")
            plt.savefig(out)
            print(f"Saved {out}")
            # print numerical values to console
            print(
                f"{dset} - Test Acc mean±SE: {means[0]:.3f}±{ses[0]:.3f}, "
                f"Test RFS mean±SE: {means[1]:.3f}±{ses[1]:.3f}"
            )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric plot for {dset}: {e}")
        plt.close()

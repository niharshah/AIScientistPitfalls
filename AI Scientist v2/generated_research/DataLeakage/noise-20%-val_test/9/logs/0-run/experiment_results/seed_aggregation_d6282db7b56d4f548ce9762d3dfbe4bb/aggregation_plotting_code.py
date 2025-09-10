import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths (provided) ----------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_67c8ec0854c64b15929b376f6d333a48_proc_3204919/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9498c6e0ecb24213ac2d3a12541a47ec_proc_3204920/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_5677ab331cb645589beb2cedcbd57ec7_proc_3204921/experiment_data.npy",
]

# ---------- load ----------
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ---------- helper ----------
def collect_metric(run_list, dataset, category, key):
    """Return list of 1-D np.arrays (one per run) for metric `key` located at data[dataset][category][key]."""
    arrs = []
    for run in run_list:
        dset_dict = run.get(dataset, {})
        cat_dict = dset_dict.get(category, {})
        if key in cat_dict:
            arrs.append(np.asarray(cat_dict[key], dtype=float))
    return arrs


def pad_to_equal_length(arr_list, pad_val=np.nan):
    max_len = max(len(a) for a in arr_list)
    out = []
    for a in arr_list:
        if len(a) < max_len:
            pad = np.full(max_len - len(a), pad_val, dtype=float)
            out.append(np.concatenate([a, pad]))
        else:
            out.append(a)
    return np.vstack(out)  # shape (n_runs, max_len)


# ---------- iterate over datasets ----------
if not all_experiment_data:
    print("No experiment data loaded – nothing to plot.")
else:
    example_run = all_experiment_data[0]
    dataset_names = example_run.keys()

    for dset in dataset_names:

        # ============ Accuracy curves (train & val) ============
        try:
            train_arrs = collect_metric(
                all_experiment_data, dset, "metrics", "train_acc"
            )
            val_arrs = collect_metric(all_experiment_data, dset, "metrics", "val_acc")
            if train_arrs and val_arrs:
                train_mat = pad_to_equal_length(train_arrs)
                val_mat = pad_to_equal_length(val_arrs)

                epochs = np.arange(1, train_mat.shape[1] + 1)

                train_mean = np.nanmean(train_mat, axis=0)
                val_mean = np.nanmean(val_mat, axis=0)
                train_sem = np.nanstd(train_mat, axis=0, ddof=1) / np.sqrt(
                    train_mat.shape[0]
                )
                val_sem = np.nanstd(val_mat, axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

                plt.figure()
                plt.fill_between(
                    epochs,
                    train_mean - train_sem,
                    train_mean + train_sem,
                    alpha=0.2,
                    label="Train SEM",
                )
                plt.fill_between(
                    epochs,
                    val_mean - val_sem,
                    val_mean + val_sem,
                    alpha=0.2,
                    label="Val SEM",
                )
                plt.plot(epochs, train_mean, label="Train Mean")
                plt.plot(epochs, val_mean, label="Val Mean")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"{dset}: Mean ± SEM Train vs Validation Accuracy")
                plt.legend()
                fname = f"{dset}_agg_accuracy_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated accuracy plot for {dset}: {e}")
            plt.close()

        # ============ Loss curves ============
        try:
            train_arrs = collect_metric(all_experiment_data, dset, "losses", "train")
            val_arrs = collect_metric(all_experiment_data, dset, "losses", "val")
            if train_arrs and val_arrs:
                train_mat = pad_to_equal_length(train_arrs)
                val_mat = pad_to_equal_length(val_arrs)

                epochs = np.arange(1, train_mat.shape[1] + 1)

                train_mean = np.nanmean(train_mat, axis=0)
                val_mean = np.nanmean(val_mat, axis=0)
                train_sem = np.nanstd(train_mat, axis=0, ddof=1) / np.sqrt(
                    train_mat.shape[0]
                )
                val_sem = np.nanstd(val_mat, axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

                plt.figure()
                plt.fill_between(
                    epochs,
                    train_mean - train_sem,
                    train_mean + train_sem,
                    alpha=0.2,
                    label="Train SEM",
                )
                plt.fill_between(
                    epochs,
                    val_mean - val_sem,
                    val_mean + val_sem,
                    alpha=0.2,
                    label="Val SEM",
                )
                plt.plot(epochs, train_mean, label="Train Mean")
                plt.plot(epochs, val_mean, label="Val Mean")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dset}: Mean ± SEM Train vs Validation Loss")
                plt.legend()
                fname = f"{dset}_agg_loss_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {dset}: {e}")
            plt.close()

        # ============ RBA vs Val Acc ============
        try:
            rba_arrs = collect_metric(all_experiment_data, dset, "metrics", "RBA")
            val_arrs = collect_metric(all_experiment_data, dset, "metrics", "val_acc")
            if rba_arrs and val_arrs:
                rba_mat = pad_to_equal_length(rba_arrs)
                val_mat = pad_to_equal_length(val_arrs)

                epochs = np.arange(1, rba_mat.shape[1] + 1)
                rba_mean = np.nanmean(rba_mat, axis=0)
                val_mean = np.nanmean(val_mat, axis=0)
                rba_sem = np.nanstd(rba_mat, axis=0, ddof=1) / np.sqrt(rba_mat.shape[0])
                val_sem = np.nanstd(val_mat, axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

                plt.figure()
                plt.fill_between(
                    epochs,
                    rba_mean - rba_sem,
                    rba_mean + rba_sem,
                    alpha=0.2,
                    label="RBA SEM",
                )
                plt.fill_between(
                    epochs,
                    val_mean - val_sem,
                    val_mean + val_sem,
                    alpha=0.2,
                    label="Val SEM",
                )
                plt.plot(epochs, val_mean, label="Val Mean")
                plt.plot(epochs, rba_mean, label="RBA Mean")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(
                    f"{dset}: Mean ± SEM Validation Accuracy vs Rule-Based Accuracy"
                )
                plt.legend()
                fname = f"{dset}_agg_rba_vs_val.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated RBA plot for {dset}: {e}")
            plt.close()

        # ============ Aggregate final test accuracy ============
        try:
            final_test_accs = []
            for run in all_experiment_data:
                preds = np.asarray(run.get(dset, {}).get("predictions", []))
                gts = np.asarray(run.get(dset, {}).get("ground_truth", []))
                if preds.size and preds.shape == gts.shape:
                    final_test_accs.append((preds == gts).mean())
            if final_test_accs:
                mean_acc = np.mean(final_test_accs)
                sem_acc = np.std(final_test_accs, ddof=1) / np.sqrt(
                    len(final_test_accs)
                )
                print(
                    f"{dset} – Aggregate Test Accuracy: {mean_acc:.3f} ± {sem_acc:.3f} (SEM, n={len(final_test_accs)})"
                )
        except Exception as e:
            print(f"Error computing aggregate test accuracy for {dset}: {e}")

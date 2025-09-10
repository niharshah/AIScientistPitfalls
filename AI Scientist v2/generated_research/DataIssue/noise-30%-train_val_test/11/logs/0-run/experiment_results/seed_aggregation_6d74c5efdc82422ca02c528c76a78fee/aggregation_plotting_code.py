import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# basic setup
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load all experiment dicts
# ------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_241b1da50ea546b3a95b3bb9d0e048a1_proc_3458377/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_f101ff6bec5243e0933a58520fe43b7d_proc_3458379/experiment_data.npy",
    "experiments/2025-08-17_23-44-10_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_32b635f2d0db4b0ca3ff425beea3d3ba_proc_3458378/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_p = os.path.join(root, p) if not os.path.isabs(p) else p
        exp_dict = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# ------------------------------------------------------------
# aggregate per dataset
# ------------------------------------------------------------
datasets = set()
for d in all_experiment_data:
    datasets.update(d.keys())

for ds_name in datasets:
    # --------------------------------------------------------
    # gather run-wise arrays
    # --------------------------------------------------------
    train_loss_lst, val_loss_lst = [], []
    train_f1_lst, val_f1_lst = [], []
    preds_lst, trues_lst = [], []
    epochs_ref = None
    for run_dict in all_experiment_data:
        ds_dict = run_dict.get(ds_name, {})
        if not ds_dict:
            continue
        epochs = np.array(ds_dict.get("epochs", []))
        if epochs_ref is None:
            epochs_ref = epochs
        else:
            # align to shortest
            min_len = min(len(epochs_ref), len(epochs))
            epochs_ref = epochs_ref[:min_len]
            epochs = epochs[:min_len]
        train_loss_lst.append(
            np.array(ds_dict.get("losses", {}).get("train", []))[: len(epochs_ref)]
        )
        val_loss_lst.append(
            np.array(ds_dict.get("losses", {}).get("val", []))[: len(epochs_ref)]
        )
        train_f1_lst.append(
            np.array(ds_dict.get("metrics", {}).get("train_macro_f1", []))[
                : len(epochs_ref)
            ]
        )
        val_f1_lst.append(
            np.array(ds_dict.get("metrics", {}).get("val_macro_f1", []))[
                : len(epochs_ref)
            ]
        )
        preds = np.array(ds_dict.get("predictions", []))
        trues = np.array(ds_dict.get("ground_truth", []))
        if preds.size and trues.size:
            preds_lst.append(preds)
            trues_lst.append(trues)

    # skip dataset if nothing loaded
    if not train_loss_lst:
        continue

    # --------------------------------------------------------
    # stack and compute mean / stderr
    # --------------------------------------------------------
    def _mean_se(arr_list):
        arr = np.stack(arr_list, axis=0)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    mean_train_loss, se_train_loss = _mean_se(train_loss_lst)
    mean_val_loss, se_val_loss = _mean_se(val_loss_lst)
    mean_train_f1, se_train_f1 = _mean_se(train_f1_lst)
    mean_val_f1, se_val_f1 = _mean_se(val_f1_lst)

    # --------------------------------------------------------
    # PLOT 1: aggregated loss curves
    # --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs_ref, mean_train_loss, label="Train Mean", color="blue")
        plt.fill_between(
            epochs_ref,
            mean_train_loss - se_train_loss,
            mean_train_loss + se_train_loss,
            alpha=0.2,
            color="blue",
            label="Train ± SE",
        )
        plt.plot(epochs_ref, mean_val_loss, label="Val Mean", color="orange")
        plt.fill_between(
            epochs_ref,
            mean_val_loss - se_val_loss,
            mean_val_loss + se_val_loss,
            alpha=0.2,
            color="orange",
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Aggregated Loss Curves\nMean ± Standard Error over Runs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_agg_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
        plt.close()

    # --------------------------------------------------------
    # PLOT 2: aggregated macro-F1 curves
    # --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs_ref, mean_train_f1, label="Train Mean", color="green")
        plt.fill_between(
            epochs_ref,
            mean_train_f1 - se_train_f1,
            mean_train_f1 + se_train_f1,
            alpha=0.2,
            color="green",
            label="Train ± SE",
        )
        plt.plot(epochs_ref, mean_val_f1, label="Val Mean", color="red")
        plt.fill_between(
            epochs_ref,
            mean_val_f1 - se_val_f1,
            mean_val_f1 + se_val_f1,
            alpha=0.2,
            color="red",
            label="Val ± SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            f"{ds_name} Aggregated Macro-F1 Curves\nMean ± Standard Error over Runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_agg_macro_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 curve for {ds_name}: {e}")
        plt.close()

    # --------------------------------------------------------
    # PLOT 3: final validation macro-F1 across runs
    # --------------------------------------------------------
    try:
        final_val_f1 = [arr[-1] for arr in val_f1_lst]
        mean_final = np.mean(final_val_f1)
        se_final = np.std(final_val_f1, ddof=1) / np.sqrt(len(final_val_f1))
        plt.figure()
        plt.bar(
            [0],
            [mean_final],
            yerr=[se_final],
            capsize=8,
            color="skyblue",
            label=f"n={len(final_val_f1)} runs",
        )
        plt.ylabel("Final Validation Macro-F1")
        plt.xticks([])
        plt.title(
            f"{ds_name} Final Validation Macro-F1\nMean ± Standard Error over Runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_final_val_macro_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final F1 bar for {ds_name}: {e}")
        plt.close()

    # --------------------------------------------------------
    # PLOT 4: aggregated confusion matrix (optional)
    # --------------------------------------------------------
    try:
        if preds_lst and trues_lst:
            preds_all = np.concatenate(preds_lst)
            trues_all = np.concatenate(trues_lst)
            num_classes = int(max(preds_all.max(), trues_all.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(trues_all, preds_all):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Aggregated Predictions"
            )
            fname = os.path.join(working_dir, f"{ds_name}_agg_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
        plt.close()

    # --------------------------------------------------------
    # PLOT 5: aggregated label distribution (optional)
    # --------------------------------------------------------
    try:
        if preds_lst and trues_lst:
            preds_all = np.concatenate(preds_lst)
            trues_all = np.concatenate(trues_lst)
            num_classes = int(max(preds_all.max(), trues_all.max())) + 1
            idx = np.arange(num_classes)
            width = 0.35
            plt.figure()
            plt.bar(
                idx - width / 2,
                np.bincount(trues_all, minlength=num_classes),
                width,
                label="Ground Truth",
            )
            plt.bar(
                idx + width / 2,
                np.bincount(preds_all, minlength=num_classes),
                width,
                label="Predictions",
            )
            plt.xlabel("Label ID")
            plt.ylabel("Count")
            plt.title(
                f"{ds_name} Label Distribution\nGround Truth vs Aggregated Predictions"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_label_distribution.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated label distribution for {ds_name}: {e}")
        plt.close()

    # --------------------------------------------------------
    # print optional overall test macro-F1 if available
    # --------------------------------------------------------
    try:
        from sklearn.metrics import f1_score

        if preds_lst and trues_lst:
            print(
                f"{ds_name} Aggregated Test Macro-F1:",
                f1_score(
                    np.concatenate(trues_lst),
                    np.concatenate(preds_lst),
                    average="macro",
                ),
            )
    except Exception as e:
        print(f"Could not compute aggregated F1 for {ds_name}: {e}")

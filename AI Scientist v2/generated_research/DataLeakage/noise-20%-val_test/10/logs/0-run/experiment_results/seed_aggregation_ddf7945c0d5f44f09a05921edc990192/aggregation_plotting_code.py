import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_42547dbfef184ebb923db3cc24284519_proc_3305854/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_aeef92afb53d4b2291b184d257cd1b7b_proc_3305855/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9deb35bbf816471d8d030c19bfbe48d2_proc_3305857/experiment_data.npy",
]

# ---------- load data ----------
all_experiment_data = []
for exp_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_dct = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dct)
    except Exception as e:
        print(f"Error loading {exp_path}: {e}")

# ---------- aggregate by dataset ----------
aggregated = {}  # {dataset : [run_dict, ...]}
for run in all_experiment_data:
    for dname, dct in run.items():
        aggregated.setdefault(dname, []).append(dct)

plot_cap, plotted = 5, 0


# ---------- helper ----------
def stack_and_trim(list_of_lists):
    """Stack ragged list of 1-D arrays by trimming to shortest length."""
    min_len = min(len(a) for a in list_of_lists)
    arr = np.vstack([np.asarray(a[:min_len]) for a in list_of_lists])
    return arr, min_len


def sem(x, axis=0):
    n = x.shape[axis]
    return np.std(x, axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(x.shape[1])


def macro_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# ---------- plotting aggregated results ----------
for dname, runs in aggregated.items():
    if plotted >= plot_cap:
        break
    n_runs = len(runs)

    # ========== Aggregated Loss Curves ==========
    try:
        tr_losses = [
            run["losses"]["train"]
            for run in runs
            if "losses" in run and "train" in run["losses"]
        ]
        val_losses = [
            run["losses"]["val"]
            for run in runs
            if "losses" in run and "val" in run["losses"]
        ]
        if tr_losses and val_losses:
            tr_arr, L = stack_and_trim(tr_losses)
            val_arr, _ = stack_and_trim(val_losses)
            tr_mean, tr_sem = tr_arr.mean(0), sem(tr_arr)
            val_mean, val_sem = val_arr.mean(0), sem(val_arr)

            epochs = np.arange(1, L + 1)
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train Mean", color="steelblue")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.3,
                color="steelblue",
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, label="Validation Mean", color="darkorange")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                color="darkorange",
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dname}: Aggregated Loss Curves\nMean with Shaded Standard Error over {n_runs} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_aggregated_loss.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error plotting aggregated loss for {dname}: {e}")
        plt.close()

    # ========== Aggregated F1 Curves ==========
    try:
        tr_f1s = [
            run["metrics"]["train_f1"]
            for run in runs
            if "metrics" in run and "train_f1" in run["metrics"]
        ]
        val_f1s = [
            run["metrics"]["val_f1"]
            for run in runs
            if "metrics" in run and "val_f1" in run["metrics"]
        ]
        if tr_f1s and val_f1s and plotted < plot_cap:
            tr_arr, Lf = stack_and_trim(tr_f1s)
            val_arr, _ = stack_and_trim(val_f1s)
            tr_mean, tr_sem = tr_arr.mean(0), sem(tr_arr)
            val_mean, val_sem = val_arr.mean(0), sem(val_arr)

            epochs = np.arange(1, Lf + 1)
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train Mean", color="seagreen")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.3,
                color="seagreen",
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, label="Validation Mean", color="tomato")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                color="tomato",
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(
                f"{dname}: Aggregated Macro-F1 Curves\nMean with Shaded Standard Error over {n_runs} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_aggregated_f1.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error plotting aggregated F1 for {dname}: {e}")
        plt.close()

    # ========== Aggregated Rule-Extraction Accuracy ==========
    try:
        if plotted < plot_cap:
            rea_dev_vals = [
                run["metrics"].get("REA_dev") for run in runs if "metrics" in run
            ]
            rea_test_vals = [
                run["metrics"].get("REA_test") for run in runs if "metrics" in run
            ]
            if all(isinstance(v, (int, float)) for v in rea_dev_vals) and all(
                isinstance(v, (int, float)) for v in rea_test_vals
            ):
                dev_mean, dev_sem = np.mean(rea_dev_vals), (
                    np.std(rea_dev_vals, ddof=1) / np.sqrt(n_runs)
                    if n_runs > 1
                    else 0.0
                )
                test_mean, test_sem = np.mean(rea_test_vals), (
                    np.std(rea_test_vals, ddof=1) / np.sqrt(n_runs)
                    if n_runs > 1
                    else 0.0
                )

                plt.figure()
                plt.bar(
                    ["Dev", "Test"],
                    [dev_mean, test_mean],
                    yerr=[dev_sem, test_sem],
                    color=["skyblue", "salmon"],
                    capsize=5,
                )
                plt.ylim(0, 1)
                plt.ylabel("Accuracy")
                plt.title(
                    f"{dname}: Aggregated Rule-Extraction Accuracy\nBars show mean ±SEM over {n_runs} runs"
                )
                fname = os.path.join(working_dir, f"{dname.lower()}_aggregated_rea.png")
                plt.savefig(fname)
                plt.close()
                plotted += 1
    except Exception as e:
        print(f"Error plotting aggregated REA for {dname}: {e}")
        plt.close()

    # ========== Aggregated Confusion Matrix ==========
    try:
        if plotted < plot_cap:
            gts_all, preds_all = [], []
            for run in runs:
                if (
                    run.get("gts_test") is not None
                    and run.get("preds_test") is not None
                ):
                    gts_all.append(np.asarray(run["gts_test"]))
                    preds_all.append(np.asarray(run["preds_test"]))

            if gts_all and preds_all:
                gts_concat = np.concatenate(gts_all)
                preds_concat = np.concatenate(preds_all)
                labels = np.unique(np.concatenate([gts_concat, preds_concat]))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for t, p in zip(gts_concat, preds_concat):
                    cm[t, p] += 1

                # print aggregated macro-F1
                mean_f1 = macro_f1(gts_concat, preds_concat)
                print(
                    f"{dname} Aggregated Test Macro-F1 over {n_runs} runs: {mean_f1:.4f}"
                )

                plt.figure(figsize=(6, 5))
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(
                    f"{dname}: Aggregated Confusion Matrix (Test)\nCounts summed over {n_runs} runs"
                )
                plt.xticks(labels)
                plt.yticks(labels)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
                fname = os.path.join(
                    working_dir, f"{dname.lower()}_aggregated_confusion_matrix.png"
                )
                plt.savefig(fname)
                plt.close()
                plotted += 1
    except Exception as e:
        print(f"Error plotting aggregated confusion matrix for {dname}: {e}")
        plt.close()

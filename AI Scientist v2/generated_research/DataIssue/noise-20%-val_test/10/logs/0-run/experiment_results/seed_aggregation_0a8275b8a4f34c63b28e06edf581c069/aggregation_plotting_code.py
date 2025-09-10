import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helper ----------
def macro_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return np.mean(f1s)


# ---------- load data ----------
experiment_data_path_list = [
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_cfbb453d99ee46b98d87e82fd25b45c2_proc_3301991/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_6579d4f81f714644865b182e26aaed54_proc_3301992/experiment_data.npy",
    "experiments/2025-08-17_17-27-17_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_2fe4bd5c069d44e9bdbc0d33da5096e8_proc_3301990/experiment_data.npy",
]
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        loaded = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(loaded)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – nothing to plot.")
else:
    # --------------------  gather aggregates --------------------
    val_curves_by_lr = {}  # {lr: [np.array(seq), ...]}
    train_loss_by_lr = {}
    val_loss_by_lr = {}
    test_f1_by_lr = {}  # {lr: [values]}
    best_lrs = []

    for exp in all_experiment_data:
        spr_exp = exp["learning_rate"]["SPR_BENCH"]
        runs = spr_exp["runs"]
        exp_best_lr = str(spr_exp["best_lr"])
        best_lrs.append(exp_best_lr)

        # test set metrics
        preds = np.array(spr_exp["predictions"])
        gts = np.array(spr_exp["ground_truth"])
        test_macro_f1 = macro_f1(gts, preds)
        test_f1_by_lr.setdefault(exp_best_lr, []).append(test_macro_f1)

        # per-lr curves
        for lr, run in runs.items():
            lr = str(lr)
            val_curves_by_lr.setdefault(lr, []).append(
                np.array(run["metrics"]["val_f1"])
            )
            train_loss_by_lr.setdefault(lr, []).append(np.array(run["losses"]["train"]))
            val_loss_by_lr.setdefault(lr, []).append(np.array(run["losses"]["val"]))

    # --------------------  figure 1 : Val-F1 curves --------------------
    try:
        plt.figure()
        for lr, curves in val_curves_by_lr.items():
            max_len = max(len(c) for c in curves)
            padded = np.stack(
                [
                    np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                    for c in curves
                ]
            )
            mean = np.nanmean(padded, axis=0)
            se = np.nanstd(padded, axis=0, ddof=1) / np.sqrt(len(curves))
            epochs = np.arange(1, max_len + 1)
            plt.plot(epochs, mean, label=f"lr={lr}")
            plt.fill_between(epochs, mean - se, mean + se, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 vs Epoch (mean ± SE)")
        plt.legend(title="Learning rate")
        fname = os.path.join(working_dir, "spr_val_f1_mean_se.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Val-F1 plot: {e}")
        plt.close()

    # --------------------  figure 2 : Loss curves of most frequent best_lr --------------------
    try:
        # choose the mode of best_lrs
        most_common_best_lr = Counter(best_lrs).most_common(1)[0][0]
        tr_curves = train_loss_by_lr.get(most_common_best_lr, [])
        vl_curves = val_loss_by_lr.get(most_common_best_lr, [])
        if tr_curves and vl_curves:
            max_len = max(len(c) for c in tr_curves)
            tr_pad = np.stack(
                [
                    np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                    for c in tr_curves
                ]
            )
            vl_pad = np.stack(
                [
                    np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                    for c in vl_curves
                ]
            )
            tr_mean, tr_se = np.nanmean(tr_pad, 0), np.nanstd(
                tr_pad, 0, ddof=1
            ) / np.sqrt(len(tr_curves))
            vl_mean, vl_se = np.nanmean(vl_pad, 0), np.nanstd(
                vl_pad, 0, ddof=1
            ) / np.sqrt(len(vl_curves))
            epochs = np.arange(1, max_len + 1)
            plt.figure()
            plt.plot(epochs, tr_mean, label="train loss")
            plt.fill_between(epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.2)
            plt.plot(epochs, vl_mean, label="val loss")
            plt.fill_between(epochs, vl_mean - vl_se, vl_mean + vl_se, alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"SPR_BENCH: Loss Curves (best lr={most_common_best_lr}, mean ± SE)"
            )
            plt.legend()
            fname = os.path.join(
                working_dir,
                f"spr_loss_curves_best_lr_{most_common_best_lr}_mean_se.png",
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # --------------------  figure 3 : Test Macro-F1 bar chart --------------------
    try:
        lrs = list(test_f1_by_lr.keys())
        means = np.array([np.mean(test_f1_by_lr[lr]) for lr in lrs])
        ses = np.array(
            [
                np.std(test_f1_by_lr[lr], ddof=1) / np.sqrt(len(test_f1_by_lr[lr]))
                for lr in lrs
            ]
        )

        plt.figure()
        x = np.arange(len(lrs))
        plt.bar(x, means, yerr=ses, alpha=0.7, capsize=5)
        plt.xticks(x, lrs)
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 per learning rate (mean ± SE)")
        plt.xlabel("Learning rate")
        fname = os.path.join(working_dir, "spr_test_macro_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Test Macro-F1 bar plot: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- SETUP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- LOAD MULTI-RUN DATA ----------
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_640a9e2ca0164805b7531bc2d86076b4_proc_3207484/experiment_data.npy",
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_b14be4ad7ace42d98372a94ce7a99372_proc_3207487/experiment_data.npy",
    "experiments/2025-08-17_02-43-40_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_8a426839c5144ace8dd8734645e9fd2e_proc_3207485/experiment_data.npy",
]

all_runs = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        run_data = np.load(full_path, allow_pickle=True).item()
        all_runs.append(run_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs = []

if len(all_runs) == 0:
    print("No experiment data could be loaded. Exiting.")
    exit()


# ---------- EXTRACT AND STACK ----------
def stack_metric(metric_key):
    """Return mean, se per config (list) for the requested metric across runs."""
    per_cfg = []  # list over cfgs, each element: (n_runs, min_epochs)
    n_cfgs = len(all_runs[0]["optimizer_type"]["SPR_BENCH"]["configs"])
    for cfg_idx in range(n_cfgs):
        cfg_runs = []
        for run in all_runs:
            arr = run["optimizer_type"]["SPR_BENCH"]["metrics"][metric_key][cfg_idx]
            cfg_runs.append(np.asarray(arr))
        # truncate to shortest run
        min_len = min(len(a) for a in cfg_runs)
        cfg_runs = np.stack([a[:min_len] for a in cfg_runs])
        per_cfg.append(cfg_runs)
    # compute mean & se
    means = [np.mean(a, axis=0) for a in per_cfg]
    ses = [np.std(a, axis=0, ddof=1) / np.sqrt(a.shape[0]) for a in per_cfg]
    return means, ses, min_len


spr_data_example = all_runs[0]["optimizer_type"]["SPR_BENCH"]
cfgs = list(spr_data_example["configs"])

# ---------- ACCURACY (TRAIN & VAL) ----------
try:
    train_mean, train_se, num_ep = stack_metric("train_acc")
    val_mean, val_se, _ = stack_metric("val_acc")
    epochs = np.arange(1, num_ep + 1)

    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(epochs, train_mean[i], label=f"{cfg}-train")
        plt.fill_between(
            epochs,
            train_mean[i] - train_se[i],
            train_mean[i] + train_se[i],
            alpha=0.2,
        )
        plt.plot(epochs, val_mean[i], "--", label=f"{cfg}-val")
        plt.fill_between(
            epochs,
            val_mean[i] - val_se[i],
            val_mean[i] + val_se[i],
            alpha=0.2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Mean Training vs Validation Accuracy (±SE)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_mean_train_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- LOSS ----------
try:
    train_mean, train_se, num_ep = stack_metric("train_loss")
    val_mean, val_se, _ = stack_metric("val_loss")
    epochs = np.arange(1, num_ep + 1)

    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(epochs, train_mean[i], label=f"{cfg}-train")
        plt.fill_between(
            epochs, train_mean[i] - train_se[i], train_mean[i] + train_se[i], alpha=0.2
        )
        plt.plot(epochs, val_mean[i], "--", label=f"{cfg}-val")
        plt.fill_between(
            epochs, val_mean[i] - val_se[i], val_mean[i] + val_se[i], alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Mean Training vs Validation Loss (±SE)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_mean_train_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- RULE FIDELITY ----------
try:
    fid_mean, fid_se, num_ep = stack_metric("rule_fidelity")
    epochs = np.arange(1, num_ep + 1)

    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(epochs, fid_mean[i], label=f"{cfg}")
        plt.fill_between(
            epochs, fid_mean[i] - fid_se[i], fid_mean[i] + fid_se[i], alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Mean Rule Fidelity per Epoch (±SE)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_mean_rule_fidelity.png"))
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# ---------- GROUND-TRUTH vs PREDICTION DISTRIBUTION ----------
try:
    all_gts, all_preds = [], []
    for run in all_runs:
        ed = run["optimizer_type"]["SPR_BENCH"]
        all_gts.append(ed["ground_truth"])
        all_preds.append(ed["predictions"])
    gts = np.concatenate(all_gts)
    preds = np.concatenate(all_preds)

    classes = np.sort(np.unique(np.concatenate([gts, preds])))
    gt_counts = np.array([np.sum(gts == c) for c in classes])
    pred_counts = np.array([np.sum(preds == c) for c in classes])

    bar_w = 0.4
    x = np.arange(len(classes))
    plt.figure(figsize=(8, 4))
    plt.bar(x - bar_w / 2, gt_counts, width=bar_w, label="Ground Truth")
    plt.bar(x + bar_w / 2, pred_counts, width=bar_w, label="Predicted")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title(
        "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_gt_vs_pred_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating GT vs Pred plot: {e}")
    plt.close()

# ---------- PRINT MEAN TEST ACCURACY ----------
try:
    test_accs = []
    for run in all_runs:
        ed = run["optimizer_type"]["SPR_BENCH"]
        test_accs.append((ed["predictions"] == ed["ground_truth"]).mean())
    print(
        f"Average test accuracy across runs: {np.mean(test_accs):.3f} ± {np.std(test_accs, ddof=1):.3f}"
    )
except Exception as e:
    print(f"Error computing average test accuracy: {e}")

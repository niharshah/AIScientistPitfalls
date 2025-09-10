import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Collect all experiment_data paths supplied by the user
experiment_data_path_list = [
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a7b27483bf964dc69a3758cdbfc8ab67_proc_2967789/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_aa20902070a4493fac6dd445e499fbef_proc_2967786/experiment_data.npy",
    "experiments/2025-08-15_18-22-30_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_902427da2c5146a69fbf8614dc10f711_proc_2967788/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
        else:
            print(f"Warning: file not found {full_path}")
    if not all_experiment_data:
        raise FileNotFoundError("No experiment_data.npy files could be loaded")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------------
def stack_and_crop(arr_list):
    """
    Stack a list of 1-D arrays to shape (runs, min_len) by truncating to min length.
    Returns stacked np.array or None if any array is empty.
    """
    if not arr_list or any(a.size == 0 for a in arr_list):
        return None
    min_len = min(len(a) for a in arr_list)
    if min_len == 0:
        return None
    cropped = np.stack([a[:min_len] for a in arr_list], axis=0)
    return cropped


# ------------------------------------------------------------------
ds_name = "SPR_transformer"
if all_experiment_data:
    # --------------------------------------------------------------
    # Collect per-run arrays
    pre_losses_runs, tr_losses_runs, val_losses_runs = [], [], []
    swa_runs, cwa_runs, scwa_runs = [], [], []
    cm_runs = []  # confusion matrices
    for exp in all_experiment_data:
        if ds_name not in exp:
            print(f"{ds_name} not found in one experiment")
            continue
        data = exp[ds_name]
        pre_losses_runs.append(
            np.asarray(data["losses"].get("pretrain", []), dtype=float)
        )
        tr_losses_runs.append(np.asarray(data["losses"].get("train", []), dtype=float))
        val_losses_runs.append(np.asarray(data["losses"].get("val", []), dtype=float))
        swa_runs.append(np.asarray(data["metrics"].get("val_SWA", []), dtype=float))
        cwa_runs.append(np.asarray(data["metrics"].get("val_CWA", []), dtype=float))
        scwa_runs.append(np.asarray(data["metrics"].get("val_SCWA", []), dtype=float))
        preds = np.asarray(data.get("predictions", []), dtype=int)
        gts = np.asarray(data.get("ground_truth", []), dtype=int)
        if preds.size and gts.size:
            num_lbl = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_lbl, num_lbl), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            cm_runs.append(cm)

    n_runs = len(pre_losses_runs)
    if n_runs == 0:
        print("No runs found containing the requested dataset")
    # --------------------------------------------------------------
    # ------------- Aggregated Pre-training loss -------------------
    try:
        data_mat = stack_and_crop(pre_losses_runs)
        if data_mat is not None:
            mean = data_mat.mean(0)
            se = data_mat.std(0, ddof=1) / np.sqrt(n_runs)
            epochs = np.arange(1, len(mean) + 1)
            plt.figure()
            plt.plot(epochs, mean, label="Mean Loss")
            plt.fill_between(epochs, mean - se, mean + se, alpha=0.3, label="±1 SE")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title(f"{ds_name} Pre-training Loss (Aggregated over {n_runs} runs)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_pretrain_loss.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated pretraining loss plot: {e}")
        plt.close()
    # --------------------------------------------------------------
    # ------------- Aggregated Fine-tune losses --------------------
    try:
        train_mat = stack_and_crop(tr_losses_runs)
        val_mat = stack_and_crop(val_losses_runs)
        if train_mat is not None and val_mat is not None:
            min_len = train_mat.shape[1]
            epochs = np.arange(1, min_len + 1)
            train_mean, train_se = train_mat.mean(0), train_mat.std(
                0, ddof=1
            ) / np.sqrt(n_runs)
            val_mean, val_se = val_mat.mean(0), val_mat.std(0, ddof=1) / np.sqrt(n_runs)
            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean")
            plt.fill_between(
                epochs, train_mean - train_se, train_mean + train_se, alpha=0.3
            )
            plt.plot(epochs, val_mean, label="Val Mean")
            plt.fill_between(epochs, val_mean - val_se, val_mean + val_se, alpha=0.3)
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} Fine-tuning Losses (Mean ± SE, {n_runs} runs)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_finetune_losses.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated fine-tune loss plot: {e}")
        plt.close()
    # --------------------------------------------------------------
    # ------------- Aggregated Validation metrics ------------------
    try:
        swa_mat = stack_and_crop(swa_runs)
        cwa_mat = stack_and_crop(cwa_runs)
        scwa_mat = stack_and_crop(scwa_runs)
        if scwa_mat is not None:
            epochs = np.arange(1, scwa_mat.shape[1] + 1)
            for name, mat, color in [
                ("SWA", swa_mat, "tab:blue"),
                ("CWA", cwa_mat, "tab:green"),
                ("SCWA", scwa_mat, "tab:red"),
            ]:
                if mat is None:
                    continue
                mean, se = mat.mean(0), mat.std(0, ddof=1) / np.sqrt(n_runs)
                plt.figure(1)
                if plt.gcf().number == 1:
                    pass
            # Build plot in one figure
            plt.figure()
            if swa_mat is not None:
                swa_mean, swa_se = swa_mat.mean(0), swa_mat.std(0, ddof=1) / np.sqrt(
                    n_runs
                )
                plt.plot(epochs, swa_mean, label="SWA")
                plt.fill_between(
                    epochs, swa_mean - swa_se, swa_mean + swa_se, alpha=0.2
                )
            if cwa_mat is not None:
                cwa_mean, cwa_se = cwa_mat.mean(0), cwa_mat.std(0, ddof=1) / np.sqrt(
                    n_runs
                )
                plt.plot(epochs, cwa_mean, label="CWA")
                plt.fill_between(
                    epochs, cwa_mean - cwa_se, cwa_mean + cwa_se, alpha=0.2
                )
            if scwa_mat is not None:
                scwa_mean, scwa_se = scwa_mat.mean(0), scwa_mat.std(
                    0, ddof=1
                ) / np.sqrt(n_runs)
                plt.plot(epochs, scwa_mean, label="SCWA")
                plt.fill_between(
                    epochs, scwa_mean - scwa_se, scwa_mean + scwa_se, alpha=0.2
                )
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title(f"{ds_name} Validation Metrics (Mean ± SE, {n_runs} runs)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_val_metrics.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metric plot: {e}")
        plt.close()
    # --------------------------------------------------------------
    # ------------- Aggregated Confusion Matrices ------------------
    try:
        if cm_runs:
            agg_cm = np.sum(cm_runs, axis=0)
            # Downsample to at most 5 confusion matrices across epochs (if multiple lengths)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(agg_cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{ds_name} Aggregated Confusion Matrix\n(Left: Ground Truth, Right: Predicted) – {n_runs} runs summed"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_agg_confusion_matrix.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix plot: {e}")
        plt.close()
    # --------------------------------------------------------------
    # ----------- Print summary of best mean SCWA epoch ------------
    if scwa_mat is not None:
        best_idx = scwa_mat.mean(0).argmax()
        mean_swa, se_swa = swa_mat.mean(0)[best_idx], swa_mat.std(0, ddof=1)[
            best_idx
        ] / np.sqrt(n_runs)
        mean_cwa, se_cwa = cwa_mat.mean(0)[best_idx], cwa_mat.std(0, ddof=1)[
            best_idx
        ] / np.sqrt(n_runs)
        mean_scwa, se_scwa = scwa_mat.mean(0)[best_idx], scwa_mat.std(0, ddof=1)[
            best_idx
        ] / np.sqrt(n_runs)
        print(f"Best epoch (by mean SCWA) = {best_idx+1}")
        print(f"SCWA  = {mean_scwa:.4f} ± {se_scwa:.4f}")
        print(f"SWA   = {mean_swa:.4f} ± {se_swa:.4f}")
        print(f"CWA   = {mean_cwa:.4f} ± {se_cwa:.4f}")

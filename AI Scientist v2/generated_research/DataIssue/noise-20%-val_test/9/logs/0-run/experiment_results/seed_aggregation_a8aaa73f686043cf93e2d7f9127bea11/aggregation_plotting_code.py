import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths & load ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# All experiment files provided in the prompt (relative to $AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_500cbfde893946b0af2d5e5124d82a11_proc_3211630/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_dbd07256943f4aa88286744ade4951d2_proc_3211632/experiment_data.npy",
    "experiments/2025-08-17_02-43-53_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9639da53be0744838f7a198e6ce77643_proc_3211631/experiment_data.npy",
]

all_experiment_data = []
for exp_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_blob = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_blob)
    except Exception as e:
        print(f"Error loading {exp_path}: {e}")


def safe_get(dct, *keys):
    cur = dct
    for k in keys:
        cur = cur.get(k, {})
    return np.array(cur)


# ------------- Collect all dataset names -------------
dataset_names = set()
for run_blob in all_experiment_data:
    dataset_names.update(run_blob.keys())

# ------------- Aggregate & plot -------------
for ds_name in dataset_names:
    # Gather data across runs for this dataset
    runs_for_ds = [
        run_blob.get(ds_name, {})
        for run_blob in all_experiment_data
        if ds_name in run_blob
    ]

    # Helper to stack metric curves (trim to min length)
    def stack_metric(metric_key_tuple):
        curves = []
        for blob in runs_for_ds:
            arr = safe_get(blob.get("metrics", {}), *metric_key_tuple)
            if arr.size:
                curves.append(arr.astype(float))
        if not curves:
            return None  # metric absent in all runs
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        stacked = np.vstack(curves)
        mean = np.nanmean(stacked, axis=0)
        stderr = np.nanstd(stacked, axis=0, ddof=0) / np.sqrt(stacked.shape[0])
        return mean, stderr

    # Accuracy curves ----------------------------------------------------------
    try:
        res = stack_metric(("train_acc",))
        res_val = stack_metric(("val_acc",))
        if res and res_val:
            mean_train, se_train = res
            mean_val, se_val = res_val
            epochs = np.arange(1, len(mean_train) + 1)
            plt.figure()
            plt.plot(epochs, mean_train, label="Train (mean)")
            plt.fill_between(
                epochs,
                mean_train - se_train,
                mean_train + se_train,
                alpha=0.3,
                label="Train ± SE",
            )
            plt.plot(epochs, mean_val, label="Validation (mean)")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.3,
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name}: Aggregated Train vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error aggregating accuracy for {ds_name}: {e}")
        plt.close()

    # Loss curves --------------------------------------------------------------
    try:
        res_loss_tr = stack_metric(("train",))
        res_loss_val = stack_metric(("val",))
        if res_loss_tr and res_loss_val:
            mean_tr, se_tr = res_loss_tr
            mean_val, se_val = res_loss_val
            epochs = np.arange(1, len(mean_tr) + 1)
            plt.figure()
            plt.plot(epochs, mean_tr, label="Train (mean)")
            plt.fill_between(
                epochs, mean_tr - se_tr, mean_tr + se_tr, alpha=0.3, label="Train ± SE"
            )
            plt.plot(epochs, mean_val, label="Validation (mean)")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.3,
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name}: Aggregated Train vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error aggregating loss for {ds_name}: {e}")
        plt.close()

    # Rule Fidelity ------------------------------------------------------------
    try:
        res_rf = stack_metric(("Rule_Fidelity",))
        if res_rf:
            mean_rf, se_rf = res_rf
            epochs = np.arange(1, len(mean_rf) + 1)
            plt.figure()
            plt.plot(epochs, mean_rf, marker="o", label="Mean")
            plt.fill_between(
                epochs, mean_rf - se_rf, mean_rf + se_rf, alpha=0.3, label="± SE"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Rule Fidelity")
            plt.title(f"{ds_name}: Aggregated Rule Fidelity")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_agg_rule_fidelity.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error aggregating rule fidelity for {ds_name}: {e}")
        plt.close()

    # Confusion Matrix ---------------------------------------------------------
    try:
        # Sum confusion matrices over runs
        aggregated_cm = None
        for blob in runs_for_ds:
            preds = np.array(blob.get("predictions", []))
            gts = np.array(blob.get("ground_truth", []))
            if preds.size and gts.size and preds.shape == gts.shape:
                n_cls = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                if aggregated_cm is None:
                    aggregated_cm = cm
                else:
                    # Resize if needed
                    max_dim = max(aggregated_cm.shape[0], cm.shape[0])
                    if aggregated_cm.shape[0] < max_dim:
                        tmp = np.zeros((max_dim, max_dim), int)
                        tmp[: aggregated_cm.shape[0], : aggregated_cm.shape[1]] = (
                            aggregated_cm
                        )
                        aggregated_cm = tmp
                    if cm.shape[0] < max_dim:
                        tmp = np.zeros((max_dim, max_dim), int)
                        tmp[: cm.shape[0], : cm.shape[1]] = cm
                        cm = tmp
                    aggregated_cm += cm
        if aggregated_cm is not None:
            plt.figure(figsize=(6, 5))
            plt.imshow(aggregated_cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{ds_name}: Aggregated Confusion Matrix\n(Left: Ground Truth, Right: Generated Samples)"
            )
            fname = os.path.join(working_dir, f"{ds_name}_agg_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error aggregating confusion matrix for {ds_name}: {e}")
        plt.close()

    # Test Accuracy printout ---------------------------------------------------
    try:
        accs = []
        for blob in runs_for_ds:
            preds = np.array(blob.get("predictions", []))
            gts = np.array(blob.get("ground_truth", []))
            if preds.size and gts.size and preds.shape == gts.shape:
                accs.append((preds == gts).mean())
        if accs:
            accs = np.array(accs, dtype=float)
            print(
                f"{ds_name} Test Accuracy: {accs.mean():.3f} ± {accs.std(ddof=0):.3f}"
            )
    except Exception as e:
        print(f"Error computing aggregated accuracy for {ds_name}: {e}")

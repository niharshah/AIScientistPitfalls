import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# paths to all experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f2c6d103a71c45df83292c18ed88c29a_proc_1437201/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6d076c1f137c489880ea470b400b1924_proc_1437199/experiment_data.npy",
    "experiments/2025-08-30_17-49-45_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_f01e5837e280473984a17a5dd884b26f_proc_1437200/experiment_data.npy",
]

# load all runs
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root, p) if root else p
        run_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(run_data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------------------------------------------------------------------
# collect all datasets that appear in any run
datasets = set()
for run in all_experiment_data:
    datasets.update(run.keys())

# ---------------------------------------------------------------------
for dset in datasets:
    # gather per-run arrays
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    cm_list = []

    for run in all_experiment_data:
        logs = run.get(dset, {})
        if not logs:
            continue

        tl = np.asarray(logs.get("losses", {}).get("train", []))
        vl = np.asarray(logs.get("losses", {}).get("val", []))
        tm = np.asarray(logs.get("metrics", {}).get("train", []))
        vm = np.asarray(logs.get("metrics", {}).get("val", []))

        # keep only runs that have both train & val curves
        if tl.size and vl.size and tm.size and vm.size:
            train_losses.append(tl)
            val_losses.append(vl)
            train_metrics.append(tm)
            val_metrics.append(vm)

        # confusion matrix components
        preds = np.asarray(logs.get("predictions", []))
        gts = np.asarray(logs.get("ground_truth", []))
        if preds.size and gts.size:
            k = max(preds.max(), gts.max()) + 1
            cm = np.zeros((k, k), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            cm_list.append(cm)

    # skip dataset if fewer than 2 runs have useful data
    if len(train_losses) < 2:
        continue

    # align by shortest length
    min_len = min(map(len, train_losses))
    tl_arr = np.stack([tl[:min_len] for tl in train_losses])
    vl_arr = np.stack([vl[:min_len] for vl in val_losses])
    tm_arr = np.stack([tm[:min_len] for tm in train_metrics])
    vm_arr = np.stack([vm[:min_len] for vm in val_metrics])

    epochs = np.arange(1, min_len + 1)

    def mean_se(x):
        mean = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
        return mean, se

    tl_mean, tl_se = mean_se(tl_arr)
    vl_mean, vl_se = mean_se(vl_arr)
    tm_mean, tm_se = mean_se(tm_arr)
    vm_mean, vm_se = mean_se(vm_arr)

    # 1) aggregated loss curve ----------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tl_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs, tl_mean - tl_se, tl_mean + tl_se, alpha=0.3, label="Train SE"
        )
        plt.plot(epochs, vl_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs, vl_mean - vl_se, vl_mean + vl_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{dset}: Aggregated Loss Curve\n(mean ± SE over {tl_arr.shape[0]} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_agg_loss_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved: {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dset}: {e}")
        plt.close()

    # 2) aggregated metric curve --------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tm_mean, label="Train Metric (mean)")
        plt.fill_between(
            epochs, tm_mean - tm_se, tm_mean + tm_se, alpha=0.3, label="Train SE"
        )
        plt.plot(epochs, vm_mean, label="Val Metric (mean)")
        plt.fill_between(
            epochs, vm_mean - vm_se, vm_mean + vm_se, alpha=0.3, label="Val SE"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(
            f"{dset}: Aggregated Metric Curve\n(mean ± SE over {tm_arr.shape[0]} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_agg_metric_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved: {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric curve for {dset}: {e}")
        plt.close()

    # 3) aggregated confusion matrix ----------------------------------
    try:
        if cm_list:
            agg_cm = np.sum(np.stack(cm_list, axis=0), axis=0)
            plt.figure(figsize=(4, 4))
            im = plt.imshow(agg_cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{dset}: Aggregated Confusion Matrix\n(over {len(cm_list)} runs)"
            )
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_agg_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved: {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset}: {e}")
        plt.close()

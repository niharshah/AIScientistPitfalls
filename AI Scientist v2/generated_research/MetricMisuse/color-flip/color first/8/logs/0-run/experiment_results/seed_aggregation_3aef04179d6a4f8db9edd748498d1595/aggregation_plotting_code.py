import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir --------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load every provided experiment_data.npy file
#    (feel free to add/remove paths in the list below)
# -----------------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_9b78000dea1d40889e99e48a9889b112_proc_1723175/experiment_data.npy",
    # add more paths here if they exist …
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def safe_get(exp_dict, *keys, default=None):
    cur = exp_dict
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur else default


def aggregate_epoch_dict(list_of_epoch_value_pairs):
    """
    Converts [(e1,v1), (e2,v2)...] collected from many runs into:
    {epoch1:[v_run1, v_run2,...], epoch2:[...], ...}
    """
    agg = {}
    for pair_list in list_of_epoch_value_pairs:
        for epoch, val in pair_list:
            agg.setdefault(epoch, []).append(val)
    return agg


# -----------------------------------------------------------------------------
# 3. Discover dataset names present in the experiments
# -----------------------------------------------------------------------------
dataset_names = set()
for ed in all_experiment_data:
    dataset_names.update(ed.keys())

# -----------------------------------------------------------------------------
# 4. Iterate over each dataset and create plots
# -----------------------------------------------------------------------------
for ds in dataset_names:

    # =============== AGGREGATE LOSSES ========================================
    try:
        # collect train and val losses from every run
        all_train_pairs = []
        all_val_pairs = []
        for ed in all_experiment_data:
            tr = safe_get(ed, ds, "losses", "train", default=[])
            vl = safe_get(ed, ds, "losses", "val", default=[])
            if tr:
                all_train_pairs.append(tr)
            if vl:
                all_val_pairs.append(vl)

        if all_train_pairs and all_val_pairs:
            train_agg = aggregate_epoch_dict(all_train_pairs)
            val_agg = aggregate_epoch_dict(all_val_pairs)

            # intersect epochs that are present in both train & val
            epochs = sorted(set(train_agg.keys()) & set(val_agg.keys()))
            if epochs:
                tr_mean = [np.mean(train_agg[e]) for e in epochs]
                tr_sem = [
                    np.std(train_agg[e], ddof=1) / np.sqrt(len(train_agg[e]))
                    for e in epochs
                ]
                vl_mean = [np.mean(val_agg[e]) for e in epochs]
                vl_sem = [
                    np.std(val_agg[e], ddof=1) / np.sqrt(len(val_agg[e]))
                    for e in epochs
                ]

                plt.figure()
                plt.plot(epochs, tr_mean, label="Train (mean)")
                plt.fill_between(
                    epochs,
                    np.array(tr_mean) - np.array(tr_sem),
                    np.array(tr_mean) + np.array(tr_sem),
                    alpha=0.2,
                    label="Train ± SEM",
                )
                plt.plot(epochs, vl_mean, label="Validation (mean)")
                plt.fill_between(
                    epochs,
                    np.array(vl_mean) - np.array(vl_sem),
                    np.array(vl_mean) + np.array(vl_sem),
                    alpha=0.2,
                    label="Val ± SEM",
                )

                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{ds}: Aggregated Loss Curves\n(mean ± SEM across runs)")
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_agg_loss_curves.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # =============== AGGREGATE VALIDATION METRICS ============================
    try:
        all_metric_pairs = []  # will hold [(epoch, cwa, swa, cshm)...] per run
        for ed in all_experiment_data:
            mv = safe_get(ed, ds, "metrics", "val", default=[])
            if mv:
                all_metric_pairs.append(mv)

        if all_metric_pairs:
            # split per metric
            metric_names = ["CWA", "SWA", "CSHM"]
            agg_dicts = [dict() for _ in metric_names]  # one dict per metric

            for run_list in all_metric_pairs:
                for tpl in run_list:
                    epoch = tpl[0]
                    for idx in range(len(metric_names)):
                        agg_dicts[idx].setdefault(epoch, []).append(tpl[idx + 1])

            epochs = (
                sorted(set.intersection(*[set(d.keys()) for d in agg_dicts]))
                if agg_dicts
                else []
            )

            if epochs:
                plt.figure()
                for idx, mname in enumerate(metric_names):
                    means = [np.mean(agg_dicts[idx][e]) for e in epochs]
                    sems = [
                        np.std(agg_dicts[idx][e], ddof=1)
                        / np.sqrt(len(agg_dicts[idx][e]))
                        for e in epochs
                    ]
                    plt.plot(epochs, means, label=f"{mname} (mean)")
                    plt.fill_between(
                        epochs,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.2,
                        label=f"{mname} ± SEM",
                    )

                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title(
                    f"{ds}: Aggregated Validation Metrics\n(mean ± SEM across runs)"
                )
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_agg_val_metrics.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot for {ds}: {e}")
        plt.close()

    # =============== CONFUSION MATRIX (aggregated counts) ====================
    try:
        # sum all confusion matrices (requires predictions + ground truth)
        preds_all = []
        gts_all = []
        for ed in all_experiment_data:
            p = np.array(safe_get(ed, ds, "predictions", default=[]))
            g = np.array(safe_get(ed, ds, "ground_truth", default=[]))
            if p.size and g.size:
                preds_all.append(p)
                gts_all.append(g)

        if preds_all and gts_all:
            preds_concat = np.concatenate(preds_all)
            gts_concat = np.concatenate(gts_all)

            n_classes = int(max(preds_concat.max(), gts_concat.max()) + 1)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(gts_concat, preds_concat):
                cm[t, p] += 1

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds}: Aggregated Confusion Matrix\n(Test Set)")
            fname = os.path.join(working_dir, f"{ds}_agg_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()

            test_acc = (preds_concat == gts_concat).mean()
            print(f"{ds} – Aggregated Test Accuracy: {test_acc:.4f}")
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()

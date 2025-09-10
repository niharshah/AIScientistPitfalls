import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from math import sqrt

# -------------------------------------------------------------------------
# basic setup
import matplotlib

matplotlib.use("Agg")  # headless safety
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------------------------------------------------------------
# helper to accumulate values over runs
def aggregate_per_epoch(list_of_ep_val_pairs):
    bucket = defaultdict(list)
    for ep, val in list_of_ep_val_pairs:
        bucket[ep].append(val)
    ep_sorted = sorted(bucket.keys())
    mean = [np.mean(bucket[e]) for e in ep_sorted]
    sem = [
        np.std(bucket[e], ddof=1) / sqrt(len(bucket[e])) if len(bucket[e]) > 1 else 0.0
        for e in ep_sorted
    ]
    return ep_sorted, mean, sem


# -------------------------------------------------------------------------
# load all experiment files
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_521bdf5de089468faed1a388f1bc0456_proc_1705232/experiment_data.npy",
    "experiments/2025-08-31_03-29-17_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_af8331a36967475ba43752b0d3e5dc8a_proc_1705234/experiment_data.npy",
]
all_runs = []
for rel_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(abs_path, allow_pickle=True).item()
        all_runs.append(data)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

# if nothing loaded stop early
if not all_runs:
    print("No experiment data loaded; exiting.")
    quit()

# -------------------------------------------------------------------------
# discover dataset names
dataset_names = set()
for run in all_runs:
    dataset_names.update(run.get("final_state_pool", {}).keys())

# limit confusion matrices to at most 5 datasets
cm_plotted = 0
MAX_CM = 5

# -------------------------------------------------------------------------
# iterate datasets
for dname in dataset_names:
    # containers keyed by epoch
    train_loss_pairs, val_loss_pairs = [], []
    metric_pairs = defaultdict(list)  # key: metric name -> list of (epoch,value)

    # confusion matrix aggregator
    cm_accum = None
    labels_global = set()

    # gather from each run
    for run in all_runs:
        ds = run.get("final_state_pool", {}).get(dname, {})
        if not ds:
            continue
        # losses
        train_loss_pairs.extend(ds.get("losses", {}).get("train", []))
        val_loss_pairs.extend(ds.get("losses", {}).get("val", []))

        # metrics
        for ep, mdict in ds.get("metrics", {}).get("val", []):
            for mname, mval in mdict.items():
                metric_pairs[mname].append((ep, mval))

        # confusion matrix
        preds = np.array(ds.get("predictions", []))
        golds = np.array(ds.get("ground_truth", []))
        if preds.size and golds.size:
            labels = sorted(list(set(golds) | set(preds)))
            labels_global.update(labels)
            lab2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(golds, preds):
                cm[lab2idx[t], lab2idx[p]] += 1
            if cm_accum is None:
                cm_accum = cm
            else:
                # resize if label sets differ
                if cm.shape != cm_accum.shape:
                    max_len = max(cm.shape[0], cm_accum.shape[0])
                    new_cm_acc = np.zeros((max_len, max_len), dtype=int)
                    new_cm_acc[: cm_accum.shape[0], : cm_accum.shape[1]] += cm_accum
                    new_cm = np.zeros((max_len, max_len), dtype=int)
                    new_cm[: cm.shape[0], : cm.shape[1]] += cm
                    cm_accum = new_cm_acc + new_cm
                else:
                    cm_accum += cm

    # ------------------- aggregated loss plot ------------------------------
    try:
        if train_loss_pairs and val_loss_pairs:
            ep_tr, mean_tr, sem_tr = aggregate_per_epoch(train_loss_pairs)
            ep_val, mean_val, sem_val = aggregate_per_epoch(val_loss_pairs)

            plt.figure()
            plt.plot(ep_tr, mean_tr, label="Train Mean", color="tab:blue")
            plt.fill_between(
                ep_tr,
                np.array(mean_tr) - np.array(sem_tr),
                np.array(mean_tr) + np.array(sem_tr),
                color="tab:blue",
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(ep_val, mean_val, label="Validation Mean", color="tab:orange")
            plt.fill_between(
                ep_val,
                np.array(mean_val) - np.array(sem_val),
                np.array(mean_val) + np.array(sem_val),
                color="tab:orange",
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} Aggregated Loss Curves\nMean ± Standard Error")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_loss.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated loss for {dname}: {e}")
        plt.close()

    # ------------------- aggregated metrics plot ---------------------------
    try:
        if metric_pairs:
            plt.figure()
            for mname, ep_val_pairs in metric_pairs.items():
                ep_m, mean_m, sem_m = aggregate_per_epoch(ep_val_pairs)
                plt.plot(ep_m, mean_m, label=f"{mname} Mean")
                plt.fill_between(
                    ep_m,
                    np.array(mean_m) - np.array(sem_m),
                    np.array(mean_m) + np.array(sem_m),
                    alpha=0.25,
                    label=f"{mname} ±SEM",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dname} Aggregated Validation Metrics\nMean ± Standard Error")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_metrics.png")
            plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated metrics for {dname}: {e}")
        plt.close()

    # ------------------- aggregated confusion matrix -----------------------
    if cm_plotted < MAX_CM and cm_accum is not None:
        try:
            labels_sorted = sorted(labels_global)
            lab2idx = {l: i for i, l in enumerate(labels_sorted)}
            # if cm_accum shape smaller, pad
            if cm_accum.shape[0] < len(labels_sorted):
                new_cm = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)
                new_cm[: cm_accum.shape[0], : cm_accum.shape[1]] = cm_accum
                cm_accum = new_cm

            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm_accum, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(np.arange(len(labels_sorted)), labels_sorted, rotation=90)
            plt.yticks(np.arange(len(labels_sorted)), labels_sorted)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} Aggregated Confusion Matrix")
            fname = os.path.join(working_dir, f"{dname}_agg_confusion_matrix.png")
            plt.savefig(fname, bbox_inches="tight")
            cm_plotted += 1
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix for {dname}: {e}")
            plt.close()

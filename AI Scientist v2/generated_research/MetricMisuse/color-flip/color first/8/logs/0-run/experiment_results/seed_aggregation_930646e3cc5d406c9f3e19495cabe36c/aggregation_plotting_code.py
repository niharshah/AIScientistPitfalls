import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------- #
# basic setup
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")  # headless safety

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# load every experiment file that really exists
# --------------------------------------------------------------------------- #
exp_rel_paths = [
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e09ef925eabe430087df354346184dad_proc_1733406/experiment_data.npy",
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_24a0a19feba54159938523ad8b60f21c_proc_1733407/experiment_data.npy",
    "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_a9543301c1e247749dbfb353b990b3b6_proc_1733408/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in exp_rel_paths:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment files found – nothing to plot.")
    exit(0)


# --------------------------------------------------------------------------- #
# helper utilities
# --------------------------------------------------------------------------- #
def aggregate_metric(list_of_epoch_value_pairs):
    """Return sorted epochs, mean values, stderr values across runs."""
    # Build a dict mapping epoch -> list(values)
    epoch_vals = {}
    for ev_pairs in list_of_epoch_value_pairs:
        for ep, val in ev_pairs:
            epoch_vals.setdefault(ep, []).append(val)

    epochs = np.array(sorted(epoch_vals.keys()))
    n_runs = len(list_of_epoch_value_pairs)
    means = []
    stderrs = []
    for ep in epochs:
        vals = np.array(epoch_vals[ep], dtype=float)
        means.append(np.nanmean(vals))
        # stderr over available runs for that epoch (could be < n_runs)
        stderrs.append(np.nanstd(vals) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return epochs, np.array(means), np.array(stderrs)


def collect_runs(ds_key, path_fn):
    """Gather a list of per-run lists [(epoch,val), ...] given a function path_fn(run_dict)."""
    series = []
    for run in all_experiment_data:
        if ds_key in run:
            try:
                s = path_fn(run[ds_key])
                if s:
                    series.append(s)
            except KeyError:
                pass
    return series


# --------------------------------------------------------------------------- #
# iterate over dataset keys found in any run
# --------------------------------------------------------------------------- #
dataset_keys = set()
for run in all_experiment_data:
    dataset_keys.update(run.keys())

for ds_key in dataset_keys:
    # ----------------- Figure 1: aggregated train/val loss ------------------ #
    try:
        train_series = collect_runs(ds_key, lambda d: d["losses"]["train"])
        val_series = collect_runs(ds_key, lambda d: d["losses"]["val"])

        if not train_series and not val_series:
            raise ValueError("No loss curves present.")

        plt.figure()
        if train_series:
            ep, mean, se = aggregate_metric(train_series)
            plt.plot(ep, mean, "--", label="train mean")
            if len(train_series) > 1:
                plt.fill_between(
                    ep, mean - se, mean + se, alpha=0.3, label="train stderr"
                )
        if val_series:
            ep, mean, se = aggregate_metric(val_series)
            plt.plot(ep, mean, "-", label="val mean")
            if len(val_series) > 1:
                plt.fill_between(
                    ep, mean - se, mean + se, alpha=0.3, label="val stderr"
                )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_key}: Aggregated Training / Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_aggregated_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_key}: {e}")
        plt.close()

    # ------------- Figure 2: aggregated validation metrics ----------------- #
    try:
        metric_series = collect_runs(ds_key, lambda d: d["metrics"]["val"])
        if not metric_series:
            raise ValueError("No validation metrics available.")

        # unpack per-run arrays into lists by metric
        keys = ["CWA", "SWA", "HM", "OCGA"]
        per_metric = {k: [] for k in keys}
        for run_metric in metric_series:
            ep, cwa, swa, hm, ocg = zip(*run_metric)
            per_metric["CWA"].append(list(zip(ep, cwa)))
            per_metric["SWA"].append(list(zip(ep, swa)))
            per_metric["HM"].append(list(zip(ep, hm)))
            per_metric["OCGA"].append(list(zip(ep, ocg)))

        plt.figure()
        for m_name, style in zip(keys, ["-", "--", "-.", ":"]):
            series = per_metric[m_name]
            ep, mean, se = aggregate_metric(series)
            plt.plot(ep, mean, style, label=f"{m_name} mean")
            if len(series) > 1:
                plt.fill_between(
                    ep, mean - se, mean + se, alpha=0.25, label=f"{m_name} stderr"
                )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_key}: Aggregated Validation Metrics")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_aggregated_validation_metrics.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot for {ds_key}: {e}")
        plt.close()

    # ------------- Figure 3: aggregated HM with best epoch ----------------- #
    try:
        hm_series = collect_runs(
            ds_key, lambda d: [(e, h) for e, _, _, h, _ in d["metrics"]["val"]]
        )
        if not hm_series:
            raise ValueError("No HM data.")

        ep, mean, se = aggregate_metric(hm_series)
        plt.figure()
        plt.plot(ep, mean, label="HM mean")
        if len(hm_series) > 1:
            plt.fill_between(ep, mean - se, mean + se, alpha=0.3, label="HM stderr")
        best_idx = np.nanargmax(mean)
        plt.scatter(
            ep[best_idx],
            mean[best_idx],
            color="red",
            label=f"Best mean@{int(ep[best_idx])}",
            zorder=5,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Mean (HM)")
        plt.title(f"{ds_key}: Aggregated HM with Best Epoch Marker")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_aggregated_HM.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated HM plot for {ds_key}: {e}")
        plt.close()

    # ------------- Figure 4: aggregated confusion matrix ------------------- #
    try:
        # build summed confusion matrix
        cm_total = None
        for run in all_experiment_data:
            if ds_key not in run:
                continue
            y_true = np.asarray(run[ds_key].get("ground_truth", []))
            y_pred = np.asarray(run[ds_key].get("predictions", []))
            if y_true.size == 0 or y_true.size != y_pred.size:
                continue
            n_cls = int(max(y_true.max(), y_pred.max()) + 1)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            if cm_total is None:
                cm_total = cm
            else:
                # ensure same shape
                max_dim = max(cm_total.shape[0], cm.shape[0])
                if cm_total.shape[0] < max_dim:
                    cm_total = np.pad(
                        cm_total,
                        (
                            (0, max_dim - cm_total.shape[0]),
                            (0, max_dim - cm_total.shape[1]),
                        ),
                        constant_values=0,
                    )
                if cm.shape[0] < max_dim:
                    cm = np.pad(
                        cm,
                        ((0, max_dim - cm.shape[0]), (0, max_dim - cm.shape[1])),
                        constant_values=0,
                    )
                cm_total += cm
        if cm_total is None:
            raise ValueError("No confusion matrices available.")

        plt.figure()
        plt.imshow(cm_total, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{ds_key}: Aggregated Test Confusion Matrix")
        for i in range(cm_total.shape[0]):
            for j in range(cm_total.shape[1]):
                plt.text(
                    j,
                    i,
                    cm_total[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )
        fname = os.path.join(working_dir, f"{ds_key}_aggregated_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_key}: {e}")
        plt.close()

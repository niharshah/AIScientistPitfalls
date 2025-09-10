import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
# basic setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load every experiment file that was provided by the platform
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_6b03c5522dd24a48b19f4b9ba9b94302_proc_1509422/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_3a5e49713ec948dc9d60d7a170d7f363_proc_1509420/experiment_data.npy",
    "experiments/2025-08-30_21-49-55_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_5cc9bc077ced4d29adeee14b085ec225_proc_1509421/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# ------------------------------------------------------------------
# aggregate per dataset name
# ------------------------------------------------------------------
agg = defaultdict(lambda: defaultdict(list))  # dict[dataset][field] -> list

for run in all_experiment_data:
    for dname, d in run.items():
        agg[dname]["epochs"].append(np.asarray(d.get("epochs", [])))
        agg[dname]["tr_loss"].append(np.asarray(d.get("losses", {}).get("train", [])))
        agg[dname]["val_loss"].append(np.asarray(d.get("losses", {}).get("val", [])))
        agg[dname]["tr_metric"].append(
            np.asarray(d.get("metrics", {}).get("train", []))
        )
        agg[dname]["val_metric"].append(np.asarray(d.get("metrics", {}).get("val", [])))
        # store final test predictions / gts for scalar metric aggregation
        agg[dname]["preds"].append(np.asarray(d.get("predictions", [])))
        agg[dname]["gts"].append(np.asarray(d.get("ground_truth", [])))
        agg[dname]["seqs"].append(d.get("seqs", None))


# ------------------------------------------------------------------
# helper for sdwa
# ------------------------------------------------------------------
def sdwa_metric(seqs, y_true, y_pred):
    def _uniq_shapes(seq):
        return len(set(tok[0] for tok in seq.split()))

    def _uniq_colors(seq):
        return len(set(tok[1] for tok in seq.split()))

    weights = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------
plot_count = 0
MAX_PLOTS = 5

for dname, fields in agg.items():
    # --------------------------------------------------------------
    # stack arrays & truncate to shortest common length
    # --------------------------------------------------------------
    def _stack(field_key):
        arrs = [a for a in fields[field_key] if a.size]  # keep non-empty
        if len(arrs) == 0:
            return None
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]
        return np.vstack(arrs)  # shape (runs, epochs)

    tr_loss = _stack("tr_loss")
    val_loss = _stack("val_loss")
    tr_met = _stack("tr_metric")
    val_met = _stack("val_metric")
    epochs_arr = _stack("epochs")
    epochs = (
        epochs_arr[0]
        if epochs_arr is not None
        else np.arange(tr_loss.shape[1] if tr_loss is not None else 0)
    )

    # --------------------------------------------------------------
    # 1) aggregated loss curves
    # --------------------------------------------------------------
    if plot_count < MAX_PLOTS and tr_loss is not None and val_loss is not None:
        try:
            plt.figure()
            for data, lbl, clr in [
                (tr_loss, "Train", "tab:blue"),
                (val_loss, "Validation", "tab:orange"),
            ]:
                mean = data.mean(axis=0)
                se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
                plt.plot(epochs, mean, label=f"{lbl} mean", color=clr)
                plt.fill_between(
                    epochs,
                    mean - se,
                    mean + se,
                    color=clr,
                    alpha=0.3,
                    label=f"{lbl} ±1SE",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dname} – Aggregated Loss Curves\n(Mean ± Standard Error across runs)"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dname.lower()}_aggregate_loss_curve.png"
            )
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {dname}: {e}")
            plt.close()
        plot_count += 1

    # --------------------------------------------------------------
    # 2) aggregated metric curves
    # --------------------------------------------------------------
    if plot_count < MAX_PLOTS and tr_met is not None and val_met is not None:
        try:
            plt.figure()
            for data, lbl, clr in [
                (tr_met, "Train", "tab:green"),
                (val_met, "Validation", "tab:red"),
            ]:
                mean = data.mean(axis=0)
                se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
                plt.plot(epochs, mean, label=f"{lbl} mean", color=clr)
                plt.fill_between(
                    epochs,
                    mean - se,
                    mean + se,
                    color=clr,
                    alpha=0.3,
                    label=f"{lbl} ±1SE",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title(
                f"{dname} – Aggregated Metric Curves\n(Mean ± Standard Error across runs)"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dname.lower()}_aggregate_metric_curve.png"
            )
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated metric plot for {dname}: {e}")
            plt.close()
        plot_count += 1

    # --------------------------------------------------------------
    # 3) scalar test evaluation (accuracy or SDWA) aggregated
    # --------------------------------------------------------------
    try:
        final_metrics = []
        for preds, gts, seqs in zip(fields["preds"], fields["gts"], fields["seqs"]):
            if preds.size == 0 or gts.size == 0:
                continue
            if seqs is not None:
                final_metrics.append(sdwa_metric(seqs, gts, preds))
            else:
                final_metrics.append(np.mean(preds == gts))
        if final_metrics:
            final_metrics = np.asarray(final_metrics)
            print(
                f"{dname} Test Metric Mean ± SE: "
                f"{final_metrics.mean():.4f} ± {final_metrics.std(ddof=1)/np.sqrt(len(final_metrics)):.4f}"
            )
    except Exception as e:
        print(f"Error computing final aggregated test metric for {dname}: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up IO ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load every experiment_data.npy supplied ----------
experiment_data_path_list = [
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e4efbc220eb34242bbb7a2e8dc653f0a_proc_1664579/experiment_data.npy",
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_5897506a3b764ccdb29729ea0f22e75b_proc_1664578/experiment_data.npy",
    "experiments/2025-08-31_03-13-24_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_0a870e1e62b5405d88e4b5b0586e2096_proc_1664577/experiment_data.npy",
]
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------- regroup by dataset ----------
data_by_ds = {}  # ds_name -> dict(metric_name -> list[1d-array])
for exp in all_experiment_data:
    for exp_name, exp_rec in exp.items():
        for ds_name, ds_rec in exp_rec.items():
            store = data_by_ds.setdefault(ds_name, {})
            # losses
            store.setdefault("train_loss", []).append(
                np.asarray(ds_rec["losses"]["train"])
            )
            store.setdefault("val_loss", []).append(np.asarray(ds_rec["losses"]["val"]))
            # metrics (validation only in prompt)
            for m in ds_rec["metrics"]["val"][0].keys():  # peek at keys
                vals = [md[m] for md in ds_rec["metrics"]["val"]]
                store.setdefault(m, []).append(np.asarray(vals))


# ---------- helper for safe plotting ----------
def safe_plot(fn, fname):
    try:
        fn()
        plt.savefig(os.path.join(working_dir, fname), dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"Error creating {fname}: {e}")
    finally:
        plt.close()


# ---------- plotting ----------
for ds_name, metrics_dict in data_by_ds.items():
    n_runs = len(metrics_dict["train_loss"])
    # figure out common epoch length per key
    common_len = {k: min(len(arr) for arr in v) for k, v in metrics_dict.items()}
    # aggregate stats
    agg = {}
    sem = {}
    for k, seq_list in metrics_dict.items():
        arr = np.stack([a[: common_len[k]] for a in seq_list], axis=0)  # (runs, epochs)
        agg[k] = arr.mean(0)
        sem[k] = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])

    epochs = np.arange(1, min(common_len.values()) + 1)

    # ---- 1. Loss curve ----
    def plot_loss():
        plt.figure()
        plt.plot(epochs, agg["train_loss"][: len(epochs)], label="Train mean")
        plt.fill_between(
            epochs,
            agg["train_loss"] - sem["train_loss"],
            agg["train_loss"] + sem["train_loss"],
            alpha=0.2,
        )
        plt.plot(epochs, agg["val_loss"][: len(epochs)], label="Val mean")
        plt.fill_between(
            epochs,
            agg["val_loss"] - sem["val_loss"],
            agg["val_loss"] + sem["val_loss"],
            alpha=0.2,
        )
        plt.title(f"{ds_name} – Aggregate Loss (n={n_runs})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    safe_plot(plot_loss, f"{ds_name}_aggregate_loss.png")

    # helper to generically plot any metric that is present
    def make_metric_plot(metric_key, ylabel):
        def _plt():
            plt.figure()
            plt.plot(epochs, agg[metric_key], label=f"{metric_key} mean")
            plt.fill_between(
                epochs,
                agg[metric_key] - sem[metric_key],
                agg[metric_key] + sem[metric_key],
                alpha=0.2,
                label="± SEM",
            )
            plt.title(f"{ds_name} – Aggregate {ylabel} (n={n_runs})")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()

        safe_plot(_plt, f"{ds_name}_aggregate_{metric_key}.png")

    for metric_key, ylabel in [
        ("acc", "Accuracy"),
        ("CWA", "Color-Weighted Acc"),
        ("SWA", "Shape-Weighted Acc"),
        ("CompWA", "Composite-Weighted Acc"),
    ]:
        if metric_key in agg:
            make_metric_plot(metric_key, ylabel)

    # ---------- console summary ----------
    for metric_key in ["acc", "CWA", "SWA", "CompWA"]:
        if metric_key in agg:
            final_mean = agg[metric_key][-1]
            final_sem = sem[metric_key][-1]
            print(
                f"{ds_name} {metric_key}: {final_mean:.4f} ± {final_sem:.4f} (final epoch, n={n_runs})"
            )

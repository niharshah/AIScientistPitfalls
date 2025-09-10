import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_863792e89e9e451dbfeaecbfe4806c82_proc_1497844/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_89b651abfe2a4cbeb2399c07a6e7da71_proc_1497846/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ---------- helper for CoWA ----------
def colour_of(token: str) -> str:
    return token[1:] if len(token) > 1 else ""


def shape_of(token: str) -> str:
    return token[0]


def complexity_weight(seq: str) -> int:
    return len({colour_of(t) for t in seq.split() if t}) + len(
        {shape_of(t) for t in seq.split() if t}
    )


# ---------- aggregate by dataset ----------
dataset_runs = {}
for exp in all_experiment_data:
    for ds_name, ds_val in exp.items():
        dataset_runs.setdefault(ds_name, []).append(ds_val)

for ds_name, runs in dataset_runs.items():
    # --------- aggregate time-series ----------
    # Align to shortest run length so shapes match
    min_len = min(len(r["losses"]["train"]) for r in runs)
    n_runs = len(runs)

    def stack(metric_getter):
        arr = np.stack([metric_getter(r)[:min_len] for r in runs])
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else None
        return mean, se

    epochs = np.arange(1, min_len + 1)
    train_loss_mean, train_loss_se = stack(lambda r: np.array(r["losses"]["train"]))
    val_loss_mean, val_loss_se = stack(lambda r: np.array(r["losses"]["val"]))
    train_acc_mean, train_acc_se = stack(
        lambda r: np.array([m["acc"] for m in r["metrics"]["train"]])
    )
    val_acc_mean, val_acc_se = stack(
        lambda r: np.array([m["acc"] for m in r["metrics"]["val"]])
    )
    val_cwa_mean, val_cwa_se = stack(
        lambda r: np.array(
            [
                m["CompWA"] if "CompWA" in m else m.get("cowa", np.nan)
                for m in r["metrics"]["val"]
            ]
        )
    )

    # --------- plotting helpers ----------
    def plot_with_band(x, mean, se, label, color):
        plt.plot(x, mean, label=label, color=color)
        if se is not None:
            plt.fill_between(
                x,
                mean - 1.96 * se,
                mean + 1.96 * se,
                color=color,
                alpha=0.25,
                linewidth=0,
                label=f"{label} ±95% CI",
            )

    # 1) Loss curve ----------------------------------------------------
    try:
        plt.figure()
        plot_with_band(epochs, train_loss_mean, train_loss_se, "Train", "tab:blue")
        plot_with_band(epochs, val_loss_mean, val_loss_se, "Validation", "tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Aggregated Loss Curves\nMean ± 95% CI, n={n_runs}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curve -----------------------------------------------
    try:
        plt.figure()
        plot_with_band(epochs, train_acc_mean, train_acc_se, "Train", "tab:blue")
        plot_with_band(epochs, val_acc_mean, val_acc_se, "Validation", "tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Aggregated Accuracy Curves\nMean ± 95% CI, n={n_runs}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) CoWA curve ----------------------------------------------------
    try:
        plt.figure()
        plot_with_band(epochs, val_cwa_mean, val_cwa_se, "Validation CoWA", "tab:green")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{ds_name} Aggregated CoWA Curve\nMean ± 95% CI, n={n_runs}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated CoWA plot for {ds_name}: {e}")
        plt.close()

    # --------- aggregated final test metrics ----------
    acc_list, cowa_list = [], []
    for r in runs:
        preds = np.array(r["predictions"])
        gts = np.array(r["ground_truth"])
        seqs = np.array(r["sequences"])
        weights = np.array([complexity_weight(s) for s in seqs])
        acc_list.append((preds == gts).mean())
        cowa_list.append((weights * (preds == gts)).sum() / weights.sum())

    acc_arr, cowa_arr = np.array(acc_list), np.array(cowa_list)
    print(
        f"{ds_name}: Test Accuracy {acc_arr.mean():.3f} ± {acc_arr.std(ddof=1):.3f} | "
        f"Test CoWA {cowa_arr.mean():.3f} ± {cowa_arr.std(ddof=1):.3f} (n={n_runs})"
    )

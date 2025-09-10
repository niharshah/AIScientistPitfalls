import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------
# Set up directories
# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
# Load all experiment_data dictionaries
# -------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_f40b984e173c4b89bdfea05737198535_proc_1634123/experiment_data.npy",
    "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_24f4c48a4aa54d4f82838a874add8695_proc_1634126/experiment_data.npy",
    "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_658a68acca3e4c4686d480dc8b249cab_proc_1634124/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# Exit early if nothing loaded
if len(all_experiment_data) == 0:
    print("No experiment data could be loaded – exiting.")
else:
    # ---------------------------------------------------------
    # Helper to collect data for SPR_BENCH
    # ---------------------------------------------------------
    dataset_name = "SPR_BENCH"
    ngram_to_acc_curves = {}  # { '1,2': [np.array([...]), ...] }
    ngram_to_loss_curves_train = {}
    ngram_to_loss_curves_val = {}
    test_metric_records = []  # list of dicts with test metrics

    for exp in all_experiment_data:
        try:
            spr = exp["ngram_range_tuning"][dataset_name]
        except KeyError:
            print(f"{dataset_name} not found in one experiment – skipping.")
            continue

        # store test metrics of this experiment
        if spr.get("metrics", {}).get("test"):
            test_metric_records.append(spr["metrics"]["test"])

        # go through each run inside that experiment
        for run in spr["runs"]:
            ngram = run["ngram"]
            # validation accuracy curve
            val_acc = np.array([m["acc"] for m in run["metrics"]["val"]], dtype=float)
            ngram_to_acc_curves.setdefault(ngram, []).append(val_acc)

            # losses
            ngram_to_loss_curves_train.setdefault(ngram, []).append(
                np.array(run["losses"]["train"], dtype=float)
            )
            ngram_to_loss_curves_val.setdefault(ngram, []).append(
                np.array(run["losses"]["val"], dtype=float)
            )

    # ---------------------------------------------------------
    # 1. Aggregated validation accuracy curves (mean ± SEM)
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(7, 4))
        for ngram, curves in ngram_to_acc_curves.items():
            # Align curves to the shortest available length
            min_len = min(len(c) for c in curves)
            data = np.stack([c[:min_len] for c in curves], axis=0)
            mean = data.mean(axis=0)
            sem = (
                data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
                if data.shape[0] > 1
                else np.zeros_like(mean)
            )
            epochs = np.arange(1, min_len + 1)
            plt.plot(epochs, mean, label=f"{ngram} mean")
            plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.25)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH — Validation Accuracy (mean ± SEM)\nLeft: mean lines, shaded: SEM bands"
        )
        plt.legend(title="n-gram")
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation accuracy plot: {e}")
        plt.close()

    # ---------------------------------------------------------
    # 2. Aggregated Train & Val Loss curves (mean ± SEM)
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(7, 4))
        for ngram, curves in ngram_to_loss_curves_train.items():
            min_len = min(len(c) for c in curves)
            train_data = np.stack([c[:min_len] for c in curves], axis=0)
            val_data = np.stack(
                [c[:min_len] for c in ngram_to_loss_curves_val[ngram]], axis=0
            )
            epochs = np.arange(1, min_len + 1)

            train_mean = train_data.mean(axis=0)
            train_sem = (
                train_data.std(axis=0, ddof=1) / np.sqrt(train_data.shape[0])
                if train_data.shape[0] > 1
                else np.zeros_like(train_mean)
            )
            val_mean = val_data.mean(axis=0)
            val_sem = (
                val_data.std(axis=0, ddof=1) / np.sqrt(val_data.shape[0])
                if val_data.shape[0] > 1
                else np.zeros_like(val_mean)
            )

            plt.plot(epochs, train_mean, label=f"{ngram} train mean")
            plt.fill_between(
                epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
            )
            plt.plot(epochs, val_mean, linestyle="--", label=f"{ngram} val mean")
            plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)

        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH — Train (solid) vs Val (dashed) Loss\nMean curves with SEM bands"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_mean_sem.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve plot: {e}")
        plt.close()

    # ---------------------------------------------------------
    # 3. Aggregated Test Metrics (mean ± SEM bars)
    # ---------------------------------------------------------
    try:
        if len(test_metric_records) > 0:
            metric_names = sorted(test_metric_records[0].keys())
            metric_arrays = {m: [] for m in metric_names}
            for rec in test_metric_records:
                for m in metric_names:
                    if m in rec:
                        metric_arrays[m].append(rec[m])

            means = np.array([np.mean(metric_arrays[m]) for m in metric_names])
            sems = np.array(
                [
                    (
                        np.std(metric_arrays[m], ddof=1)
                        / np.sqrt(len(metric_arrays[m]))
                        if len(metric_arrays[m]) > 1
                        else 0.0
                    )
                    for m in metric_names
                ]
            )

            plt.figure(figsize=(6, 4))
            bars = plt.bar(metric_names, means, yerr=sems, capsize=5, color="skyblue")
            plt.ylim(0, 1)
            plt.title("SPR_BENCH — Aggregated Test Metrics (mean ± SEM)")
            for bar, mean, sem in zip(bars, means, sems):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + 0.03,
                    f"{mean:.2f}±{sem:.2f}",
                    ha="center",
                )
            fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_mean_sem.png")
            plt.savefig(fname)
            plt.close()

            # Print aggregated numbers
            print("Aggregated SPR_BENCH Test Metrics (mean ± SEM):")
            for m, mean, se in zip(metric_names, means, sems):
                print(f"{m.upper():6s}: {mean:.3f} ± {se:.3f}")
        else:
            print("No test metrics found across experiments.")
    except Exception as e:
        print(f"Error creating aggregated test metrics plot: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# aggregate experiment data from multiple runs
experiment_data_path_list = [
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_a1bd646e973a490ab38f7be4a1193535_proc_2808978/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_7edad94c3c494d23baa42b3e30f7cfd2_proc_2808980/experiment_data.npy",
    "experiments/2025-08-15_02-34-04_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_6705e6a9051246f9891effaa9d34e8a5_proc_2808981/experiment_data.npy",
]

all_spr_runs = []
for rel_path in experiment_data_path_list:
    try:
        load_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(load_path, allow_pickle=True).item()
        if "spr_bench" in data:
            all_spr_runs.append(data["spr_bench"])
        else:
            print(f"'spr_bench' key not found in {rel_path}")
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

if len(all_spr_runs) == 0:
    print("No runs could be loaded – nothing to plot.")
else:
    n_runs = len(all_spr_runs)

    # ------------------------------------------------------------------
    # stack losses (truncate to min length so shapes match)
    train_loss_min_len = min(len(run["losses"]["train"]) for run in all_spr_runs)
    val_loss_min_len = min(len(run["losses"]["val"]) for run in all_spr_runs)
    loss_epochs = np.arange(1, train_loss_min_len + 1)

    train_losses = np.stack(
        [
            np.asarray(run["losses"]["train"][:train_loss_min_len])
            for run in all_spr_runs
        ]
    )  # [R, E]
    val_losses = np.stack(
        [np.asarray(run["losses"]["val"][:val_loss_min_len]) for run in all_spr_runs]
    )

    train_loss_mean = train_losses.mean(axis=0)
    train_loss_sem = train_losses.std(axis=0, ddof=1) / np.sqrt(n_runs)
    val_loss_mean = val_losses.mean(axis=0)
    val_loss_sem = val_losses.std(axis=0, ddof=1) / np.sqrt(n_runs)

    # ------------------------------------------------------------------
    # stack metrics (truncate to min length)
    train_metric_min_len = min(len(run["metrics"]["train"]) for run in all_spr_runs)
    val_metric_min_len = min(len(run["metrics"]["val"]) for run in all_spr_runs)
    metric_epochs = np.arange(1, train_metric_min_len + 1)

    train_metrics = np.stack(
        [
            np.asarray(run["metrics"]["train"][:train_metric_min_len])
            for run in all_spr_runs
        ]
    )  # [R, E, 3]
    val_metrics = np.stack(
        [np.asarray(run["metrics"]["val"][:val_metric_min_len]) for run in all_spr_runs]
    )

    train_metric_mean = train_metrics.mean(axis=0)  # [E,3]
    train_metric_sem = train_metrics.std(axis=0, ddof=1) / np.sqrt(n_runs)
    val_metric_mean = val_metrics.mean(axis=0)
    val_metric_sem = val_metrics.std(axis=0, ddof=1) / np.sqrt(n_runs)

    # ------------------------------------------------------------------
    # aggregate test metrics
    test_metrics = np.stack(
        [np.asarray(run["metrics"]["test"]) for run in all_spr_runs]
    )  # [R,3]
    test_mean = test_metrics.mean(axis=0)
    test_sem = test_metrics.std(axis=0, ddof=1) / np.sqrt(n_runs)
    print(
        f"Aggregated Test Metrics (mean ± SEM)  ->  "
        f"SWA={test_mean[0]:.4f}±{test_sem[0]:.4f}  "
        f"CWA={test_mean[1]:.4f}±{test_sem[1]:.4f}  "
        f"HWA={test_mean[2]:.4f}±{test_sem[2]:.4f}"
    )

    # ------------------------------------------------------------------
    # Plot 1: Aggregated loss curves
    try:
        plt.figure()
        plt.plot(loss_epochs, train_loss_mean, label="Mean Train Loss", color="blue")
        plt.fill_between(
            loss_epochs,
            train_loss_mean - train_loss_sem,
            train_loss_mean + train_loss_sem,
            color="blue",
            alpha=0.2,
            label="Train SEM",
        )
        plt.plot(loss_epochs, val_loss_mean, label="Mean Val Loss", color="orange")
        plt.fill_between(
            loss_epochs,
            val_loss_mean - val_loss_sem,
            val_loss_mean + val_loss_sem,
            color="orange",
            alpha=0.2,
            label="Val SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Aggregated Loss Curves\nMean ± Standard Error Across Runs")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves_agg.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Aggregated metric curves
    try:
        labels = ["SWA", "CWA", "HWA"]
        colors = ["green", "red", "purple"]
        plt.figure()
        for i, (lab, col) in enumerate(zip(labels, colors)):
            # Train line
            plt.plot(
                metric_epochs,
                train_metric_mean[:, i],
                label=f"Train-{lab} Mean",
                color=col,
            )
            plt.fill_between(
                metric_epochs,
                train_metric_mean[:, i] - train_metric_sem[:, i],
                train_metric_mean[:, i] + train_metric_sem[:, i],
                color=col,
                alpha=0.15,
            )
            # Val line (dashed)
            plt.plot(
                metric_epochs,
                val_metric_mean[:, i],
                label=f"Val-{lab} Mean",
                color=col,
                linestyle="--",
            )
            plt.fill_between(
                metric_epochs,
                val_metric_mean[:, i] - val_metric_sem[:, i],
                val_metric_mean[:, i] + val_metric_sem[:, i],
                color=col,
                alpha=0.15,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Aggregated Accuracy Metrics\n"
            "Solid: Train, Dashed: Val | Shaded: ±SEM"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_metric_curves_agg.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric curve: {e}")
        plt.close()

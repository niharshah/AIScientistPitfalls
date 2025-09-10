import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("num_epochs", {})
    # 1–4: loss curves for each run
    for run_idx, (run_key, run_data) in enumerate(runs.items()):
        try:
            train_losses = run_data["losses"]["train"]
            val_losses = run_data["losses"]["val"]
            epochs = list(range(1, len(train_losses) + 1))

            plt.figure()
            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Synthetic SPR Dataset – Loss Curves\nRun {run_key}")
            plt.legend()
            fname = f"synthetic_spr_loss_curve_{run_key}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {run_key}: {e}")
            plt.close()

    # 5: bar chart comparing final test metrics across runs (max 5 bars * 3 groups)
    try:
        metrics_names = ["CWA", "SWA", "GCWA"]
        x = np.arange(len(runs))  # bar positions
        width = 0.25
        plt.figure()
        for i, m in enumerate(metrics_names):
            vals = [runs[r]["metrics"]["test"][m] for r in runs]
            plt.bar(x + i * width, vals, width, label=m)
        plt.xticks(x + width, list(runs.keys()), rotation=45)
        plt.ylim(0, 1.05)
        plt.ylabel("Weighted Accuracy")
        plt.title(
            "Synthetic SPR Dataset – Test Weighted Accuracies\nLeft: CWA, Middle: SWA, Right: GCWA"
        )
        plt.legend()
        fname = "synthetic_spr_test_metric_comparison.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric comparison plot: {e}")
        plt.close()

    # print evaluation metrics
    print("\nFinal Test Metrics per Run:")
    for run_key, run_data in runs.items():
        tm = run_data["metrics"]["test"]
        print(
            f"{run_key}: CWA={tm['CWA']:.3f}  SWA={tm['SWA']:.3f}  GCWA={tm['GCWA']:.3f}"
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- paths -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
    # Collect test metrics for later aggregated plot
    agg_metrics = {}
    for lr_key, run in runs.items():
        try:
            # --------- loss curves ----------
            train_losses = run["losses"]["train"]
            val_losses = run["losses"]["val"]
            epochs = list(range(1, len(train_losses) + 1))
            plt.figure()
            plt.plot(epochs, train_losses, label="Train")
            plt.plot(epochs, val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f'SPR_BENCH Loss Curves (lr={lr_key.split("_",1)[1]})')
            plt.legend()
            plt.tight_layout()
            fname = f"spr_bench_{lr_key}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curve for {lr_key}: {e}")
            plt.close()

        # store final test metrics for later
        agg_metrics[lr_key] = run["metrics"]["test"]

    # --------- aggregated bar chart -------------
    try:
        metrics_names = ["acc", "swa", "cwa", "nrgs"]
        x = np.arange(len(metrics_names))
        bar_width = 0.2
        plt.figure(figsize=(8, 4))
        for i, (lr_key, mdict) in enumerate(agg_metrics.items()):
            values = [mdict.get(name, 0) for name in metrics_names]
            plt.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=lr_key.split("_", 1)[1],
            )
        plt.ylim(0, 1)
        plt.xticks(
            x + bar_width * (len(agg_metrics) - 1) / 2,
            [n.upper() for n in metrics_names],
        )
        plt.title(
            "SPR_BENCH Test Metrics by Learning Rate\nLeft: Acc  Right: NRGS etc."
        )
        plt.legend(title="Learning Rate")
        plt.tight_layout()
        fname = "spr_bench_lr_comparison_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot: {e}")
        plt.close()

    # -------- print metrics table ---------------
    print("==== Final Test Metrics ====")
    for lr_key, m in agg_metrics.items():
        print(
            lr_key.split("_", 1)[1],
            " | Acc={:.3f} SWA={:.3f} CWA={:.3f} NRGS={:.3f}".format(
                m["acc"], m["swa"], m["cwa"], m["nrgs"]
            ),
        )

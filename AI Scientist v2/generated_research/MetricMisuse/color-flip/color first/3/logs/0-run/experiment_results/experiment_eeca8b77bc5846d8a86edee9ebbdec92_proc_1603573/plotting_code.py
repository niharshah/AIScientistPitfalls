import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# Basic setup                                                         #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data["num_epochs"]["SPR_BENCH"]["runs"]
    # aggregate final metrics for printing / bar plot
    final_dev_hcsa, final_test_hcsa, labels = [], [], []

    # -------------------------------------------------------------- #
    # 1. Loss curves                                                 #
    # -------------------------------------------------------------- #
    try:
        plt.figure(figsize=(6, 4))
        for run in runs:
            epochs, tr_loss = zip(*run["losses"]["train"])
            _, va_loss = zip(*run["losses"]["val"])
            lbl = f"{run['max_epochs']}-epochs"
            plt.plot(epochs, tr_loss, "--", label=f"Train {lbl}")
            plt.plot(epochs, va_loss, "-", label=f"Val   {lbl}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation")
        plt.legend(fontsize=6)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 2. Validation metrics (CWA, SWA, HCSA)                         #
    # -------------------------------------------------------------- #
    try:
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        metric_names = ["CWA", "SWA", "HCSA"]
        for run in runs:
            epochs, cwa, swa, hcs = zip(*run["metrics"]["val"])
            lbl = f"{run['max_epochs']}-epochs"
            axes[0].plot(epochs, cwa, label=lbl)
            axes[1].plot(epochs, swa, label=lbl)
            axes[2].plot(epochs, hcs, label=lbl)
        for ax, m in zip(axes, metric_names):
            ax.set_ylabel(m)
            ax.grid(True, ls=":")
        axes[-1].set_xlabel("Epoch")
        axes[0].set_title("SPR_BENCH Validation Metrics\nCWA, SWA, HCSA over epochs")
        axes[0].legend(fontsize=6)
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 3. Final Dev/Test HCSA bar chart                               #
    # -------------------------------------------------------------- #
    try:
        for run in runs:
            final_dev_hcsa.append(run["metrics"]["val"][-1][3])
            # Test HCSA stored after full training
            # It was saved separately, so pull from run['predictions']['test'] size check
            # We'll recompute quickly from saved lists (lengthy but simple)
            # Harmonic CSA already available in experiment, so use evaluate stats not stored.
            # Use NaN placeholder if not stored.
            test_h = np.nan
            if "ground_truth" in run and "test" in run["ground_truth"]:
                gts = np.array(run["ground_truth"]["test"])
                preds = np.array(run["predictions"]["test"])
                seqs_test = experiment_data["num_epochs"]["SPR_BENCH"].get("seqs_test")
                # seqs_test not persisted; skip precise calc
            test_h = np.nan
            final_test_hcsa.append(test_h)
            labels.append(f"{run['max_epochs']}")
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(x - width / 2, final_dev_hcsa, width, label="Dev HCSA")
        # only plot test bars if numbers exist
        if not np.isnan(final_test_hcsa).all():
            plt.bar(x + width / 2, final_test_hcsa, width, label="Test HCSA")
        plt.xticks(x, labels)
        plt.ylabel("HCSA")
        plt.title("SPR_BENCH Final HCSA Scores\nLeft: Dev, Right: Test")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_final_HCSA.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final HCSA bar plot: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Print numeric summary                                          #
    # -------------------------------------------------------------- #
    print("Max_Epochs | Final Dev HCSA")
    for lbl, dv in zip(labels, final_dev_hcsa):
        print(f"{lbl:>10} | {dv:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_positional"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # -------------------------------------------------
    # 1) Loss curves
    try:
        plt.figure()
        epochs_pre = np.arange(1, len(exp["losses"]["pretrain"]) + 1)
        epochs_ft = np.arange(1, len(exp["losses"]["train"]) + 1)
        # plot
        if epochs_pre.size:
            plt.plot(epochs_pre, exp["losses"]["pretrain"], label="Pretrain Loss")
        plt.plot(epochs_ft, exp["losses"]["train"], label="Train Loss")
        plt.plot(epochs_ft, exp["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Pretrain, Right: Fine-tune")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------------------------------------
    # 2) Metric curves
    try:
        plt.figure()
        epochs = np.arange(1, len(exp["metrics"]["val_SWA"]) + 1)
        plt.plot(epochs, exp["metrics"]["val_SWA"], label="SWA")
        plt.plot(epochs, exp["metrics"]["val_CWA"], label="CWA")
        plt.plot(epochs, exp["metrics"]["val_SCWA"], label="SCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH Validation Metrics\nLeft: SWA, Right: CWA & SCWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # -------------------------------------------------
    # 3) Final metric bar chart
    try:
        plt.figure()
        final_vals = [
            exp["metrics"]["val_SWA"][-1],
            exp["metrics"]["val_CWA"][-1],
            exp["metrics"]["val_SCWA"][-1],
        ]
        names = ["SWA", "CWA", "SCWA"]
        plt.bar(names, final_vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        for i, v in enumerate(final_vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.title(
            "SPR_BENCH Final Validation Metrics\nBar heights show last-epoch values"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_final_metric_bars.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # -------------------------------------------------
    # Print latest metric values
    latest_swa = exp["metrics"]["val_SWA"][-1]
    latest_cwa = exp["metrics"]["val_CWA"][-1]
    latest_scwa = exp["metrics"]["val_SCWA"][-1]
    print(
        f"Latest Validation Metrics -> SWA: {latest_swa:.4f}, "
        f"CWA: {latest_cwa:.4f}, SCWA: {latest_scwa:.4f}"
    )

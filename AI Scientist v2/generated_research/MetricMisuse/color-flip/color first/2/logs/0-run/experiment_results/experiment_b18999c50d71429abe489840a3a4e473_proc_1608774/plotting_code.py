import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]["transformer_baseline"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    val_metrics = exp["metrics"]["val"]  # list of dicts
    epochs = range(1, len(train_loss) + 1)

    # collect metric curves
    cwa_curve = [m.get("CWA", 0) for m in val_metrics]
    swa_curve = [m.get("SWA", 0) for m in val_metrics]
    gcwa_curve = [m.get("GCWA", 0) for m in val_metrics]

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- plot 2: validation metrics ----------
    try:
        plt.figure()
        plt.plot(epochs, cwa_curve, label="CWA")
        plt.plot(epochs, swa_curve, label="SWA")
        plt.plot(epochs, gcwa_curve, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(
            "SPR_BENCH Validation Metrics over Epochs\nLeft: CWA, Center: SWA, Right: GCWA"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # ---------- plot 3: test metrics ----------
    try:
        test_m = exp["metrics"]["test"]
        labels = ["CWA", "SWA", "GCWA"]
        vals = [test_m.get(k, 0) for k in labels]
        plt.figure()
        plt.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics\nLeft: CWA, Center: SWA, Right: GCWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ---------- print final metrics ----------
    print("Final SPR_BENCH test metrics:", exp["metrics"]["test"])
else:
    print("No SPR_BENCH experiment data found.")

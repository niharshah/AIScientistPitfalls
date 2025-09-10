import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

if exp:
    train_loss = exp["losses"].get("train", [])
    val_loss = exp["losses"].get("val", [])
    val_metrics = exp["metrics"].get("val", [])
    test_metrics = exp["metrics"].get("test", {})

    epochs = range(1, 1 + max(len(train_loss), len(val_loss)))

    # ---------- plot 1: loss curves ----------
    try:
        plt.figure()
        if train_loss:
            plt.plot(epochs[: len(train_loss)], train_loss, label="Train")
        if val_loss:
            plt.plot(epochs[: len(val_loss)], val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- plot 2: validation metrics ----------
    try:
        plt.figure()
        cwa = [m["CWA"] for m in val_metrics] if val_metrics else []
        swa = [m["SWA"] for m in val_metrics] if val_metrics else []
        gcw = [m["GCWA"] for m in val_metrics] if val_metrics else []
        if cwa:
            plt.plot(epochs[: len(cwa)], cwa, label="CWA")
        if swa:
            plt.plot(epochs[: len(swa)], swa, label="SWA")
        if gcw:
            plt.plot(epochs[: len(gcw)], gcw, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(
            "SPR_BENCH Validation Metrics Over Epochs\nLeft: CWA, Center: SWA, Right: GCWA"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot: {e}")
        plt.close()

    # ---------- plot 3: test metrics ----------
    try:
        labels = ["CWA", "SWA", "GCWA"]
        vals = [test_metrics.get(l, 0) for l in labels]
        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, vals, color=["steelblue", "orange", "green"])
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Test Metrics\nLeft: CWA, Center: SWA, Right: GCWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()

    # ---------- print metrics ----------
    print("Final test metrics:", test_metrics)
else:
    print("No experiment data found to plot.")

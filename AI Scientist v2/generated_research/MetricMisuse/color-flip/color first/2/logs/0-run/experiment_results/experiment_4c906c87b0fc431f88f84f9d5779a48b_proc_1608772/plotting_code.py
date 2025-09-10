import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = {}

# ---------- plot 1: train / val loss ----------
try:
    train_loss = spr.get("losses", {}).get("train", [])
    val_loss = spr.get("losses", {}).get("val", [])
    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
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
    val_mets = spr.get("metrics", {}).get("val", [])
    if val_mets:
        cwa = [m["CWA"] for m in val_mets]
        swa = [m["SWA"] for m in val_mets]
        gcwa = [m["GCWA"] for m in val_mets]
        epochs = range(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Metrics per Epoch")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# ---------- plot 3: test metrics ----------
try:
    test_m = spr.get("metrics", {}).get("test", {})
    if test_m:
        labels = ["CWA", "SWA", "GCWA"]
        values = [test_m.get(l, 0) for l in labels]
        plt.figure()
        x = np.arange(len(labels))
        plt.bar(x, values, width=0.5)
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test-Set Metrics\nLeft: CWA, Center: SWA, Right: GCWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# ---------- print test metrics ----------
if spr.get("metrics", {}).get("test"):
    print("SPR_BENCH test metrics:", spr["metrics"]["test"])

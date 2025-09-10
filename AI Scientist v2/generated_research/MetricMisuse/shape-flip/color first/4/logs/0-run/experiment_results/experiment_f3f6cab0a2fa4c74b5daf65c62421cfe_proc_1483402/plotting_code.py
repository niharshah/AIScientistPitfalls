import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("no_color_edges", {}).get("SPR", {})

losses_tr = spr_data.get("losses", {}).get("train", [])
losses_val = spr_data.get("losses", {}).get("val", [])
val_mets = spr_data.get("metrics", {}).get("val", [])
test_mets = spr_data.get("metrics", {}).get("test", {})

epochs = range(1, len(losses_tr) + 1)

# ---------------- plot 1: loss curves ----------------
try:
    plt.figure()
    plt.plot(epochs, losses_tr, label="Train")
    plt.plot(epochs, losses_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR Dataset – Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------------- plot 2: validation metrics ----------------
try:
    cwa = [d.get("CWA", np.nan) for d in val_mets]
    swa = [d.get("SWA", np.nan) for d in val_mets]
    cmp = [d.get("CompWA", np.nan) for d in val_mets]

    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cmp, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR Dataset – Validation Metrics Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_val_metrics_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val metrics curve: {e}")
    plt.close()

# ---------------- plot 3: test metrics bar chart ----------------
try:
    labels = ["CWA", "SWA", "CompWA"]
    values = [test_mets.get(k, np.nan) for k in labels]

    plt.figure()
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    plt.title("SPR Dataset – Test Weighted Accuracies")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_test_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar: {e}")
    plt.close()

# ---------------- print test metrics ----------------
print("Test metrics:", test_mets)

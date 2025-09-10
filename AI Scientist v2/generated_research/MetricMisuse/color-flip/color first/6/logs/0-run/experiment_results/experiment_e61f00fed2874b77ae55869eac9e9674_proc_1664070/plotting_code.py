import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# -------------------------------------------------------------------------
# Parse data (only if correctly loaded)
dropouts, train_losses, val_losses, test_metrics, val_cwa = [], {}, {}, {}, {}
if exp:
    runs = exp.get("dropout_rate", {}).get("SPR_BENCH", {})
    for k in sorted(runs.keys(), key=lambda x: float(x.split("=")[1])):
        p = float(k.split("=")[1])
        dropouts.append(p)

        # losses
        train_losses[p] = [l for _, l in runs[k]["losses"]["train"]]
        val_losses[p] = [l for _, l in runs[k]["losses"]["val"]]

        # validation CWA trajectory
        val_cwa[p] = [m["CWA"] for _, m in runs[k]["metrics"]["val"]]

        # test metrics
        test_metrics[p] = runs[k]["metrics"]["test"]

# -------------------------------------------------------------------------
# 1) Combined loss curves
try:
    plt.figure(figsize=(6, 4))
    for p in dropouts:
        epochs = list(range(1, len(train_losses[p]) + 1))
        plt.plot(epochs, train_losses[p], label=f"train p={p}")
        plt.plot(epochs, val_losses[p], linestyle="--", label=f"val p={p}")
    plt.title("SPR_BENCH: Train/Val Loss vs Epoch\n(dropout grid)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) Test metrics vs dropout
try:
    metrics_names = ["CWA", "SWA", "EWA"]
    width = 0.2
    x = np.arange(len(dropouts))
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(metrics_names):
        vals = [test_metrics[p][m] for p in dropouts]
        plt.bar(x + (i - 1) * width, vals, width=width, label=m)
    plt.xticks(x, dropouts)
    plt.xlabel("Dropout Rate")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Metrics vs Dropout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-metric figure: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) Validation CWA curves (first ≤5 dropouts)
try:
    plt.figure(figsize=(6, 4))
    for p in dropouts[:5]:
        epochs = list(range(1, len(val_cwa[p]) + 1))
        plt.plot(epochs, val_cwa[p], label=f"p={p}")
    plt.title("SPR_BENCH: Validation CWA vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_CWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-CWA figure: {e}")
    plt.close()

# -------------------------------------------------------------------------
# Print out collected test metrics
print("=== Test metrics per dropout ===")
for p in dropouts:
    print(f"p={p:.1f} -> {test_metrics[p]}")

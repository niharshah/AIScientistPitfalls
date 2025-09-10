import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("hidden_dim_size", {}).get("SPR_BENCH", {})
hidden_sizes = sorted([int(k) for k in spr_data.keys()])
print(f"Loaded hidden sizes: {hidden_sizes}")

# ----------------- helper: collect arrays -----------------
epochs = None
train_losses, val_losses = {}, {}
val_accs = {}
test_metrics = {"acc": [], "cwa": [], "swa": [], "compwa": []}

for h in hidden_sizes:
    dat = spr_data[str(h)]
    tl = np.array(dat["losses"]["train"])
    vl = np.array(dat["losses"]["val"])
    train_losses[h], val_losses[h] = tl, vl
    if epochs is None:
        epochs = np.arange(1, len(tl) + 1)
    acc_curve = [m["acc"] for m in dat["metrics"]["val"]]
    val_accs[h] = np.array(acc_curve)
    for k in test_metrics:
        test_metrics[k].append(dat["metrics"]["test"][k])

# ----------------- 1) loss curves -----------------
try:
    plt.figure()
    for h in hidden_sizes:
        plt.plot(epochs, train_losses[h], label=f"Train h={h}")
        plt.plot(epochs, val_losses[h], "--", label=f"Val h={h}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR-BENCH: Training vs Validation Loss")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- 2) validation ACC curves -----------------
try:
    plt.figure()
    for h in hidden_sizes:
        plt.plot(epochs, val_accs[h], label=f"h={h}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR-BENCH: Validation ACC across Hidden Sizes")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_val_acc_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating ACC curve plot: {e}")
    plt.close()

# ----------------- 3) test metric vs hidden size -----------------
try:
    plt.figure()
    for metric, vals in test_metrics.items():
        plt.plot(hidden_sizes, vals, marker="o", label=metric.upper())
    plt.xlabel("Hidden Dimension Size")
    plt.ylabel("Score")
    plt.title("SPR-BENCH: Test Metrics vs Hidden Size")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ----------------- echo final test metrics -----------------
print("\nTest metrics by hidden size:")
for i, h in enumerate(hidden_sizes):
    print(
        f"h={h}: "
        f"ACC={test_metrics['acc'][i]:.3f}, "
        f"CWA={test_metrics['cwa'][i]:.3f}, "
        f"SWA={test_metrics['swa'][i]:.3f}, "
        f"CompWA={test_metrics['compwa'][i]:.3f}"
    )

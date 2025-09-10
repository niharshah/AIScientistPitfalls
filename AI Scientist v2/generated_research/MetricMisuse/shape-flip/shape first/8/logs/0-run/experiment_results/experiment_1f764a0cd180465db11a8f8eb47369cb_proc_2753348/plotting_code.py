import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("num_hidden_layers", {}).get("SPR_BENCH", {})
depths = sorted(int(k.split("_")[-1]) for k in runs.keys())
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# helper to fetch per-depth series
def series(key_path, depth_key):
    d = runs[depth_key]
    for k in key_path[:-1]:
        d = d[k]
    return d[key_path[-1]]


# ---------- plot 1: training loss ----------
try:
    plt.figure()
    for i, d in enumerate(depths):
        rk = f"layers_{d}"
        y = series(["losses", "train"], rk)
        plt.plot(range(1, len(y) + 1), y, label=f"{d} hidden", color=colors[i])
    plt.title("SPR_BENCH – Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_by_layers.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- plot 2: validation accuracy ----------
try:
    plt.figure()
    for i, d in enumerate(depths):
        rk = f"layers_{d}"
        y = series(["metrics", "val_acc"], rk)
        plt.plot(range(1, len(y) + 1), y, label=f"{d} hidden", color=colors[i])
    plt.title("SPR_BENCH – Validation Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_by_layers.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy plot: {e}")
    plt.close()

# ---------- plot 3: validation URA ----------
try:
    plt.figure()
    for i, d in enumerate(depths):
        rk = f"layers_{d}"
        y = series(["metrics", "val_ura"], rk)
        plt.plot(range(1, len(y) + 1), y, label=f"{d} hidden", color=colors[i])
    plt.title("SPR_BENCH – Validation URA vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("URA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_URA_by_layers.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation URA plot: {e}")
    plt.close()

# ---------- compute test accuracies ----------
test_accs = {}
for d in depths:
    rk = f"layers_{d}"
    preds = np.array(runs[rk]["predictions"])
    gts = np.array(runs[rk]["ground_truth"])
    test_accs[d] = (preds == gts).mean()

# ---------- plot 4: test accuracy bar ----------
try:
    plt.figure()
    xs = np.arange(len(depths))
    ys = [test_accs[d] for d in depths]
    plt.bar(xs, ys, color=[colors[i] for i in range(len(depths))])
    plt.xticks(xs, [f"{d}" for d in depths])
    plt.title("SPR_BENCH – Final Test Accuracy vs. #Hidden Layers")
    plt.xlabel("# Hidden Layers")
    plt.ylabel("Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
for d in depths:
    print(f"Hidden layers: {d}, Test Accuracy: {test_accs[d]:.3f}")

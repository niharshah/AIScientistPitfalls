import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- data loading ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
runs = experiment_data.get("batch_size_tuning", {}).get(dataset, {})
if not runs:
    print("No data found for plotting")
    exit()

batch_sizes, train_accs, val_accs, train_losses, val_losses, rule_fids, test_accs = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)

# gather metrics
for bs_key, store in sorted(runs.items(), key=lambda x: int(x[0].split("_")[1])):
    bs = int(bs_key.split("_")[1])
    batch_sizes.append(bs)
    train_accs.append(store["metrics"]["train_acc"])
    val_accs.append(store["metrics"]["val_acc"])
    train_losses.append(store["losses"]["train"])
    val_losses.append(store["losses"]["val"])
    rule_fids.append(store["metrics"]["rule_fidelity"])
    test_accs.append(store["test_acc"])

epochs = range(1, len(train_accs[0]) + 1)


# ---------- plotting helpers ----------
def safe_plot(fig_name, plot_fn):
    try:
        plot_fn()
        plt.savefig(os.path.join(working_dir, fig_name))
        plt.close()
    except Exception as e:
        print(f"Error creating {fig_name}: {e}")
        plt.close()


# 1. Accuracy curves
def plot_accuracy():
    plt.figure()
    for bs, t, v in zip(batch_sizes, train_accs, val_accs):
        plt.plot(epochs, t, label=f"train bs={bs}")
        plt.plot(epochs, v, linestyle="--", label=f"val bs={bs}")
    plt.title("Training vs Validation Accuracy\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()


safe_plot("SPR_BENCH_accuracy_curves.png", plot_accuracy)


# 2. Loss curves
def plot_loss():
    plt.figure()
    for bs, t, v in zip(batch_sizes, train_losses, val_losses):
        plt.plot(epochs, t, label=f"train bs={bs}")
        plt.plot(epochs, v, linestyle="--", label=f"val bs={bs}")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()


safe_plot("SPR_BENCH_loss_curves.png", plot_loss)


# 3. Rule fidelity curves
def plot_rule_fid():
    plt.figure()
    for bs, rf in zip(batch_sizes, rule_fids):
        plt.plot(epochs, rf, label=f"bs={bs}")
    plt.title("Rule Fidelity Across Epochs\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.legend()


safe_plot("SPR_BENCH_rule_fidelity.png", plot_rule_fid)


# 4. Test accuracy bar plot
def plot_test_bar():
    plt.figure()
    plt.bar([str(bs) for bs in batch_sizes], test_accs, color="skyblue")
    plt.title("Final Test Accuracy by Batch Size\nDataset: SPR_BENCH")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")


safe_plot("SPR_BENCH_test_accuracy_bar.png", plot_test_bar)


# 5. Scatter best val accuracy vs batch size
def plot_best_val_scatter():
    best_val = [max(v) for v in val_accs]
    plt.figure()
    plt.scatter(batch_sizes, best_val, c="green")
    for bs, acc in zip(batch_sizes, best_val):
        plt.text(bs, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
    plt.title("Best Validation Accuracy vs Batch Size\nDataset: SPR_BENCH")
    plt.xlabel("Batch Size")
    plt.ylabel("Best Validation Accuracy")


safe_plot("SPR_BENCH_best_val_accuracy_scatter.png", plot_best_val_scatter)

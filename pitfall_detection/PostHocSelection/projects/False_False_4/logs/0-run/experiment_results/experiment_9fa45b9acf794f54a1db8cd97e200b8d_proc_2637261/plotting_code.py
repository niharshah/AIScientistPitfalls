import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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

ds = "SPR_BENCH"
if not experiment_data:
    raise SystemExit("No experiment data found.")

data = experiment_data["batch_size"][ds]
metrics = data["metrics"]
losses = data["losses"]
rgs = data["rgs"]

batch_sizes = sorted(map(int, metrics["train_acc"].keys()))
best_bs = data.get("best_batch_size")

# ---------- 1) accuracy vs batch size ----------
try:
    fig, ax = plt.subplots()
    x = np.arange(len(batch_sizes))
    width = 0.25
    ax.bar(
        x - width,
        [metrics["train_acc"][bs] for bs in batch_sizes],
        width,
        label="Train",
    )
    ax.bar(x, [metrics["dev_acc"][bs] for bs in batch_sizes], width, label="Dev")
    test_vals = [metrics["test_acc"].get(bs, np.nan) for bs in batch_sizes]
    ax.bar(x + width, test_vals, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{ds}: Accuracy vs Batch Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds}_accuracy_vs_batchsize.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- 2) RGS vs batch size ----------
try:
    fig, ax = plt.subplots()
    x = np.arange(len(batch_sizes))
    width = 0.3
    ax.bar(
        x - width / 2,
        [rgs["dev"].get(bs, np.nan) for bs in batch_sizes],
        width,
        label="Dev RGS",
    )
    ax.bar(
        x + width / 2,
        [rgs["test"].get(bs, np.nan) for bs in batch_sizes],
        width,
        label="Test RGS",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("RGS")
    ax.set_title(f"{ds}: Rule-Generalisation Score vs Batch Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds}_RGS_vs_batchsize.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RGS plot: {e}")
    plt.close()

# ---------- 3) loss curves ----------
try:
    fig, ax = plt.subplots()
    for bs in batch_sizes:
        tr = losses["train"].get(bs)
        dv = losses["dev"].get(bs)
        if tr is None or dv is None:
            continue
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, label=f"train_bs{bs}", alpha=0.7)
        ax.plot(epochs, dv, linestyle="--", label=f"dev_bs{bs}", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{ds}: Loss Curves (Train & Dev)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- print summary ----------
print(f"Best batch size: {best_bs}")
print(f"Dev accuracy at best bs: {metrics['dev_acc'][best_bs]:.3f}")
test_acc_best = metrics["test_acc"].get(best_bs, np.nan)
print(f"Test accuracy at best bs: {test_acc_best:.3f}")

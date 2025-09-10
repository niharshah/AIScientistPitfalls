import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_logs = experiment_data.get("batch_size_tuning", {})
batch_sizes = sorted(int(k.split("_")[1]) for k in bs_logs.keys())
colors = ["tab:blue", "tab:orange", "tab:green"]  # enough for 3 bsz
test_metrics = []

# ----------------- Plot 1: Accuracy curves -----------------
try:
    plt.figure(figsize=(6, 4))
    for idx, bs in enumerate(batch_sizes):
        logs = bs_logs[f"bs_{bs}"]
        tr_acc = [m["acc"] for m in logs["metrics"]["train"]]
        dv_acc = [m["acc"] for m in logs["metrics"]["dev"]]
        epochs = np.arange(1, len(tr_acc) + 1)
        plt.plot(
            epochs, tr_acc, color=colors[idx], linestyle="-", label=f"Train bs={bs}"
        )
        plt.plot(
            epochs, dv_acc, color=colors[idx], linestyle="--", label=f"Val bs={bs}"
        )
    plt.title("SPR_BENCH (synthetic) – Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ----------------- Plot 2: Test Acc & NRGS -----------------
try:
    plt.figure(figsize=(6, 4))
    width = 0.35
    x = np.arange(len(batch_sizes))
    acc_vals, nrg_vals = [], []
    for bs in batch_sizes:
        logs = bs_logs[f"bs_{bs}"]
        acc_vals.append(logs["metrics"]["test"]["acc"])
        nrg_vals.append(logs["metrics"]["NRGS"])
        test_metrics.append(
            (bs, logs["metrics"]["test"]["acc"], logs["metrics"]["NRGS"])
        )
    plt.bar(x - width / 2, acc_vals, width, label="Test Accuracy")
    plt.bar(x + width / 2, nrg_vals, width, label="NRGS")
    plt.xticks(x, [str(b) for b in batch_sizes])
    plt.title("SPR_BENCH (synthetic) – Test Accuracy vs NRGS by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_acc_NRGS.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ----------------- Plot 3: Loss curves -----------------
try:
    plt.figure(figsize=(6, 4))
    for idx, bs in enumerate(batch_sizes):
        logs = bs_logs[f"bs_{bs}"]
        tr_loss = logs["losses"]["train"]
        dv_loss = logs["losses"]["dev"]
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.plot(
            epochs, tr_loss, color=colors[idx], linestyle="-", label=f"Train bs={bs}"
        )
        plt.plot(
            epochs, dv_loss, color=colors[idx], linestyle="--", label=f"Val bs={bs}"
        )
    plt.title("SPR_BENCH (synthetic) – Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- Print numerical metrics ---------
print("BatchSize | TestAcc | NRGS")
for bs, acc, nrg in test_metrics:
    print(f"{bs:9d} | {acc:.3f}   | {nrg:.3f}")

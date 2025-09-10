import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {})

# ensure float keys are sorted
wds = sorted(spr_data.keys(), key=float)

# containers for final metrics
test_acc = []
test_loss = []

# --- gather per-epoch arrays ---
epochs_dict = {}
for wd in wds:
    run = spr_data[wd]
    metrics = run["metrics"]
    epochs_dict[wd] = {
        "train_acc": np.asarray(metrics["train_acc"]),
        "val_acc": np.asarray(metrics["val_acc"]),
        "train_loss": np.asarray(run["losses"]["train"]),
        "val_loss": np.asarray(run["losses"]["val"]),
        "rule_fid": np.asarray(metrics["rule_fidelity"]),
    }
    test_acc.append(run["test_acc"])
    test_loss.append(run["test_loss"])

# -------- 1) Accuracy curves --------
try:
    plt.figure(figsize=(6, 4))
    for wd in wds:
        ep = np.arange(1, len(epochs_dict[wd]["train_acc"]) + 1)
        plt.plot(ep, epochs_dict[wd]["train_acc"], "--", label=f"train (wd={wd})")
        plt.plot(ep, epochs_dict[wd]["val_acc"], "-", label=f"val (wd={wd})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Training vs Validation Accuracy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_acc_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------- 2) Loss curves --------
try:
    plt.figure(figsize=(6, 4))
    for wd in wds:
        ep = np.arange(1, len(epochs_dict[wd]["train_loss"]) + 1)
        plt.plot(ep, epochs_dict[wd]["train_loss"], "--", label=f"train (wd={wd})")
        plt.plot(ep, epochs_dict[wd]["val_loss"], "-", label=f"val (wd={wd})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- 3) Rule fidelity curves --------
try:
    plt.figure(figsize=(6, 4))
    for wd in wds:
        ep = np.arange(1, len(epochs_dict[wd]["rule_fid"]) + 1)
        plt.plot(ep, epochs_dict[wd]["rule_fid"], label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH – Rule Fidelity Over Epochs")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule-fidelity plot: {e}")
    plt.close()

# -------- 4) Test accuracy bar --------
try:
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(wds)), test_acc, tick_label=wds)
    plt.ylabel("Accuracy")
    plt.xlabel("Weight Decay")
    plt.title("SPR_BENCH – Test Accuracy vs Weight Decay")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy plot: {e}")
    plt.close()

# -------- 5) Test loss bar --------
try:
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(wds)), test_loss, tick_label=wds, color="orange")
    plt.ylabel("Loss")
    plt.xlabel("Weight Decay")
    plt.title("SPR_BENCH – Test Loss vs Weight Decay")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test-loss plot: {e}")
    plt.close()

# --- print summary ---
print("=== SPR_BENCH final metrics ===")
for wd, acc, los in zip(wds, test_acc, test_loss):
    print(f"weight_decay={wd:>8}:  test_acc={acc:.3f}  test_loss={los:.4f}")

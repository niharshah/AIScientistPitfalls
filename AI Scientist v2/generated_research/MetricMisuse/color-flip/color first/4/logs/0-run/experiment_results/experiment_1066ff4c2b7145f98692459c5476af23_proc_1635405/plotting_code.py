import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

wd_dict = experiment_data.get("weight_decay", {})
if not wd_dict:
    print("No weight_decay experiments found in data.")
    exit()

# ---------- gather metrics ----------
best_summary = []  # (wd, best_acc, epoch)
epochs_dict = {}
train_losses, val_losses, val_accs = {}, {}, {}

for key, store in wd_dict.items():
    train_losses[key] = store["losses"]["train"]
    val_losses[key] = store["losses"]["val"]
    acc_list = [m["acc"] for m in store["metrics"]["val"]]
    val_accs[key] = acc_list
    best_epoch = int(np.argmax(acc_list))
    best_summary.append((key, acc_list[best_epoch], best_epoch + 1))
    epochs_dict[key] = list(range(1, len(acc_list) + 1))

# ---------- print summary ----------
print("\nBest validation accuracy per weight decay")
print("{:<12} {:<10} {:<6}".format("weight_decay", "best_acc", "epoch"))
for wd, acc, ep in sorted(best_summary):
    print(f"{wd:<12} {acc:<10.4f} {ep:<6}")

# ---------- plot 1: loss curves ----------
try:
    plt.figure(figsize=(8, 6))
    for key in wd_dict:
        ep = epochs_dict[key]
        plt.plot(ep, train_losses[key], label=f"{key} train")
        plt.plot(ep, val_losses[key], "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "SPR_BENCH – Training vs Validation Loss\n(Left/solid: Train, Right/dashed: Val)"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "sprbench_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- plot 2: validation accuracy ----------
try:
    plt.figure(figsize=(8, 6))
    for key in wd_dict:
        ep = epochs_dict[key]
        plt.plot(ep, val_accs[key], label=f"{key}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH – Validation Accuracy Across Epochs")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "sprbench_val_accuracy.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating val accuracy plot: {e}")
    plt.close()

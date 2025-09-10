import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- load data ---------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# guard
wd_dict = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {})
if not wd_dict:
    print("No SPR_BENCH data found; nothing to plot.")
    exit()

# collect arrays
train_losses, val_losses, val_scwas, test_scwas = {}, {}, {}, {}
epochs = None
for wd, rec in wd_dict.items():
    train_losses[wd] = rec["losses"]["train"]
    val_losses[wd] = rec["losses"]["val"]
    val_scwas[wd] = rec["metrics"]["val"]
    test_scwas[wd] = rec["test_SCWA"]
    if epochs is None:
        epochs = rec["epochs"]

# --------------------------- plots ------------------------------------
# Plot 1: loss curves
try:
    plt.figure(figsize=(10, 4))
    # left subplot: train loss
    plt.subplot(1, 2, 1)
    for wd, l in train_losses.items():
        plt.plot(epochs, l, label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    # right subplot: val loss
    plt.subplot(1, 2, 2)
    for wd, l in val_losses.items():
        plt.plot(epochs, l, label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.suptitle("SPR_BENCH Loss Curves\nLeft: Training Loss, Right: Validation Loss")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves figure: {e}")
    plt.close()

# Plot 2: validation SCWA curves
try:
    plt.figure()
    for wd, sc in val_scwas.items():
        plt.plot(epochs, sc, marker="o", label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.title("SPR_BENCH Validation SCWA vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_scwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SCWA curve: {e}")
    plt.close()

# Plot 3: test SCWA bar chart
try:
    plt.figure()
    wds = list(test_scwas.keys())
    vals = [test_scwas[wd] for wd in wds]
    plt.bar(range(len(wds)), vals, tick_label=wds)
    plt.ylabel("Test SCWA")
    plt.title("SPR_BENCH Test SCWA by Weight Decay")
    plt.savefig(os.path.join(working_dir, "spr_bench_test_scwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test SCWA bar chart: {e}")
    plt.close()

# --------------------------- summary print ----------------------------
print("Final Test SCWA per weight_decay:", test_scwas)

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["batch_size"][
        "SPR_BENCH"
    ]  # dict keyed by batch size as str
    bss = sorted(spr_data.keys(), key=int)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data, bss = {}, []

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# 1) Train / Val loss curves
try:
    plt.figure()
    for i, bs in enumerate(bss):
        epochs = spr_data[bs]["epochs"]
        plt.plot(
            epochs,
            spr_data[bs]["metrics"]["train_loss"],
            "--",
            color=colors[i],
            label=f"train bs={bs}",
        )
        plt.plot(
            epochs,
            spr_data[bs]["metrics"]["val_loss"],
            "-",
            color=colors[i],
            label=f"val bs={bs}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Validation F1 curves
try:
    plt.figure()
    for i, bs in enumerate(bss):
        epochs = spr_data[bs]["epochs"]
        plt.plot(
            epochs, spr_data[bs]["metrics"]["val_f1"], color=colors[i], label=f"bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH: Validation F1 across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val f1 curves: {e}")
    plt.close()

# 3) Test F1 vs Batch Size
try:
    plt.figure()
    test_f1s = [spr_data[bs]["test_f1"] for bs in bss]
    plt.bar(range(len(bss)), test_f1s, color=colors[: len(bss)])
    plt.xticks(range(len(bss)), bss)
    plt.xlabel("Batch Size")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH: Test Macro F1 by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_vs_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test f1 bar chart: {e}")
    plt.close()

# 4) Train vs Val loss at last epoch
try:
    plt.figure()
    train_last = [spr_data[bs]["metrics"]["train_loss"][-1] for bs in bss]
    val_last = [spr_data[bs]["metrics"]["val_loss"][-1] for bs in bss]
    plt.scatter(train_last, val_last, c=colors[: len(bss)])
    for i, bs in enumerate(bss):
        plt.text(train_last[i], val_last[i], bs)
    plt.xlabel("Train Loss (final epoch)")
    plt.ylabel("Val Loss (final epoch)")
    plt.title("SPR_BENCH: Final Epoch Loss Comparison")
    fname = os.path.join(working_dir, "SPR_BENCH_final_loss_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final loss scatter: {e}")
    plt.close()

print(f"Figures saved to {working_dir}")

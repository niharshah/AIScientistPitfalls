import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
#                         load experiment data                       #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_key = ("batch_size_sweep", "SPR_BENCH")
bs_dict = (
    experiment_data.get(spr_key[0], {}).get(spr_key[1], {}) if experiment_data else {}
)

# Short console summary
for bs_name, res in bs_dict.items():
    best_val = max(res["metrics"]["val_MCC"])
    test_mcc = res.get("test_MCC", None)
    print(
        f"{bs_name}: best Val MCC={best_val:.4f}, Test MCC={test_mcc:.4f}"
        if test_mcc is not None
        else f"{bs_name}: data unavailable"
    )

# Convenience list of batch sizes in sorted order
batch_sizes = sorted([int(k.split("_")[-1]) for k in bs_dict.keys()])
bs_names_sorted = [f"bs_{b}" for b in batch_sizes]

# ------------------------------------------------------------------ #
#                           Plot 1: Loss curves                      #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(6, 4))
    for bs_name in bs_names_sorted:
        epochs = bs_dict[bs_name]["epochs"]
        plt.plot(epochs, bs_dict[bs_name]["losses"]["train"], label=f"train, {bs_name}")
        plt.plot(
            epochs,
            bs_dict[bs_name]["losses"]["val"],
            label=f"val,   {bs_name}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (Batch-size sweep)")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
#                          Plot 2: MCC curves                        #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(6, 4))
    for bs_name in bs_names_sorted:
        epochs = bs_dict[bs_name]["epochs"]
        plt.plot(
            epochs, bs_dict[bs_name]["metrics"]["train_MCC"], label=f"train, {bs_name}"
        )
        plt.plot(
            epochs,
            bs_dict[bs_name]["metrics"]["val_MCC"],
            label=f"val,   {bs_name}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Corrcoef")
    plt.title("SPR_BENCH: Training vs Validation MCC (Batch-size sweep)")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
#                       Plot 3: Best val MCC bar                     #
# ------------------------------------------------------------------ #
try:
    best_vals = [max(bs_dict[f"bs_{b}"]["metrics"]["val_MCC"]) for b in batch_sizes]
    plt.figure(figsize=(5, 3))
    plt.bar([str(b) for b in batch_sizes], best_vals)
    plt.xlabel("Batch size")
    plt.ylabel("Best Val MCC")
    plt.title("SPR_BENCH: Best Validation MCC per Batch size")
    fname = os.path.join(working_dir, "SPR_BENCH_best_val_MCC_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating best-val bar plot: {e}")
    plt.close()

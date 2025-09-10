import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup & load -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_runs = experiment_data.get("batch_size_tuning", {})
batch_sizes = sorted(bs_runs.keys())  # ['bs_32', 'bs_64', 'bs_128']


# helper to fetch per-epoch arrays
def get_curves(key1, key2=None):
    curves = {}
    for bs in batch_sizes:
        data = bs_runs[bs]
        if key2 is None:  # losses
            curves[bs] = data[key1]
        else:  # metrics
            curves[bs] = [m[key2] for m in data[key1]["val"]]
    return curves


loss_train = get_curves("losses")["train"] if "train" in get_curves("losses") else {}
loss_val = get_curves("losses")["val"] if "val" in get_curves("losses") else {}
acc_curves = get_curves("metrics", "acc")

# ----------- PLOT 1: Loss curves --------------
try:
    plt.figure()
    for bs in batch_sizes:
        plt.plot(loss_train.get(bs, []), label=f"{bs} – train")
        plt.plot(loss_val.get(bs, []), "--", label=f"{bs} – val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Batch-Size Tuning\nLeft: Train Loss, Right: Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ----------- PLOT 2: Accuracy curves ----------
try:
    plt.figure()
    for bs in batch_sizes:
        plt.plot(acc_curves.get(bs, []), label=bs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation ACC")
    plt.title("SPR_BENCH – Batch-Size Tuning\nValidation Accuracy Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_val_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# ----------- PLOT 3: Final accuracy bar chart ---
try:
    final_acc = {
        bs: (acc_curves[bs][-1] if acc_curves.get(bs) else 0) for bs in batch_sizes
    }
    plt.figure()
    plt.bar(
        range(len(final_acc)),
        list(final_acc.values()),
        tick_label=[bs.split("_")[1] for bs in final_acc.keys()],
    )
    plt.ylabel("Final Val ACC")
    plt.title("SPR_BENCH – Batch-Size Tuning\nFinal Epoch Accuracy by Batch Size")
    fname = os.path.join(working_dir, "spr_bench_final_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()

# ----------- print evaluation metrics ----------
print("Final Validation Accuracy per Batch Size:")
for bs, acc in final_acc.items():
    print(f"{bs}: {acc:.3f}")

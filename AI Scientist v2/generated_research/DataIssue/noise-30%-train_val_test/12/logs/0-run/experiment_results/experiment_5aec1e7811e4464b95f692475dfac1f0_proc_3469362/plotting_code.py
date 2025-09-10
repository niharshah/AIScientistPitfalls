import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---- prepare arrays ----
models = list(experiment_data.keys())
epochs_dict, tr_loss_dict, val_loss_dict, val_f1_dict, best_f1 = {}, {}, {}, {}, {}
for m in models:
    tr_loss = np.array(experiment_data[m]["losses"]["train"])
    val_loss = np.array(experiment_data[m]["losses"]["val"])
    val_f1 = np.array(experiment_data[m]["metrics"]["val"])
    ep = np.arange(1, len(tr_loss) + 1)
    epochs_dict[m] = ep
    tr_loss_dict[m] = tr_loss
    val_loss_dict[m] = val_loss
    val_f1_dict[m] = val_f1
    best_f1[m] = val_f1.max()

# ---- 1: training loss curves ----
try:
    plt.figure()
    for m in models:
        plt.plot(epochs_dict[m], tr_loss_dict[m], label=f"{m} train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_baseline_vs_hybrid.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ---- 2: validation loss curves ----
try:
    plt.figure()
    for m in models:
        plt.plot(epochs_dict[m], val_loss_dict[m], label=f"{m} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Validation Loss vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_baseline_vs_hybrid.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# ---- 3: validation macro-F1 curves ----
try:
    plt.figure()
    for m in models:
        plt.plot(epochs_dict[m], val_f1_dict[m], label=f"{m} val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Validation Macro-F1 vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_f1_baseline_vs_hybrid.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-F1 plot: {e}")
    plt.close()

# ---- 4: best macro-F1 per model ----
try:
    plt.figure()
    names, f1_vals = zip(*sorted(best_f1.items()))
    plt.bar(range(len(names)), f1_vals, tick_label=list(names))
    plt.xlabel("Model")
    plt.ylabel("Best Macro F1")
    plt.title("SPR_BENCH Best Validation Macro-F1 by Model")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_val_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar plot: {e}")
    plt.close()

# ---- numeric summary ----
for m in models:
    final_f1 = val_f1_dict[m][-1] if m in val_f1_dict else None
    print(f"{m:>8}: final val Macro-F1 = {final_f1:.4f} | best = {best_f1[m]:.4f}")

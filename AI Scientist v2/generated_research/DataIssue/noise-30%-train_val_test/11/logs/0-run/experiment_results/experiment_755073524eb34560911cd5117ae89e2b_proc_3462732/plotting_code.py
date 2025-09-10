import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    wd_logs = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    wd_logs = {}

# Helper to sort by numeric value
wd_list = sorted(wd_logs.keys(), key=lambda x: float(x))

# ---------------- figure 1: loss curves -----------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for wd in wd_list:
        epochs = wd_logs[wd]["epochs"]
        ax1.plot(epochs, wd_logs[wd]["losses"]["train"], label=f"wd={wd}")
        ax2.plot(epochs, wd_logs[wd]["losses"]["val"], label=f"wd={wd}")
    ax1.set_title("Training Loss")
    ax2.set_title("Validation Loss")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy")
        ax.legend(fontsize=6)
    plt.suptitle("SPR_BENCH: Left: Train Loss, Right: Val Loss (Weight-Decay Study)")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ---------------- figure 2: F1 curves --------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for wd in wd_list:
        epochs = wd_logs[wd]["epochs"]
        ax1.plot(epochs, wd_logs[wd]["metrics"]["train"], label=f"wd={wd}")
        ax2.plot(epochs, wd_logs[wd]["metrics"]["val"], label=f"wd={wd}")
    ax1.set_title("Training Macro-F1")
    ax2.set_title("Validation Macro-F1")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-F1")
        ax.legend(fontsize=6)
    plt.suptitle("SPR_BENCH: Left: Train F1, Right: Val F1 (Weight-Decay Study)")
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_weight_decay.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve figure: {e}")
    plt.close()

# ---------------- figure 3: test macro-F1 bar chart -----------------------------------
best_wd, best_f1 = None, -1.0
try:
    test_scores = [wd_logs[wd]["test_macro_f1"] for wd in wd_list]
    best_idx = int(np.argmax(test_scores))
    best_wd, best_f1 = wd_list[best_idx], test_scores[best_idx]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(wd_list)), test_scores, tick_label=wd_list)
    plt.ylabel("Test Macro-F1")
    plt.xlabel("Weight Decay")
    plt.title("SPR_BENCH: Test Macro-F1 vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_weight_decay.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar chart: {e}")
    plt.close()

# ---------------- print best configuration --------------------------------------------
if best_wd is not None:
    print(f"Best weight_decay: {best_wd}  |  Test Macro-F1: {best_f1:.4f}")

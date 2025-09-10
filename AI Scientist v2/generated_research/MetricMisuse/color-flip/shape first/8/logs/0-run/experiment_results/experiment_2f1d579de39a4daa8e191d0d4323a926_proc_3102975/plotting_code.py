import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

plot_paths = []

# ----------------------- load data ----------------------------
try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = ed["mask_only"]["SPR_BENCH"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    val_ccwa = ed["metrics"]["val_CCWA"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed, train_loss, val_loss, val_ccwa = None, None, None, None

# ------------------ plot loss curves --------------------------
try:
    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (Mask-Only Views Ablation)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plot_paths.append(fname)
    else:
        print("Loss data missing; skipping loss curve plot.")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
finally:
    plt.close()

# ------------------ plot CCWA metric --------------------------
try:
    if val_ccwa:
        epochs = range(1, len(val_ccwa) + 1)
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA Score")
        plt.title("SPR_BENCH Validation CCWA Across Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png")
        plt.savefig(fname)
        plot_paths.append(fname)
    else:
        print("CCWA data missing; skipping CCWA plot.")
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
finally:
    plt.close()

print("Plots saved:", plot_paths)

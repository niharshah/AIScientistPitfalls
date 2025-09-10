import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["dropout_sweep"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    dropouts = ed["dropouts"]
    epochs_list = ed["epochs"]
    train_f1 = ed["metrics"]["train_f1"]
    val_f1 = ed["metrics"]["val_f1"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]

    # helper to compute best val f1 per run
    best_val_f1 = [max(v) for v in val_f1]

    # 1. Train F1 curves
    try:
        plt.figure()
        for d, ep, f1s in zip(dropouts, epochs_list, train_f1):
            plt.plot(ep, f1s, label=f"dropout={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Train F1")
        plt.title("SPR_BENCH: Training F1 vs Epoch\nLeft: Different Dropouts")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_F1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train F1 plot: {e}")
        plt.close()

    # 2. Val F1 curves
    try:
        plt.figure()
        for d, ep, f1s in zip(dropouts, epochs_list, val_f1):
            plt.plot(ep, f1s, label=f"dropout={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1")
        plt.title("SPR_BENCH: Validation F1 vs Epoch\nLeft: Different Dropouts")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_F1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val F1 plot: {e}")
        plt.close()

    # 3. Train loss curves
    try:
        plt.figure()
        for d, ep, losses in zip(dropouts, epochs_list, train_loss):
            plt.plot(ep, losses, label=f"dropout={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("SPR_BENCH: Training Loss vs Epoch\nLeft: Different Dropouts")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train loss plot: {e}")
        plt.close()

    # 4. Val loss curves
    try:
        plt.figure()
        for d, ep, losses in zip(dropouts, epochs_list, val_loss):
            plt.plot(ep, losses, label=f"dropout={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH: Validation Loss vs Epoch\nLeft: Different Dropouts")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val loss plot: {e}")
        plt.close()

    # 5. Best val F1 vs dropout
    try:
        plt.figure()
        plt.scatter(dropouts, best_val_f1, c="red")
        for d, f in zip(dropouts, best_val_f1):
            plt.text(d, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
        plt.xlabel("Dropout Probability")
        plt.ylabel("Best Validation F1")
        plt.title("SPR_BENCH: Best Validation F1 per Dropout\nRight: Summary Scatter")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_val_F1_vs_dropout.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating summary scatter: {e}")
        plt.close()

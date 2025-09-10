import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "learning_rate" in experiment_data:

    lr_entries = experiment_data["learning_rate"]
    # Collect meta-info once to avoid repetition
    lrs, train_losses, val_losses, val_f1s, test_f1s, epochs = [], [], [], [], [], []
    for key, entry in lr_entries.items():
        lrs.append(entry["lr"])
        train_losses.append(entry["metrics"]["train_loss"])
        val_losses.append(entry["metrics"]["val_loss"])
        val_f1s.append(entry["metrics"]["val_f1"])
        test_f1s.append(entry.get("test_f1", np.nan))
        epochs.append(entry["epochs"])

    # ---------- 1) Loss curves ----------
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            ep = epochs[i]
            plt.plot(ep, train_losses[i], "--", label=f"Train lr={lr:.0e}")
            plt.plot(ep, val_losses[i], "-", label=f"Val lr={lr:.0e}")
        plt.title("Training & Validation Loss vs Epoch\nDataset: SPR-Bench")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR-Bench_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 2) Validation F1 curves ----------
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            plt.plot(epochs[i], val_f1s[i], label=f"lr={lr:.0e}")
        plt.title("Validation Macro-F1 vs Epoch\nDataset: SPR-Bench")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend(fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR-Bench_valF1_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating val F1 curves: {e}")
        plt.close()

    # ---------- 3) Test F1 bar chart ----------
    try:
        plt.figure()
        x_pos = np.arange(len(lrs))
        plt.bar(x_pos, test_f1s, color="skyblue")
        plt.xticks(x_pos, [f"{lr:.0e}" for lr in lrs])
        plt.ylabel("Macro-F1")
        plt.title("Test Macro-F1 across Learning Rates\nDataset: SPR-Bench")
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR-Bench_testF1_bar.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar: {e}")
        plt.close()

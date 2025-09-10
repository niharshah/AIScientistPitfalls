import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths / load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------- helper to print summary ----------
def summarise_runs(runs):
    header = f"{'lr':>8} | {'best_val_f1':>12} | {'test_f1':>8}"
    print(header)
    print("-" * len(header))
    for r in runs:
        best_val = (
            max(r["metrics"]["val_f1"]) if r["metrics"]["val_f1"] else float("nan")
        )
        print(f"{r['lr']:8.4g} | {best_val:12.4f} | {r['test_f1']:8.4f}")


# -------- iterate datasets ----------
for ds_name, lr_dict in experiment_data.items():
    runs = lr_dict.get(ds_name, []) if isinstance(lr_dict, dict) else lr_dict
    if not runs:
        continue
    summarise_runs(runs)

    # --- per-lr plots (<=5) ---
    for r in runs[:5]:  # ensure at most 5
        lr = r["lr"]
        epochs = r["epochs"]
        # F1 curve
        try:
            plt.figure()
            plt.plot(epochs, r["metrics"]["train_f1"], label="Train F1")
            plt.plot(epochs, r["metrics"]["val_f1"], label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{ds_name} F1 vs Epoch (lr={lr})")
            plt.legend()
            fname = f"{ds_name}_f1_curve_lr{lr}.png".replace(".", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot for lr={lr}: {e}")
            plt.close()

        # Loss curve
        try:
            plt.figure()
            plt.plot(epochs, r["losses"]["train"], label="Train Loss")
            plt.plot(epochs, r["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} Loss vs Epoch (lr={lr})")
            plt.legend()
            fname = f"{ds_name}_loss_curve_lr{lr}.png".replace(".", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating Loss plot for lr={lr}: {e}")
            plt.close()

    # --- aggregated test-F1 bar plot ---
    try:
        plt.figure()
        lrs = [r["lr"] for r in runs]
        testF = [r["test_f1"] for r in runs]
        plt.bar(range(len(lrs)), testF, tick_label=[str(lr) for lr in lrs])
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Macro-F1")
        plt.title(f"{ds_name} Test F1 across Learning Rates")
        fname = f"{ds_name}_testF1_vs_lr.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated bar plot: {e}")
        plt.close()

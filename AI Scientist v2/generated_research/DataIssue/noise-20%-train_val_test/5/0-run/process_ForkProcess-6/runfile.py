import matplotlib.pyplot as plt
import numpy as np
import os

# --- set up paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

best_run_id, best_val_acc = None, -1
if experiment_data:
    runs = experiment_data["num_epochs"]["SPR_BENCH"]["runs"]
    for idx, run in enumerate(runs):
        epochs = run["epochs"]
        # ------------- accuracy curve ---------------
        try:
            plt.figure()
            plt.plot(epochs, run["train_acc"], label="Train Acc")
            plt.plot(epochs, run["val_acc"], label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f'SPR_BENCH Acc Curves (epochs={run["setting"]})')
            plt.legend()
            fname = f'train_val_accuracy_SPR_BENCH_epochs{run["setting"]}.png'
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for run {idx}: {e}")
            plt.close()
        # ------------- loss curve -------------------
        try:
            plt.figure()
            plt.plot(epochs, run["train_loss"], label="Train Loss")
            plt.plot(epochs, run["val_loss"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f'SPR_BENCH Loss Curves (epochs={run["setting"]})')
            plt.legend()
            fname = f'train_val_loss_SPR_BENCH_epochs{run["setting"]}.png'
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for run {idx}: {e}")
            plt.close()
        # ------------- select best run --------------
        if run["val_acc"][-1] > best_val_acc:
            best_val_acc = run["val_acc"][-1]
            best_run_id = idx

    # --- print summary metrics ---
    print("\n=== Summary of Runs ===")
    for idx, run in enumerate(runs):
        print(
            f"Run {idx}: epochs={run['setting']}, "
            f"train_acc={run['train_acc'][-1]:.3f}, "
            f"val_acc={run['val_acc'][-1]:.3f}, "
            f"test_acc={run['test_acc']:.3f}"
        )
    if best_run_id is not None:
        print(
            f"\nBest run: {best_run_id} (epochs={runs[best_run_id]['setting']}) "
            f"with val_acc={best_val_acc:.3f} and test_acc={runs[best_run_id]['test_acc']:.3f}"
        )
else:
    print("No experiment data available.")

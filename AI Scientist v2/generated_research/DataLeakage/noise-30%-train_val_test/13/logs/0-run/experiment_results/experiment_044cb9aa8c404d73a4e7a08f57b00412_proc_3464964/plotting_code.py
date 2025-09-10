import matplotlib.pyplot as plt
import numpy as np
import os

# ----- set up working dir -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("nhead_tuning", {}).get("SPR_BENCH", {})
    if not runs:
        print("No SPR_BENCH runs found inside experiment_data.")
    else:
        # compute best dev F1 & gather test scores
        best_nhead, best_dev_f1, best_test_f1 = None, -1.0, -1.0
        test_scores = {}
        for key, run in runs.items():
            dev_f1 = run["metrics"]["val_f1"][-1]
            test_f1 = run["test_f1"]
            test_scores[key] = test_f1
            if dev_f1 > best_dev_f1:
                best_dev_f1, best_nhead, best_test_f1 = dev_f1, key, test_f1

        # ----------- PLOT 1: F1 curves -----------
        try:
            plt.figure()
            for key, run in runs.items():
                epochs = run["epochs"]
                plt.plot(
                    epochs,
                    run["metrics"]["train_f1"],
                    label=f"{key} train",
                    linestyle="--",
                )
                plt.plot(epochs, run["metrics"]["val_f1"], label=f"{key} val")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title("SPR_BENCH: Train & Val Macro-F1 vs Epochs (nhead tuning)")
            plt.legend()
            save_path = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating F1 curve plot: {e}")
            plt.close()

        # ----------- PLOT 2: Loss curves -----------
        try:
            plt.figure()
            for key, run in runs.items():
                epochs = run["epochs"]
                plt.plot(
                    epochs, run["losses"]["train"], label=f"{key} train", linestyle="--"
                )
                plt.plot(epochs, run["losses"]["val"], label=f"{key} val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH: Train & Val Loss vs Epochs (nhead tuning)")
            plt.legend()
            save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve plot: {e}")
            plt.close()

        # ----------- PLOT 3: Test macro-F1 summary -----------
        try:
            plt.figure()
            labels, values = zip(*test_scores.items())
            plt.bar(range(len(values)), values, tick_label=labels)
            plt.ylabel("Test Macro-F1")
            plt.title("SPR_BENCH: Test Macro-F1 by nhead configuration")
            plt.ylim(0, 1)
            save_path = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating test F1 bar plot: {e}")
            plt.close()

        # ----------- print evaluation metrics -----------
        print("Test Macro-F1 per configuration:")
        for k, v in test_scores.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nBest configuration based on dev Macro-F1: {best_nhead}")
        print(f"  Dev Macro-F1:  {best_dev_f1:.4f}")
        print(f"  Test Macro-F1: {best_test_f1:.4f}")

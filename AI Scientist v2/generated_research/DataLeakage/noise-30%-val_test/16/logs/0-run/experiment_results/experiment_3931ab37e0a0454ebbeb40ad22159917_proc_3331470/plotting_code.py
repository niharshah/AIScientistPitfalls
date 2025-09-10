import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data["learning_rate"]["SPR_BENCH"]
    # Identify best run by final val_MCC
    best_key, best_val = None, -1
    for k, v in runs.items():
        mcc = v["metrics"]["val_MCC"][-1] if v["metrics"]["val_MCC"] else -1
        if mcc > best_val:
            best_val, best_key = mcc, k
    print(f"Best run: {best_key} with final val_MCC={best_val:.4f}")

    # 1) Combined val loss curves
    try:
        plt.figure()
        for k, v in runs.items():
            plt.plot(v["epochs"], v["losses"]["val"], label=f"lr={k}")
        plt.title("SPR_BENCH: Validation Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_combined_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating combined val loss plot: {e}")
        plt.close()

    # 2) Combined val MCC curves
    try:
        plt.figure()
        for k, v in runs.items():
            plt.plot(v["epochs"], v["metrics"]["val_MCC"], label=f"lr={k}")
        plt.title("SPR_BENCH: Validation MCC vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_combined_val_MCC.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating combined val MCC plot: {e}")
        plt.close()

    # 3) Best run train/val loss
    try:
        best = runs[best_key]
        plt.figure()
        plt.plot(best["epochs"], best["losses"]["train"], label="Train")
        plt.plot(best["epochs"], best["losses"]["val"], label="Val")
        plt.title(f"SPR_BENCH: Loss Curves (Best lr={best_key})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_best_lr_{best_key}_loss_curves.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating best run loss plot: {e}")
        plt.close()

    # 4) Best run train/val MCC
    try:
        plt.figure()
        plt.plot(best["epochs"], best["metrics"]["train_MCC"], label="Train MCC")
        plt.plot(best["epochs"], best["metrics"]["val_MCC"], label="Val MCC")
        plt.title(f"SPR_BENCH: MCC Curves (Best lr={best_key})")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_best_lr_{best_key}_MCC_curves.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating best run MCC plot: {e}")
        plt.close()

    # 5) Bar chart of test MCC
    try:
        lrs, test_mccs = [], []
        for k, v in runs.items():
            lrs.append(k)
            test_mccs.append(v["metrics"]["test_MCC"])
        plt.figure()
        plt.bar(lrs, test_mccs)
        plt.title("SPR_BENCH: Test MCC by Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Test MCC")
        fname = os.path.join(working_dir, "SPR_BENCH_test_MCC_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test MCC bar plot: {e}")
        plt.close()

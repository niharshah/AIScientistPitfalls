import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["hidden_size"]["SPR_BENCH"]
    hs = np.array(ed["hidden_sizes"])
    tr_acc = np.array(ed["metrics"]["train_acc"])
    val_acc = np.array(ed["metrics"]["val_acc"])
    val_ura = np.array(ed["metrics"]["val_ura"])
    best_test_acc = ed["metrics"]["test_acc"][0] if ed["metrics"]["test_acc"] else None
    best_test_ura = ed["metrics"]["test_ura"][0] if ed["metrics"]["test_ura"] else None
    best_hid = hs[np.argmax(val_acc)] if len(hs) else None

    print(f"Best hidden size: {best_hid}")
    print(f"Test Accuracy   : {best_test_acc}")
    print(f"Test URA        : {best_test_ura}")

    # ---------- PLOT 1: Training accuracy ----------
    try:
        plt.figure()
        plt.plot(hs, tr_acc, marker="o")
        plt.title("SPR_BENCH – Training Accuracy vs. Hidden Size")
        plt.xlabel("Hidden Size")
        plt.ylabel("Training Accuracy")
        plt.grid(True)
        fname = os.path.join(working_dir, "SPR_BENCH_train_acc_vs_hidden.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating training accuracy plot: {e}")
        plt.close()

    # ---------- PLOT 2: Validation accuracy ----------
    try:
        plt.figure()
        plt.plot(hs, val_acc, marker="s", color="orange")
        plt.title("SPR_BENCH – Validation Accuracy vs. Hidden Size")
        plt.xlabel("Hidden Size")
        plt.ylabel("Validation Accuracy")
        plt.grid(True)
        fname = os.path.join(working_dir, "SPR_BENCH_val_acc_vs_hidden.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation accuracy plot: {e}")
        plt.close()

    # ---------- PLOT 3: Validation URA ----------
    try:
        plt.figure()
        plt.plot(hs, val_ura, marker="^", color="green")
        plt.title("SPR_BENCH – Validation URA vs. Hidden Size")
        plt.xlabel("Hidden Size")
        plt.ylabel("Validation URA")
        plt.grid(True)
        fname = os.path.join(working_dir, "SPR_BENCH_val_ura_vs_hidden.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation URA plot: {e}")
        plt.close()

    # ---------- PLOT 4: Best model test metrics ----------
    try:
        if best_test_acc is not None and best_test_ura is not None:
            plt.figure()
            plt.bar(
                ["Accuracy", "URA"],
                [best_test_acc, best_test_ura],
                color=["steelblue", "salmon"],
            )
            plt.ylim(0, 1)
            plt.title(f"SPR_BENCH – Test Metrics (Best Hidden Size = {best_hid})")
            for i, v in enumerate([best_test_acc, best_test_ura]):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
            fname = os.path.join(working_dir, "SPR_BENCH_best_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating best test metrics plot: {e}")
        plt.close()

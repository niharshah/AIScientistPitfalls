import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    lr_vals = ed["lr_values"]
    tr_acc_hist = ed["metrics"]["train_acc"]  # list[list]
    val_acc_hist = ed["metrics"]["val_acc"]  # list[list]
    tr_loss_hist = ed["losses"]["train"]  # list[list]
    val_loss_hist = ed["losses"]["val"]  # list[list]
    final_val_acc = [v[-1] for v in val_acc_hist] if val_acc_hist else []
    test_metrics = {
        "Overall Acc": (
            np.mean(
                np.array(ed.get("predictions", []))
                == np.array(ed.get("ground_truth", []))
            )
            if ed.get("predictions")
            else None
        ),
        "SWA": ed.get("ZSRTA", [None])[-1] if ed.get("ZSRTA") else None,
        "CWA": None,  # CWA stored per-test; not replicated here
        "ZSRTA": ed.get("ZSRTA", [None])[-1] if ed.get("ZSRTA") else None,
    }

    # --------------------------------------------------------------
    # 1) Accuracy curves
    try:
        plt.figure()
        for lr, tr, va in zip(lr_vals, tr_acc_hist, val_acc_hist):
            epochs = np.arange(1, len(tr) + 1)
            plt.plot(epochs, tr, "--", label=f"train lr={lr}")
            plt.plot(epochs, va, "-", label=f"val lr={lr}")
        plt.title("SPR_BENCH: Train vs Validation Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 2) Loss curves
    try:
        plt.figure()
        for lr, tr, va in zip(lr_vals, tr_loss_hist, val_loss_hist):
            epochs = np.arange(1, len(tr) + 1)
            plt.plot(epochs, tr, "--", label=f"train lr={lr}")
            plt.plot(epochs, va, "-", label=f"val lr={lr}")
        plt.title("SPR_BENCH: Train vs Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 3) Final validation accuracy vs learning rate
    try:
        plt.figure()
        plt.plot(lr_vals, final_val_acc, "o-")
        plt.xscale("log")
        plt.title("SPR_BENCH: Final Validation Accuracy vs Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Validation Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_val_acc_vs_lr.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating val-acc-vs-lr plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 4) Test metrics bar chart (if metrics exist)
    try:
        metrics = {k: v for k, v in test_metrics.items() if v is not None}
        if metrics:
            plt.figure()
            plt.bar(list(metrics.keys()), list(metrics.values()))
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Test Metrics (Overall, SWA, CWA, ZSRTA)")
            fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

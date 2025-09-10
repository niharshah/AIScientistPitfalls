import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]

    # Helpers ----------------------------------------------------------
    losses_tr = ed["losses"].get("train", [])
    losses_val = ed["losses"].get("val", [])
    acc_tr = [m["acc"] for m in ed["metrics"].get("train", [])]
    acc_val = [m["acc"] for m in ed["metrics"].get("val", [])]
    swa_tr = [m["swa"] for m in ed["metrics"].get("train", [])]
    swa_val = [m["swa"] for m in ed["metrics"].get("val", [])]
    test_metrics = ed["metrics"].get("test", {})
    test_acc, test_swa = test_metrics.get("acc"), test_metrics.get("swa")
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])

    # 1) Loss Curves ---------------------------------------------------
    try:
        plt.figure()
        plt.plot(range(1, len(losses_tr) + 1), losses_tr, label="Train")
        plt.plot(range(1, len(losses_val) + 1), losses_val, label="Validation")
        plt.title("SPR_BENCH – Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Accuracy Curves ----------------------------------------------
    try:
        if acc_tr and acc_val:
            plt.figure()
            plt.plot(range(1, len(acc_tr) + 1), acc_tr, label="Train")
            plt.plot(range(1, len(acc_val) + 1), acc_val, label="Validation")
            plt.title("SPR_BENCH – Accuracy Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # 3) Shape-Weighted Accuracy Curves -------------------------------
    try:
        if swa_tr and swa_val:
            plt.figure()
            plt.plot(range(1, len(swa_tr) + 1), swa_tr, label="Train")
            plt.plot(range(1, len(swa_val) + 1), swa_val, label="Validation")
            plt.title("SPR_BENCH – Shape-Weighted Accuracy Curves")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # 4) Final Test Metrics Bar ---------------------------------------
    try:
        if test_acc is not None and test_swa is not None:
            plt.figure()
            metrics = ["Accuracy", "SWA"]
            vals = [test_acc, test_swa]
            plt.bar(metrics, vals, color=["steelblue", "tan"])
            plt.title("SPR_BENCH – Test Metrics")
            for i, v in enumerate(vals):
                plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
            fname = os.path.join(working_dir, "spr_bench_test_metrics_bar.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()

    # 5) Confusion Matrix ---------------------------------------------
    try:
        if preds and gts:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.colorbar()
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------- print final metrics --------------------------
    if test_acc is not None and test_swa is not None:
        print(f"Final Test Accuracy: {test_acc:.3f}")
        print(f"Final Test SWA     : {test_swa:.3f}")
else:
    print("No experiment data found for SPR_BENCH.")

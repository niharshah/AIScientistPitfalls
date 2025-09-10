import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ paths & data loading -------------
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
    logs = experiment_data.get("hybrid", {})
    epochs = logs.get("epochs", [])
    tr_loss = logs.get("losses", {}).get("train", [])
    val_loss = logs.get("losses", {}).get("val", [])
    tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
    val_mcc = logs.get("metrics", {}).get("val_MCC", [])
    test_mcc = logs.get("test_MCC")
    test_f1 = logs.get("test_F1")

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Train vs Val Loss — synthetic SPR_BENCH (hybrid)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_hybrid_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Plot 2: MCC curves ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_mcc, label="Train")
        plt.plot(epochs, val_mcc, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("Train vs Val MCC — synthetic SPR_BENCH (hybrid)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_hybrid_mcc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve plot: {e}")
        plt.close()

    # ---------- Plot 3: Test metrics ----------
    try:
        if test_mcc is not None and test_f1 is not None:
            plt.figure(figsize=(4, 4))
            metrics = ["MCC", "Macro-F1"]
            scores = [test_mcc, test_f1]
            plt.bar(metrics, scores, color=["skyblue", "salmon"])
            plt.ylim(0, 1)
            plt.title("Test Metrics — synthetic SPR_BENCH (hybrid)")
            plt.tight_layout()
            fname = os.path.join(working_dir, "spr_bench_hybrid_test_scores.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # --------- Print final evaluation ----------
    if test_mcc is not None and test_f1 is not None:
        print(f"Final Test MCC:  {test_mcc:.3f}")
        print(f"Final Test F1 :  {test_f1:.3f}")

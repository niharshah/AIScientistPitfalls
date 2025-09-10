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
    test_mcc = logs.get("test_MCC", None)

    # -------- 1) loss curve -------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Hybrid SPR_BENCH — Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = "spr_bench_hybrid_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------- 2) MCC curve -------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_mcc, label="Train MCC")
        plt.plot(epochs, val_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.title("Hybrid SPR_BENCH — Train vs Val MCC")
        plt.legend()
        plt.tight_layout()
        fname = "spr_bench_hybrid_mcc_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # -------- 3) Test MCC bar chart -------------
    try:
        if test_mcc is not None:
            plt.figure(figsize=(4, 4))
            plt.bar(["Hybrid"], [test_mcc], color="mediumseagreen")
            plt.ylabel("Test MCC")
            plt.title("Test MCC — SPR_BENCH")
            plt.tight_layout()
            fname = "spr_bench_hybrid_test_mcc.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            print(f"Hybrid Test MCC: {test_mcc:.3f}")
    except Exception as e:
        print(f"Error creating test MCC bar chart: {e}")
        plt.close()

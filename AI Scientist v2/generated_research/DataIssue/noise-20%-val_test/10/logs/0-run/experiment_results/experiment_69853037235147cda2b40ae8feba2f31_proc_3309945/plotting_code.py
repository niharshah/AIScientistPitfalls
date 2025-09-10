import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# safely get the inner dict
ed = experiment_data.get("NoAttnBiLSTM", {}).get("SPR_BENCH", {})

# 1) Loss curves ----------------------------------------------------
try:
    losses = ed.get("losses", {})
    tr_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    epochs = range(1, len(tr_loss) + 1)

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("SPR_BENCH – Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) F1 curves ------------------------------------------------------
try:
    metrics = ed.get("metrics", {})
    tr_f1 = metrics.get("train_f1", [])
    val_f1 = metrics.get("val_f1", [])
    epochs = range(1, len(tr_f1) + 1)

    plt.figure()
    plt.plot(epochs, tr_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Validation Macro-F1")
    plt.title("SPR_BENCH – Macro-F1 Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# 3) Rule-Extraction Accuracy bar chart -----------------------------
try:
    rea_dev = metrics.get("REA_dev")
    rea_test = metrics.get("REA_test")
    if rea_dev is not None and rea_test is not None:
        plt.figure()
        plt.bar(["Dev", "Test"], [rea_dev, rea_test], color=["skyblue", "lightgreen"])
        plt.title("SPR_BENCH – Rule-Extraction Accuracy\nLeft: Dev, Right: Test")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        for i, v in enumerate([rea_dev, rea_test]):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_REA_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating REA accuracy plot: {e}")
    plt.close()

print("Plot generation complete.")

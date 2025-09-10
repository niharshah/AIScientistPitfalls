import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
# Helper: MCC in pure numpy
def np_mcc(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


# ------------------------------------------------------------------
# 1) Per-learning-rate curves
test_mcc_scores = []
lr_labels = []

for lr_key, lr_dict in experiment_data.get("learning_rate", {}).items():
    losses = lr_dict["losses"]
    metrics = lr_dict["metrics"]
    epochs = range(1, len(losses["train"]) + 1)
    try:
        plt.figure(figsize=(10, 4))
        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses["train"], label="Train")
        plt.plot(epochs, losses["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss")
        plt.legend()
        # MCC subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics["train"], label="Train")
        plt.plot(epochs, metrics["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC")
        plt.legend()
        # Overall figure title
        plt.suptitle(f"SPR_BENCH | Learning Rate {lr_key} | Left: Loss, Right: MCC")
        save_name = f"SPR_BENCH_{lr_key}_curves.png"
        plt.savefig(os.path.join(working_dir, save_name))
    except Exception as e:
        print(f"Error creating plot for {lr_key}: {e}")
    finally:
        plt.close()
    # Compute and store test MCC
    try:
        y_true = np.array(lr_dict["ground_truth"])
        y_pred = np.array(lr_dict["predictions"])
        mcc_val = np_mcc(y_true, y_pred)
        test_mcc_scores.append(mcc_val)
        lr_labels.append(lr_key)
        print(f"Test MCC for {lr_key}: {mcc_val:.4f}")
    except Exception as e:
        print(f"Error computing test MCC for {lr_key}: {e}")

# ------------------------------------------------------------------
# 2) Bar chart of test MCC across learning rates
try:
    plt.figure()
    plt.bar(lr_labels, test_mcc_scores, color="skyblue")
    plt.ylabel("Test MCC")
    plt.title("SPR_BENCH | Test MCC Across Learning Rates")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_MCC_bar.png"))
except Exception as e:
    print(f"Error creating test MCC bar chart: {e}")
finally:
    plt.close()

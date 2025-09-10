import matplotlib.pyplot as plt
import numpy as np
import os

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

test_metrics = []  # collect for summary plot

for ds_name, rec in experiment_data.items():
    try:
        epochs = rec["epochs"]
        tr_loss, vl_loss = rec["losses"]["train"], rec["losses"]["val"]
        tr_f1, vl_f1 = rec["metrics"]["train_macro_f1"], rec["metrics"]["val_macro_f1"]
        tr_ema, vl_ema = rec["metrics"]["train_ema"], rec["metrics"]["val_ema"]

        plt.figure(figsize=(12, 3))
        plt.suptitle(f"{ds_name} – Training Curves")

        # Loss subplot
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(epochs, tr_loss, label="train")
        ax1.plot(epochs, vl_loss, label="val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy")
        ax1.legend()

        # Macro-F1 subplot
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(epochs, tr_f1, label="train")
        ax2.plot(epochs, vl_f1, label="val")
        ax2.set_title("Macro F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1")
        ax2.legend()

        # EMA subplot
        ax3 = plt.subplot(1, 3, 3)
        ax3.plot(epochs, tr_ema, label="train")
        ax3.plot(epochs, vl_ema, label="val")
        ax3.set_title("Exact-Match Acc")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, f"{ds_name}_training_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ds_name}: {e}")
        plt.close()

    # store test metrics
    test_metrics.append((ds_name, rec["test_macro_f1"], rec["test_ema"]))

# summary bar chart of test metrics
try:
    if test_metrics:
        labels, f1s, emas = zip(*test_metrics)
        x = np.arange(len(labels))
        w = 0.35
        plt.figure()
        plt.bar(x - w / 2, f1s, width=w, label="Test Macro F1")
        plt.bar(x + w / 2, emas, width=w, label="Test EMA")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.title("Test Performance Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "test_performance_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
    plt.close()

# print evaluation table
for name, f1, ema in test_metrics:
    print(f"{name:15s} | Test F1: {f1:.3f} | Test EMA: {ema:.3f}")

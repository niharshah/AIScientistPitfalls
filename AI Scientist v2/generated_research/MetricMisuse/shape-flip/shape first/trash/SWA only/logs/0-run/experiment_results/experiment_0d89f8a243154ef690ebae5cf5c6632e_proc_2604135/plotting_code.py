import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- paths and data --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    train_loss = data["losses"]["train"]
    val_swa_curve = data["losses"]["val"]  # stored as SWA during training
    epochs = range(1, len(train_loss) + 1)
    test_swa = data["metrics"]["test"]["swa"]
    val_swa_last = val_swa_curve[-1] if val_swa_curve else None
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])

    # 1) Training loss + Val SWA curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_swa_curve, label="Val SWA")
        plt.xlabel("Epoch")
        plt.title("SPR_BENCH – Training Loss & Validation SWA vs Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_and_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss/SWA curve: {e}")
        plt.close()

    # 2) Val vs Test SWA bar chart
    try:
        plt.figure()
        plt.bar([0, 1], [val_swa_last, test_swa], tick_label=["Val SWA", "Test SWA"])
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH – Validation vs Test SWA")
        fname = os.path.join(working_dir, "spr_bench_val_vs_test_swa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA bar chart: {e}")
        plt.close()

    # 3) Confusion matrix heat-map
    try:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    print(f"TEST SWA = {test_swa:.4f}")

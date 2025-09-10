import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_macroF1"], label="Train")
        plt.plot(epochs, ed["metrics"]["val_macroF1"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot: {e}")
        plt.close()

    # 3) SC-Gmean curve
    try:
        plt.figure()
        sc_gmean = ed["metrics"]["val_SC_Gmean"]
        plt.plot(epochs, sc_gmean, marker="o")
        best_ep = int(np.argmax(sc_gmean)) + 1
        plt.scatter(
            best_ep, sc_gmean[best_ep - 1], color="red", label=f"Best (epoch {best_ep})"
        )
        plt.xlabel("Epoch")
        plt.ylabel("SC-Gmean")
        plt.title("SPR_BENCH Validation SC-Gmean\nBest epoch highlighted")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_SC_Gmean_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SC-Gmean plot: {e}")
        plt.close()

    # 4) SWA vs CWA scatter
    try:
        plt.figure()
        plt.scatter(
            ed["metrics"]["val_SWA"], ed["metrics"]["val_CWA"], c=epochs, cmap="viridis"
        )
        plt.colorbar(label="Epoch")
        plt.xlabel("Shape-Weighted Accuracy (SWA)")
        plt.ylabel("Color-Weighted Accuracy (CWA)")
        plt.title("SPR_BENCH SWA vs. CWA Scatter\nColor indicates epoch number")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_vs_CWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA vs CWA plot: {e}")
        plt.close()

    # 5) Confusion matrix for best epoch
    try:
        y_true = ed["ground_truth"]
        y_pred = ed["predictions"]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

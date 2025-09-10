import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load metrics
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if experiment_data:
    model_key = next(iter(experiment_data))  # 'Shallow_GNN_1hop'
    dataset_key = next(iter(experiment_data[model_key]))  # 'synthetic' or real
    ed = experiment_data[model_key][dataset_key]

    epochs = ed["epochs"]
    tr_loss = ed["losses"]["train"]
    vl_loss = ed["losses"]["val"]
    tr_cwa = ed["metrics"]["train"]["CWA"]
    vl_cwa = ed["metrics"]["val"]["CWA"]
    tr_swa = ed["metrics"]["train"]["SWA"]
    vl_swa = ed["metrics"]["val"]["SWA"]
    tr_cmp = ed["metrics"]["train"]["CmpWA"]
    vl_cmp = ed["metrics"]["val"]["CmpWA"]
    preds = np.array(ed["predictions"])
    gtruth = np.array(ed["ground_truth"])
    n_cls = len(np.unique(gtruth))

    # 1) loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, vl_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset_key}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) complexity weighted accuracy
    try:
        plt.figure()
        plt.plot(epochs, tr_cmp, label="Train")
        plt.plot(epochs, vl_cmp, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cmp-Weighted Accuracy")
        plt.title(f"{dataset_key}: Complexity-Weighted Accuracy (Train vs Validation)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_cmpWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating cmpWA plot: {e}")
        plt.close()

    # 3) color weighted accuracy
    try:
        plt.figure()
        plt.plot(epochs, tr_cwa, label="Train")
        plt.plot(epochs, vl_cwa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Color-Weighted Accuracy")
        plt.title(f"{dataset_key}: Color-Weighted Accuracy (Train vs Validation)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_CWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # 4) shape weighted accuracy
    try:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train")
        plt.plot(epochs, vl_swa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dataset_key}: Shape-Weighted Accuracy (Train vs Validation)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_SWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 5) confusion matrix (optional, keeps total ≤5)
    if n_cls <= 10:
        try:
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gtruth, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dataset_key}: Confusion Matrix (Test)")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dataset_key}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            plt.close()

    # print final test metrics
    print("Test metrics:", ed.get("test_metrics", {}))

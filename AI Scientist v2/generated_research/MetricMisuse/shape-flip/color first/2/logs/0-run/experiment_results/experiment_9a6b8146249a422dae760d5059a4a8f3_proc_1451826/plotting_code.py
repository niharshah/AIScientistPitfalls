import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ablation_key = list(experiment_data.keys())[0]
    dataset_key = list(experiment_data[ablation_key].keys())[0]
    data_dict = experiment_data[ablation_key][dataset_key]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data_dict = None

if data_dict is not None:
    # --------------------------- helpers ------------------------
    def unpack(k):
        arr = data_dict[k]  # dict(train,val)
        epochs, train_v = zip(*arr["train"])
        _, val_v = zip(*arr["val"])
        return epochs, train_v, val_v

    epochs, tr_losses, va_losses = unpack("losses")
    _, tr_pc, va_pc = unpack("metrics")

    # --------------------------- PLOT 1 -------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, va_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = f"{dataset_key}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------------------------- PLOT 2 -------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_pc, label="Train")
        plt.plot(epochs, va_pc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("SPR PCWA Metric Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = f"{dataset_key}_pcwa_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA curve plot: {e}")
        plt.close()

    # --------------------------- PLOT 3 -------------------------
    try:
        y_true = np.array(data_dict["ground_truth"])
        y_pred = np.array(data_dict["predictions"])
        conf = np.zeros((2, 2), int)
        for t, p in zip(y_true, y_pred):
            conf[t, p] += 1
        plt.figure()
        plt.imshow(conf, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, conf[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        fname = f"{dataset_key}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # --------------------------- METRICS ------------------------
    acc = (y_true == y_pred).mean()

    # quick PCWA recompute
    def pcwa(seqs, y_t, y_p):
        def cvar(s):
            return len(set(tok[1] for tok in s.split()))

        def svar(s):
            return len(set(tok[0] for tok in s.split()))

        w = [cvar(s) * svar(s) for s in seqs]
        corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
        return sum(corr) / sum(w) if sum(w) else 0.0

    pcwa_val = (
        pcwa(data_dict.get("ground_truth_seqs", []), y_true.tolist(), y_pred.tolist())
        if "ground_truth_seqs" in data_dict
        else "n/a"
    )
    print(f"Test ACC: {acc:.4f} | Test PCWA: {pcwa_val}")

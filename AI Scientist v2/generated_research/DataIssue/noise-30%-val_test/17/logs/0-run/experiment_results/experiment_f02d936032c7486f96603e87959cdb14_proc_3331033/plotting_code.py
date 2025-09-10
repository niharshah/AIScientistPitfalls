import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp_dict = experiment_data["dropout_rate"]
    keys = sorted(exp_dict.keys(), key=lambda x: float(x.split("=")[1]))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # ---------- Figure 1: loss curves ----------
    try:
        plt.figure()
        for i, k in enumerate(keys):
            ls_tr = exp_dict[k]["losses"]["train"]
            ls_val = exp_dict[k]["losses"]["val"]
            epochs = np.arange(1, len(ls_tr) + 1)
            plt.plot(
                epochs,
                ls_tr,
                linestyle="--",
                color=colors[i % len(colors)],
                label=f"{k} train",
            )
            plt.plot(
                epochs,
                ls_val,
                linestyle="-",
                color=colors[i % len(colors)],
                label=f"{k} val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Figure 2: MCC curves ----------
    try:
        plt.figure()
        for i, k in enumerate(keys):
            mcc_tr = exp_dict[k]["metrics"]["train"]
            mcc_val = exp_dict[k]["metrics"]["val"]
            epochs = np.arange(1, len(mcc_tr) + 1)
            plt.plot(
                epochs,
                mcc_tr,
                linestyle="--",
                color=colors[i % len(colors)],
                label=f"{k} train",
            )
            plt.plot(
                epochs,
                mcc_val,
                linestyle="-",
                color=colors[i % len(colors)],
                label=f"{k} val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Corrcoef")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_MCC_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve plot: {e}")
        plt.close()

    # ---------- Figure 3: final val MCC bar ----------
    try:
        plt.figure()
        final_val_mcc = [exp_dict[k]["metrics"]["val"][-1] for k in keys]
        plt.bar(keys, final_val_mcc, color=colors[: len(keys)])
        plt.ylabel("Final Validation MCC")
        plt.title("SPR_BENCH: Final Validation MCC per Dropout Rate")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_MCC_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final val MCC bar plot: {e}")
        plt.close()

    # ---------- Figure 4: confusion matrix for best model ----------
    try:
        # detect best key (has 'ground_truth')
        best_key = [k for k in keys if "ground_truth" in exp_dict[k]][0]
        y_true = exp_dict[best_key]["ground_truth"]
        y_pred = exp_dict[best_key]["predictions"]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xticks([0, 1], ["True 0", "True 1"])
        plt.yticks([0, 1], ["Pred 0", "Pred 1"])
        plt.title(f"SPR_BENCH: Confusion Matrix (Best Model {best_key})")
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

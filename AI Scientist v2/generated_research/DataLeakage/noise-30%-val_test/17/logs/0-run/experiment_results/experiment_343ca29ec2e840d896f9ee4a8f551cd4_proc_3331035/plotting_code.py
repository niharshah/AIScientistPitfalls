import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef, f1_score  # only used for bar chart

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data is not None:
    # ---------- gather per-epoch loss & metric sequences ---------- #
    train_losses = np.asarray(spr_data["losses"]["train"])
    val_losses = np.asarray(spr_data["losses"]["val"])
    train_mcc = np.asarray([m["mcc"] for m in spr_data["metrics"]["train"]])
    val_mcc = np.asarray([m["mcc"] for m in spr_data["metrics"]["val"]])
    train_f1 = np.asarray([m["macro_f1"] for m in spr_data["metrics"]["train"]])
    val_f1 = np.asarray([m["macro_f1"] for m in spr_data["metrics"]["val"]])
    epochs = np.arange(1, len(train_losses) + 1)

    # ------------------- PLOT 1: loss curves ---------------------- #
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCEWithLogits Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------------- PLOT 2: MCC & macro-F1 curves --------------- #
    try:
        fig, ax1 = plt.subplots()
        ax1.plot(epochs, train_mcc, "b-", label="Train MCC")
        ax1.plot(epochs, val_mcc, "c-", label="Val MCC")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MCC", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax2 = ax1.twinx()
        ax2.plot(epochs, train_f1, "r--", label="Train macro-F1")
        ax2.plot(epochs, val_f1, "m--", label="Val macro-F1")
        ax2.set_ylabel("macro-F1", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        lines, labels = [], []
        for ax in [ax1, ax2]:
            line, label = ax.get_legend_handles_labels()
            lines += line
            labels += label
        plt.legend(lines, labels, loc="lower right")
        plt.title("SPR_BENCH Performance Curves\nLeft: MCC, Right: macro-F1")
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_F1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC/F1 curve: {e}")
        plt.close()

    # ----------- PLOT 3: bar chart of test MCC per run ----------- #
    try:
        preds_list = spr_data["predictions"]
        labels_list = spr_data["ground_truth"]
        test_mccs = [matthews_corrcoef(l, p) for p, l in zip(preds_list, labels_list)]
        run_tags = [f"{b}ep" for b in spr_data["epoch_budgets"]]
        plt.figure()
        plt.bar(run_tags, test_mccs, color="skyblue")
        plt.ylim(0, 1)
        for i, v in enumerate(test_mccs):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title("SPR_BENCH Test MCC by Training Budget\nBar heights = MCC scores")
        plt.ylabel("Test MCC")
        fname = os.path.join(working_dir, "SPR_BENCH_test_MCC_bar.png")
        plt.savefig(fname)
        plt.close()
        print(
            "Test MCC per run:", dict(zip(run_tags, [round(x, 4) for x in test_mccs]))
        )
    except Exception as e:
        print(f"Error creating test MCC bar plot: {e}")
        plt.close()

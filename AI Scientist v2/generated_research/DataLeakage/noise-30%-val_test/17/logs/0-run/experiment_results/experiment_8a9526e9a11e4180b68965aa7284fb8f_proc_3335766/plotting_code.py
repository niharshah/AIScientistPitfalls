import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------#
for dname, dct in experiment_data.items():
    n_runs = len(dct["configs"])
    tr_losses = np.array(dct["losses"]["train"])
    val_losses = np.array(dct["losses"]["val"])
    tr_mcc = np.array(dct["metrics"]["train"])
    val_mcc = np.array(dct["metrics"]["val"])

    # figure out epoch splits for multiple runs
    lengths = []
    ptr = 0
    for _ in range(n_runs):
        # assume each run logged val loss every epoch until a new lr run started
        # detect by early-stop length stored in configs["epochs"]
        L = dct["configs"][_]["epochs"]
        lengths.append(L)
        ptr += L

    # --------------------- per-run plots ---------------------------#
    start = 0
    for r, L in enumerate(lengths):
        end = start + L
        ep_range = np.arange(1, L + 1)

        # Loss curves
        try:
            plt.figure()
            plt.plot(ep_range, tr_losses[start:end], label="Train")
            plt.plot(ep_range, val_losses[start:end], label="Validation")
            plt.title(
                f"{dname} Loss Curves (Run {r+1})\nLeft: Train, Right: Validation"
            )
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_loss_run{r+1}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot run {r+1}: {e}")
            plt.close()

        # MCC curves
        try:
            plt.figure()
            plt.plot(ep_range, tr_mcc[start:end], label="Train")
            plt.plot(ep_range, val_mcc[start:end], label="Validation")
            plt.title(f"{dname} MCC Curves (Run {r+1})\nLeft: Train, Right: Validation")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_mcc_run{r+1}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating MCC plot run {r+1}: {e}")
            plt.close()
        start = end

    # ---------------- comparison of best val MCC across runs -------#
    try:
        best_val_mcc = [
            val_mcc[sum(lengths[:i]) : sum(lengths[: i + 1])].max()
            for i in range(n_runs)
        ]
        plt.figure()
        plt.bar([f"Run {i+1}" for i in range(n_runs)], best_val_mcc, color="seagreen")
        plt.ylim(0, 1)
        plt.title(f"{dname} Best Validation MCC per Run")
        for i, v in enumerate(best_val_mcc):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, f"{dname.lower()}_best_val_mcc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()

    # ----------------- Test metrics bar chart ----------------------#
    try:
        preds = np.array(dct["predictions"][0]).flatten()
        gts = np.array(dct["ground_truth"][0]).flatten()
        test_f1 = f1_score(gts, preds, average="macro")
        test_mcc = matthews_corrcoef(gts, preds)

        plt.figure()
        plt.bar(["Macro-F1", "MCC"], [test_f1, test_mcc], color=["steelblue", "orange"])
        plt.ylim(0, 1)
        plt.title(f"{dname} Test Metrics\nMacro-F1 vs MCC")
        for i, v in enumerate([test_f1, test_mcc]):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, f"{dname.lower()}_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"{dname} | Test Macro-F1: {test_f1:.4f} | Test MCC: {test_mcc:.4f}")
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()

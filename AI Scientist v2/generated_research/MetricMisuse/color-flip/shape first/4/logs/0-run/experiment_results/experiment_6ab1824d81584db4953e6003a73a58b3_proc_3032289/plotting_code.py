import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
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

# iterate over datasets contained in experiment_data
for dname, rec in experiment_data.items():
    losses_tr = rec["losses"]["train"]
    losses_va = rec["losses"]["val"]
    metrics_va = rec["metrics"]["val"]
    preds_list = rec["predictions"]
    gts_list = rec["ground_truth"]

    # -------- Plot 1: loss curves --------
    try:
        if losses_tr and losses_va:
            x = np.arange(1, len(losses_tr) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(x, losses_tr, "--o", label="train")
            plt.plot(x, losses_va, "-s", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Train vs Val Loss\nLeft: Train, Right: Val")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"{dname}: no loss data to plot.")
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
    finally:
        plt.close()

    # -------- Plot 2: weighted accuracy curves --------
    try:
        if metrics_va and isinstance(metrics_va[0], dict):
            x = np.arange(1, len(metrics_va) + 1)
            swa = [m["SWA"] for m in metrics_va]
            cwa = [m["CWA"] for m in metrics_va]
            cowa = [m["CoWA"] for m in metrics_va]
            plt.figure(figsize=(6, 4))
            plt.plot(x, swa, "-o", label="SWA")
            plt.plot(x, cwa, "-s", label="CWA")
            plt.plot(x, cowa, "-^", label="CoWA")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(f"{dname}: Validation Weighted Accuracies\nSWA/CWA/CoWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_weighted_acc_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"{dname}: no weighted accuracy data to plot.")
    except Exception as e:
        print(f"Error creating weighted accuracy plot for {dname}: {e}")
    finally:
        plt.close()

    # -------- Plot 3: plain validation accuracy --------
    try:
        if preds_list and gts_list:
            acc = [
                np.mean(np.array(p) == np.array(g))
                for p, g in zip(preds_list, gts_list)
            ]
            x = np.arange(1, len(acc) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(x, acc, "-d")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Validation Accuracy over Epochs")
            fname = os.path.join(working_dir, f"{dname}_val_accuracy.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"{dname}: no prediction data to plot accuracy.")
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
    finally:
        plt.close()

    # -------- Print final metrics summary --------
    if metrics_va:
        last = metrics_va[-1]
        print(
            f"{dname} final metrics -> SWA: {last['SWA']:.3f}, "
            f"CWA: {last['CWA']:.3f}, CoWA: {last['CoWA']:.3f}"
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(os.getcwd(), "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# extract needed arrays
ed = experiment_data.get("weight_decay", {}).get("SPR_BENCH", None)
if ed:
    wds = [hp["weight_decay"] for hp in ed["hyperparams"]]
    cwa_tr = ed["metrics"]["train"]
    cwa_val = ed["metrics"]["val"]
    loss_tr = ed["losses"]["train"]
    loss_val = ed["losses"]["val"]

    # print metrics table
    print("\nWeight Decay | CWA2_train | CWA2_val | Loss_train | Loss_val")
    for a, b, c, d, e in zip(wds, cwa_tr, cwa_val, loss_tr, loss_val):
        print(f"{a:12g} | {b:10.4f} | {c:8.4f} | {d:10.4f} | {e:8.4f}")

    # 1) CWA bar chart
    try:
        x = np.arange(len(wds))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, cwa_tr, width, label="Train")
        plt.bar(x + width / 2, cwa_val, width, label="Validation")
        plt.xticks(x, [str(wd) for wd in wds])
        plt.ylabel("CWA2 Score")
        plt.title(
            "SPR_BENCH: Final Complexity Weighted Accuracy\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # 2) Loss bar chart
    try:
        x = np.arange(len(wds))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, loss_tr, width, label="Train")
        plt.bar(x + width / 2, loss_val, width, label="Validation")
        plt.xticks(x, [str(wd) for wd in wds])
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Final Loss Values\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_Loss_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) Scatter/line of val CWA vs weight decay
    try:
        plt.figure()
        plt.plot(wds, cwa_val, marker="o")
        plt.xscale("log")
        plt.xlabel("Weight Decay (log scale)")
        plt.ylabel("Validation CWA2")
        plt.title("SPR_BENCH: Validation CWA2 vs. Weight Decay")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_vs_WD.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA-vs-WD plot: {e}")
        plt.close()
else:
    print("No SPR_BENCH data found in experiment_data.")

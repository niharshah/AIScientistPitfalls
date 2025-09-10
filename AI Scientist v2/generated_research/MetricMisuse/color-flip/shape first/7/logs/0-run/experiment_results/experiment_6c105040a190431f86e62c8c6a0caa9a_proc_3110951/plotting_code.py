import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup & load data ------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# proceed only if data exist
if experiment_data:
    run = experiment_data.get("bag_of_words", {}).get("SPR_BENCH", {})
    losses_tr = run.get("losses", {}).get("train", [])
    losses_va = run.get("losses", {}).get("val", [])
    metrics_va = run.get("metrics", {}).get("val", [])  # (epoch, swa, cwa, hwa)

    # unpack data
    ep_l_tr, l_tr = zip(*losses_tr) if losses_tr else ([], [])
    ep_l_va, l_va = zip(*losses_va) if losses_va else ([], [])
    ep_m, swa, cwa, hwa = zip(*metrics_va) if metrics_va else ([], [], [], [])

    # --------- plot 1 : Loss curves ---------
    try:
        plt.figure()
        if ep_l_tr:
            plt.plot(ep_l_tr, l_tr, label="Train Loss")
        if ep_l_va:
            plt.plot(ep_l_va, l_va, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves - Bag of Words")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_bow_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------- plot 2 : Accuracy curves -----
    try:
        plt.figure()
        if ep_m:
            plt.plot(ep_m, swa, label="SWA")
            plt.plot(ep_m, cwa, label="CWA")
            plt.plot(ep_m, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Weighted Accuracy Curves - Bag of Words")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_bow_weighted_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # --------- print final metrics ----------
    if hwa:
        print(
            f"Final Epoch Metrics -> SWA: {swa[-1]:.4f}, CWA: {cwa[-1]:.4f}, HWA: {hwa[-1]:.4f}"
        )

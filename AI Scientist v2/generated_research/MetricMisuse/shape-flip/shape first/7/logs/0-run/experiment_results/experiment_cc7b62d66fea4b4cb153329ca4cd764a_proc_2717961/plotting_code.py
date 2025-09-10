import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    edim_dict = experiment_data.get("embed_dim", {})
    exp_keys = sorted(
        edim_dict.keys(), key=lambda k: int(k.split("ed")[-1])
    )  # sort by embed dim

    # --------------------------------------------------------------
    # Collect data
    losses_train, losses_val = {}, {}
    hwa_train, hwa_val = {}, {}
    test_acc = {}

    for k in exp_keys:
        tr = edim_dict[k]["losses"]["train"]  # list[(epoch, value)]
        vl = edim_dict[k]["losses"]["val"]
        mtr = edim_dict[k]["metrics"]["train"]
        mvl = edim_dict[k]["metrics"]["val"]

        epochs_tr = [e for e, _ in tr]
        losses_train[k] = [v for _, v in tr]
        losses_val[k] = [v for _, v in vl]
        hwa_train[k] = [v for _, v in mtr]
        hwa_val[k] = [v for _, v in mvl]

        g = edim_dict[k]["ground_truth"]
        p = edim_dict[k]["predictions"]
        if len(g):
            test_acc[k] = sum(int(gt == pr) for gt, pr in zip(g, p)) / len(g)
        else:
            test_acc[k] = np.nan

    # --------------------------------------------------------------
    # Figure 1 : Loss curves
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        for k in exp_keys:
            plt.plot(epochs_tr, losses_train[k], label=k)
        plt.title("SPR_BENCH – Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        for k in exp_keys:
            plt.plot(epochs_tr, losses_val[k], label=k)
        plt.title("Validation Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend(fontsize="small")

        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --------------------------------------------------------------
    # Figure 2 : HWA curves
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        for k in exp_keys:
            plt.plot(epochs_tr, hwa_train[k], label=k)
        plt.title("SPR_BENCH – Training HWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        for k in exp_keys:
            plt.plot(epochs_tr, hwa_val[k], label=k)
        plt.title("Validation HWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize="small")

        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_hwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curves: {e}")
        plt.close()

    # --------------------------------------------------------------
    # Figure 3 : Test accuracy bar chart
    try:
        plt.figure(figsize=(6, 4))
        dims = [int(k.split("ed")[-1]) for k in exp_keys]
        accs = [test_acc[k] for k in exp_keys]
        plt.bar([str(d) for d in dims], accs)
        plt.title("SPR_BENCH – Test Accuracy by Embedding Dim")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Accuracy")
        for i, a in enumerate(accs):
            plt.text(i, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)

        fname = os.path.join(working_dir, "spr_bench_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar chart: {e}")
        plt.close()

    # --------------------------------------------------------------
    # Print evaluation metric
    print("Test Accuracy by embed_dim:")
    for k in exp_keys:
        print(f"  {k}: {test_acc[k]:.4f}")

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
    variants = ["neural", "symbolic", "hybrid"]
    epochs = {}
    loss_tr, loss_val = {}, {}
    swa_tr, swa_val = {}, {}
    test_acc = {}

    for v in variants:
        lt = experiment_data[v]["losses"]["train"]
        lv = experiment_data[v]["losses"]["val"]
        mt = experiment_data[v]["metrics"]["train"]
        mv = experiment_data[v]["metrics"]["val"]
        if lt and lv:
            epochs[v] = [e for e, _ in lt]
            loss_tr[v] = [x for _, x in lt]
            loss_val[v] = [x for _, x in lv]
        if mt and mv:
            swa_tr[v] = [x for _, x in mt]
            swa_val[v] = [x for _, x in mv]
        g = experiment_data[v]["ground_truth"]
        p = experiment_data[v]["predictions"]
        test_acc[v] = (
            sum(int(a == b) for a, b in zip(g, p)) / len(g) if len(g) else np.nan
        )

    # ---------------- Loss curves ----------------
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        for v in variants:
            if v in loss_tr:
                plt.plot(epochs[v], loss_tr[v], label=f"{v}-train")
        plt.title("SPR_BENCH – Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        for v in variants:
            if v in loss_val:
                plt.plot(epochs[v], loss_val[v], label=f"{v}-val")
        plt.title("Validation Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------------- SWA curves -----------------
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        for v in variants:
            if v in swa_tr:
                plt.plot(epochs[v], swa_tr[v], label=f"{v}-train")
        plt.title("SPR_BENCH – Training SWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        for v in variants:
            if v in swa_val:
                plt.plot(epochs[v], swa_val[v], label=f"{v}-val")
        plt.title("Validation SWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc")
        plt.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curves: {e}")
        plt.close()

    # ---------------- Test accuracy --------------
    try:
        plt.figure(figsize=(6, 4))
        names = list(test_acc.keys())
        vals = [test_acc[n] for n in names]
        plt.bar(names, vals)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.title("SPR_BENCH – Test Accuracy by Model")
        plt.xlabel("Model Variant")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(working_dir, "spr_bench_test_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar chart: {e}")
        plt.close()

    # --------------- Print metrics ---------------
    print("Test Accuracy:")
    for k, v in test_acc.items():
        print(f"  {k}: {v:.4f}")

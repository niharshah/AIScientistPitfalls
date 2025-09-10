import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    dr_data = experiment_data.get("dropout_rate", {})
    dropouts = sorted([float(k) for k in dr_data.keys()])
    epochs = range(1, len(next(iter(dr_data.values()))["losses"]["train"]) + 1)

    # collect metrics
    val_hwa_last = []
    test_hwa = []
    for p in dropouts:
        rec = dr_data[str(p)]
        val_hwa_last.append(rec["metrics"]["val"][-1][2])
        test_hwa.append(rec["metrics"]["test"][2])

    # 1) Loss curves
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for p in dropouts:
            plt.plot(epochs, dr_data[str(p)]["losses"]["train"], label=f"drop={p}")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        for p in dropouts:
            plt.plot(epochs, dr_data[str(p)]["losses"]["val"], label=f"drop={p}")
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.suptitle(
            "Loss Curves Across Dropout Rates\nLeft: Training, Right: Validation"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fname = os.path.join(working_dir, "spr_loss_curves_vs_dropout.png")
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # 2) HWA curves
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for p in dropouts:
            hwa_train = [m[2] for m in dr_data[str(p)]["metrics"]["train"]]
            plt.plot(epochs, hwa_train, label=f"drop={p}")
        plt.title("Training HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.subplot(1, 2, 2)
        for p in dropouts:
            hwa_val = [m[2] for m in dr_data[str(p)]["metrics"]["val"]]
            plt.plot(epochs, hwa_val, label=f"drop={p}")
        plt.title("Validation HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.suptitle(
            "Harmonic Weighted Accuracy (HWA) Curves\nLeft: Training, Right: Validation"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fname = os.path.join(working_dir, "spr_hwa_curves_vs_dropout.png")
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curves plot: {e}")
        plt.close()

    # 3) Test HWA bar chart
    try:
        plt.figure(figsize=(6, 4))
        plt.bar([str(p) for p in dropouts], test_hwa, color="skyblue")
        plt.title("Final Test HWA vs. Dropout Rate\nDataset: SPR_BENCH")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Test HWA")
        fname = os.path.join(working_dir, "spr_test_hwa_barplot.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA bar plot: {e}")
        plt.close()

    # print key metrics
    print("Dropout | Final Val HWA | Test HWA")
    for p, v, t in zip(dropouts, val_hwa_last, test_hwa):
        print(f"  {p:4.1f}  |    {v:0.4f}    |  {t:0.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Abort early if dataset missing
if (
    not experiment_data
    or "dropout_rate" not in experiment_data
    or "SPR_BENCH" not in experiment_data["dropout_rate"]
):
    print("No SPR_BENCH data found in experiment_data.npy")
else:
    spr_dict = experiment_data["dropout_rate"]["SPR_BENCH"]
    dropouts = sorted(float(k) for k in spr_dict.keys())
    epochs = np.arange(1, len(next(iter(spr_dict.values()))["metrics"]["val"]) + 1)

    # Collect curves & test metrics
    val_curves, train_curves, test_bwa = {}, {}, {}
    for dr in dropouts:
        d = spr_dict[str(dr)]
        train_curves[dr] = d["metrics"]["train"]
        val_curves[dr] = d["metrics"]["val"]
        test_bwa[dr] = d["test_metrics"]["BWA"]

    # -------- Plot 1: Dev BWA across dropout rates --------
    try:
        plt.figure()
        for dr in dropouts:
            plt.plot(epochs, val_curves[dr], label=f"dropout={dr}")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title("SPR_BENCH: Dev BWA vs Epochs for different dropouts")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_dev_BWA_all_dropouts.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Identify best dropout (highest final dev BWA)
    best_dr = max(dropouts, key=lambda d: val_curves[d][-1])

    # -------- Plot 2: Train vs Dev BWA for best dropout --------
    try:
        plt.figure()
        plt.plot(epochs, train_curves[best_dr], label="Train BWA")
        plt.plot(epochs, val_curves[best_dr], label="Dev BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"SPR_BENCH: Train vs Dev BWA (best dropout={best_dr})")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_BWA_curves_best_dropout_{best_dr}.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # -------- Plot 3: Test BWA bar chart --------
    try:
        plt.figure()
        plt.bar([str(d) for d in dropouts], [test_bwa[d] for d in dropouts])
        plt.xlabel("Dropout Rate")
        plt.ylabel("Test BWA")
        plt.title("SPR_BENCH: Test BWA vs Dropout Rate")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_BWA_vs_dropout.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # -------- Console summary --------
    print(f"Best dropout (by final Dev BWA): {best_dr}")
    print(f"  Final Dev BWA:  {val_curves[best_dr][-1]:.4f}")
    print(f"  Test BWA:       {test_bwa[best_dr]:.4f}")

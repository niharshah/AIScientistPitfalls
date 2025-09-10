import matplotlib.pyplot as plt
import numpy as np
import os

# set and load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    dr_data = experiment_data.get("dropout_rate", {})
    dropout_tags = sorted(dr_data.keys(), key=lambda x: float(x))
    # -------- collect test metrics for printing --------
    test_summary = {
        tag: dr_data[tag]["metrics"]["test"][-1]
        for tag in dropout_tags
        if "metrics" in dr_data[tag]
    }
    print("TEST DWHS by dropout:")
    for tag, (cwa, swa, dwhs) in test_summary.items():
        print(f"  dropout={tag}: DWHS={dwhs:.3f} (CWA={cwa:.3f}, SWA={swa:.3f})")
    # ------------- PLOT 1: loss curves -----------------
    try:
        plt.figure()
        for tag in dropout_tags:
            tr = np.array(dr_data[tag]["losses"]["train"])
            va = np.array(dr_data[tag]["losses"]["val"])
            plt.plot(tr[:, 0], tr[:, 1], label=f"train dr={tag}")
            plt.plot(va[:, 0], va[:, 1], "--", label=f"val dr={tag}")
        plt.title("SPR_BENCH Training/Validation Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()
    # ------------- PLOT 2: validation DWHS -------------
    try:
        plt.figure()
        for tag in dropout_tags:
            val_metrics = np.array(dr_data[tag]["metrics"]["val"])
            epochs, _, _, dwhs = (
                val_metrics[:, 0],
                val_metrics[:, 1],
                val_metrics[:, 2],
                val_metrics[:, 3],
            )
            plt.plot(epochs, dwhs, label=f"dr={tag}")
        plt.title("SPR_BENCH Validation DWHS vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("DWHS")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_DWHS_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val DWHS plot: {e}")
        plt.close()
    # ------------- PLOT 3: test DWHS bar chart ----------
    try:
        plt.figure()
        tags = list(test_summary.keys())
        scores = [test_summary[t][-1] for t in tags]  # DWHS
        plt.bar(tags, scores)
        plt.title("SPR_BENCH Test DWHS by Dropout Rate")
        plt.xlabel("Dropout Rate")
        plt.ylabel("DWHS")
        fname = os.path.join(working_dir, "SPR_BENCH_test_DWHS_bars.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test DWHS bar chart: {e}")
        plt.close()

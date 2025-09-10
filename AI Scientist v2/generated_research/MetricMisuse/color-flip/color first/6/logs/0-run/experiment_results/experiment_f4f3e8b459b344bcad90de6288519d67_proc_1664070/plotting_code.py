import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------
# load experiment data
try:
    exp_path1 = os.path.join(working_dir, "experiment_data.npy")
    exp_path2 = "experiment_data.npy"
    if os.path.exists(exp_path1):
        exp_file = exp_path1
    else:
        exp_file = exp_path2
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to shorten path building
def save_fig(name):
    return os.path.join(working_dir, name)


# ------------------------------------------------------
# guard against empty data
if experiment_data:
    try:
        wd_dict = experiment_data["weight_decay"]["SPR_BENCH"]
        wds = sorted(wd_dict.keys(), key=float)
    except Exception as e:
        print(f"Unexpected data format: {e}")
        wd_dict, wds = {}, []

    # --------------- Figure 1: loss curves ----------------
    try:
        plt.figure()
        for wd in wds:
            d = wd_dict[wd]
            eps = [e for e, _ in d["losses"]["train"]]
            tr = [l for _, l in d["losses"]["train"]]
            val = [l for _, l in d["losses"]["val"]]
            plt.plot(eps, tr, label=f"train wd={wd}")
            plt.plot(eps, val, linestyle="--", label=f"val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss\n(Left: Train, Right: Val)")
        plt.legend(fontsize=8)
        plt.savefig(save_fig("SPR_BENCH_loss_curves_weight_decay.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------------- Figures 2-4: metric curves ------------
    metric_names = ["CWA", "SWA", "EWA"]
    for m in metric_names:
        try:
            plt.figure()
            for wd in wds:
                d = wd_dict[wd]
                eps = [e for e, _ in d["metrics"]["val"]]
                vals = [meas[m] for _, meas in d["metrics"]["val"]]
                plt.plot(eps, vals, label=f"wd={wd}")
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.title(
                f"SPR_BENCH – Validation {m} Across Epochs\n(Weight-decay comparison)"
            )
            plt.legend(fontsize=8)
            plt.savefig(save_fig(f"SPR_BENCH_{m}_curves_weight_decay.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating {m} plot: {e}")
            plt.close()

    # --------------- Figure 5: test metrics bar ------------
    try:
        ind = np.arange(len(wds))
        width = 0.2
        plt.figure()
        for i, m in enumerate(metric_names):
            vals = [wd_dict[wd]["test_metrics"][m] for wd in wds]
            plt.bar(ind + i * width, vals, width, label=m)
        plt.xticks(ind + width, wds)
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH – Test Metrics by Weight Decay\n(Left: CWA, Middle: SWA, Right: EWA)"
        )
        plt.legend()
        plt.savefig(save_fig("SPR_BENCH_test_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot: {e}")
        plt.close()

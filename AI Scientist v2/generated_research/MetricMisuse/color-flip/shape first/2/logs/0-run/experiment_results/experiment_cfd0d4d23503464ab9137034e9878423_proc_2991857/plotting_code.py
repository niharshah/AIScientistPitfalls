import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {}).get("runs", {})
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # --------- plot 1: loss curves ---------
    try:
        plt.figure()
        for (wd, run), c in zip(runs.items(), colors):
            epochs = range(1, len(run["losses"]["train"]) + 1)
            plt.plot(
                epochs,
                run["losses"]["train"],
                color=c,
                linestyle="-",
                label=f"train wd={wd}",
            )
            plt.plot(
                epochs,
                run["losses"]["val"],
                color=c,
                linestyle="--",
                label=f"val   wd={wd}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH Training vs Validation Loss\nLeft: Train (solid), Right: Val (dashed)"
        )
        plt.legend(fontsize="small")
        f_name = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(f_name, dpi=150)
        plt.close()
        print(f"Saved {f_name}")
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --------- plot 2: HWA across epochs ---------
    try:
        plt.figure()
        for (wd, run), c in zip(runs.items(), colors):
            hwa = [m["hwa"] for m in run["metrics"]["val"]]
            plt.plot(range(1, len(hwa) + 1), hwa, color=c, label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy (HWA)")
        plt.title("SPR_BENCH Validation HWA Across Epochs")
        plt.legend(fontsize="small")
        f_name = os.path.join(working_dir, "SPR_BENCH_HWA_epoch_curves.png")
        plt.savefig(f_name, dpi=150)
        plt.close()
        print(f"Saved {f_name}")
    except Exception as e:
        print(f"Error creating HWA curves: {e}")
        plt.close()

    # --------- plot 3: final HWA vs weight decay ---------
    try:
        plt.figure()
        wd_vals, final_hwa = [], []
        for wd, run in runs.items():
            wd_vals.append(float(wd))
            final_hwa.append(run["metrics"]["val"][-1]["hwa"])
        order = np.argsort(wd_vals)
        wd_sorted = np.array(wd_vals)[order]
        hwa_sorted = np.array(final_hwa)[order]
        plt.bar(
            range(len(wd_sorted)), hwa_sorted, tick_label=[f"{w:g}" for w in wd_sorted]
        )
        plt.xlabel("Weight Decay")
        plt.ylabel("Final-Epoch HWA")
        plt.title("SPR_BENCH Final HWA vs Weight Decay")
        f_name = os.path.join(working_dir, "SPR_BENCH_HWA_vs_weight_decay.png")
        plt.savefig(f_name, dpi=150)
        plt.close()
        print(f"Saved {f_name}")
    except Exception as e:
        print(f"Error creating HWA vs WD plot: {e}")
        plt.close()

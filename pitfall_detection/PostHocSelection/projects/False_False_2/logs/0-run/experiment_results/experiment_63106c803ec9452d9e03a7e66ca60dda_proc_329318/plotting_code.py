import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    # unpack once for convenience
    runs = experiment_data.get("weight_decay", {})
    wds = sorted(list(runs.keys()), key=float)  # keep numerical order
    # determine best run (highest final dev_PHA)
    best_wd, best_dev_pha = None, -1
    for wd in wds:
        dev_pha = runs[wd]["spr_bench"]["metrics"]["dev_PHA"][-1]
        if dev_pha > best_dev_pha:
            best_dev_pha, best_wd = dev_pha, wd

    # ---------- 1) PHA curves ----------
    try:
        plt.figure()
        for wd in wds:
            epochs = runs[wd]["spr_bench"]["epochs"]
            train_pha = runs[wd]["spr_bench"]["metrics"]["train_PHA"]
            dev_pha = runs[wd]["spr_bench"]["metrics"]["dev_PHA"]
            plt.plot(epochs, train_pha, "--", label=f"train_PHA wd={wd}")
            plt.plot(epochs, dev_pha, "-", label=f"dev_PHA wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("Training & Validation PHA vs Epochs (SPR_BENCH)")
        plt.legend(fontsize=6)
        fname = os.path.join(working_dir, "spr_bench_PHA_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating PHA curve plot: {e}")
        plt.close()

    # ---------- 2) Final dev_PHA per weight_decay ----------
    try:
        plt.figure()
        dev_values = [runs[wd]["spr_bench"]["metrics"]["dev_PHA"][-1] for wd in wds]
        plt.bar(range(len(wds)), dev_values, tick_label=wds)
        plt.xlabel("Weight Decay")
        plt.ylabel("Final Dev PHA")
        plt.title("Final Dev PHA by Weight Decay (SPR_BENCH)")
        fname = os.path.join(working_dir, "spr_bench_final_dev_PHA.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating dev PHA bar plot: {e}")
        plt.close()

    # ---------- 3) Final test metrics ----------
    try:
        plt.figure()
        test_pha = [runs[wd]["spr_bench"]["test_metrics"]["PHA"] for wd in wds]
        test_swa = [runs[wd]["spr_bench"]["test_metrics"]["SWA"] for wd in wds]
        test_cwa = [runs[wd]["spr_bench"]["test_metrics"]["CWA"] for wd in wds]
        bar_w = 0.25
        x = np.arange(len(wds))
        plt.bar(x - bar_w, test_swa, width=bar_w, label="SWA")
        plt.bar(x, test_cwa, width=bar_w, label="CWA")
        plt.bar(x + bar_w, test_pha, width=bar_w, label="PHA")
        plt.xticks(x, wds)
        plt.xlabel("Weight Decay")
        plt.ylabel("Score")
        plt.title("Test Metrics by Weight Decay (SPR_BENCH)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()

    # ---------- 4) Loss curves for best run ----------
    try:
        plt.figure()
        best_run = runs[best_wd]["spr_bench"]
        epochs = best_run["epochs"]
        plt.plot(epochs, best_run["losses"]["train"], label="Train Loss")
        plt.plot(epochs, best_run["losses"]["dev"], label="Dev Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Loss Curves (Best wd={best_wd}) – SPR_BENCH")
        plt.legend()
        fname = os.path.join(
            working_dir, f"spr_bench_loss_curves_best_wd_{best_wd}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- console summary ----------
    best_test = runs[best_wd]["spr_bench"]["test_metrics"]
    print(f"Best weight_decay={best_wd} | Test metrics: {best_test}")

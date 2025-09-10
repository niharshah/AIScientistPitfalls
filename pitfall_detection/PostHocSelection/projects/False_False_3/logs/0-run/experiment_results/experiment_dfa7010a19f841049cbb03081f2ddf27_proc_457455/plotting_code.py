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
    runs = experiment_data["max_grad_norm"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

if runs:
    clips = sorted(runs.keys(), key=float)
    epochs = len(runs[clips[0]]["losses"]["train"])

    # ------------- Figure 1 : train / val loss curves -------------
    try:
        plt.figure()
        for c in clips:
            plt.plot(
                range(1, epochs + 1),
                runs[c]["losses"]["train"],
                label=f"train (clip={c})",
                linestyle="-",
            )
            plt.plot(
                range(1, epochs + 1),
                runs[c]["losses"]["val"],
                label=f"val   (clip={c})",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nSolid: Train, Dashed: Validation")
        plt.legend(fontsize=6, ncol=2)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------- Figure 2 : test weighted accuracies -------------
    try:
        ind = np.arange(len(clips))
        width = 0.25
        swa = [runs[c]["test_metrics"]["SWA"] for c in clips]
        cwa = [runs[c]["test_metrics"]["CWA"] for c in clips]
        hwa = [runs[c]["test_metrics"]["HWA"] for c in clips]

        plt.figure(figsize=(8, 4))
        plt.bar(ind - width, swa, width, label="SWA")
        plt.bar(ind, cwa, width, label="CWA")
        plt.bar(ind + width, hwa, width, label="HWA")
        plt.xticks(ind, clips)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Test Weighted Accuracies\nEffect of max_grad_norm")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_test_weighted_accuracies.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy plot: {e}")
        plt.close()

    # ------------- Figure 3 : val HWA over epochs (clip 0 vs 5) -------------
    try:
        focus = [clips[0], clips[-1]]  # smallest and largest clipping norms
        plt.figure()
        for c in focus:
            hwas = [m["HWA"] for m in runs[c]["metrics"]["val"]]
            plt.plot(range(1, epochs + 1), hwas, label=f"clip={c}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation HWA")
        plt.title(
            "SPR_BENCH Validation HWA Over Epochs\nComparing Extreme Clipping Values"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_over_epochs.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating val HWA plot: {e}")
        plt.close()

    # ---------- print best metrics ----------
    best_clip = max(clips, key=lambda c: runs[c]["test_metrics"]["HWA"])
    best = runs[best_clip]["test_metrics"]
    print(f"Best clip value: {best_clip}")
    print(
        f"Test metrics -> Loss: {best['loss']:.4f}, SWA: {best['SWA']:.4f}, "
        f"CWA: {best['CWA']:.4f}, HWA: {best['HWA']:.4f}"
    )

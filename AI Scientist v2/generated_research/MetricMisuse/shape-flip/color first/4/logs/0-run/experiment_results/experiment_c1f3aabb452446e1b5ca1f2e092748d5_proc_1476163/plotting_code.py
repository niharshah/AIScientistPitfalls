import matplotlib.pyplot as plt
import numpy as np
import os

# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_dict = experiment_data.get("batch_size_tuning", {})
if not bs_dict:
    print("No batch_size_tuning data found, aborting plots.")
    exit()

# gather summaries ----------------------------------------------------
tags = sorted(bs_dict.keys(), key=lambda t: int(t[2:]))  # e.g. 'bs16'
epochs = len(next(iter(bs_dict.values()))["losses"]["train"])
cwa2_final = {t: bs_dict[t]["metrics"]["val_cwa2"][-1] for t in tags}
best_tag = max(cwa2_final, key=cwa2_final.get)

# 1) train / val loss -------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        ep = range(1, epochs + 1)
        plt.plot(
            ep, bs_dict[tag]["losses"]["train"], label=f"{tag} train", linestyle="-"
        )
        plt.plot(ep, bs_dict[tag]["losses"]["val"], label=f"{tag} val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs. Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) validation CWA2 --------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(range(1, epochs + 1), bs_dict[tag]["metrics"]["val_cwa2"], label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Validation CWA2")
    plt.title("SPR_BENCH: Validation Complexity-Weighted Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_cwa2_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 curve plot: {e}")
    plt.close()

# 3) final CWA2 bar chart --------------------------------------------
try:
    plt.figure()
    plt.bar(range(len(tags)), [cwa2_final[t] for t in tags], tick_label=tags)
    plt.ylabel("Final Validation CWA2")
    plt.title("SPR_BENCH: Final CWA2 vs Batch Size")
    plt.savefig(os.path.join(working_dir, "spr_bench_final_cwa2_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 bar plot: {e}")
    plt.close()

# 4) confusion matrix for best model ---------------------------------
try:
    gt = np.array(bs_dict[best_tag]["ground_truth"])
    pr = np.array(bs_dict[best_tag]["predictions"])
    n_cls = max(gt.max(), pr.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gt, pr):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix\nBest Batch Size: {best_tag}")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    plt.savefig(os.path.join(working_dir, f"spr_bench_confusion_{best_tag}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# final summary print -------------------------------------------------
print(
    f"Best batch size by final CWA2: {best_tag} "
    f"with score {cwa2_final[best_tag]:.4f}"
)

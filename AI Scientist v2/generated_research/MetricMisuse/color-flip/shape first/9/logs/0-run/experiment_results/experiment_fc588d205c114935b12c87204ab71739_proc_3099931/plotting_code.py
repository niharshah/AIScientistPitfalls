import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- paths --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["gradient_clip_norm"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# ---------------- summarise metrics -------------
summary = {}
for tag, entry in exp.items():
    vals = entry.get("metrics", {}).get("val", [])
    summary[tag] = vals[-1] if vals else None
print("Final CWA-2D per clip setting:", summary)

# ------------------- plot 1 ---------------------
try:
    plt.figure()
    for tag, entry in exp.items():
        val_losses = entry.get("losses", {}).get("val", [])
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label=tag)
    plt.title("Validation Loss across Gradient Clip Norms\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_vs_clip.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# ------------------- plot 2 ---------------------
try:
    plt.figure()
    tags, cwas = zip(*[(k, v) for k, v in summary.items() if v is not None])
    plt.bar(tags, cwas)
    plt.title("Final CWA-2D by Gradient Clip Norm\nDataset: SPR_BENCH")
    plt.ylabel("CWA-2D")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_cwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA bar plot: {e}")
    plt.close()

# ------------------- plot 3 ---------------------
try:
    # choose best tag by highest CWA
    best_tag = max(
        (t for t in summary if summary[t] is not None), key=lambda t: summary[t]
    )
    best_entry = exp[best_tag]
    tr_losses = best_entry.get("losses", {}).get("train", [])
    val_losses = best_entry.get("losses", {}).get("val", [])
    plt.figure()
    if tr_losses:
        plt.plot(range(1, len(tr_losses) + 1), tr_losses, label="train")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.title(
        f"Best Model Loss Curves (Train vs Val)\nDataset: SPR_BENCH, Setting: {best_tag}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_best_loss_curves_{best_tag}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best-model loss plot: {e}")
    plt.close()

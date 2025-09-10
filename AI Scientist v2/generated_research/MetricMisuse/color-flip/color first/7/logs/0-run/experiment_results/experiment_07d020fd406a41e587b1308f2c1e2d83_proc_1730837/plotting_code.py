import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_runs = experiment_data.get("weight_decay_tuning", {})
if not wd_runs:
    print("No weight-decay runs found, nothing to plot.")
    exit()

# ------------------------------------------------------------------
# Gather metrics and print summary
final_val_cpx = {}
for tag, rec in wd_runs.items():
    val_cpx_curve = [m["cpx"] for m in rec["metrics"]["val"]]
    final_val_cpx[tag] = val_cpx_curve[-1]

print("Final Validation CpxWA per weight decay:")
for tag, score in sorted(final_val_cpx.items(), key=lambda x: x[0]):
    print(f"{tag:>10}: {score:.4f}")

best_tag = max(final_val_cpx, key=final_val_cpx.get)

# ------------------------------------------------------------------
# 1) Training loss curves
try:
    plt.figure()
    for tag, rec in wd_runs.items():
        plt.plot(rec["epochs"], rec["losses"]["train"], marker="o", label=tag)
    plt.title("Training Loss vs Epochs\nDataset: Synthetic SPR", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_train_loss_by_wd.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Validation CpxWA curves
try:
    plt.figure()
    for tag, rec in wd_runs.items():
        val_cpx_curve = [m["cpx"] for m in rec["metrics"]["val"]]
        plt.plot(rec["epochs"], val_cpx_curve, marker="o", label=tag)
    plt.title("Validation CpxWA vs Epochs\nDataset: Synthetic SPR", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_val_cpxwa_by_wd.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-cpxwa plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Bar chart of final CpxWA
try:
    plt.figure()
    tags, scores = zip(*sorted(final_val_cpx.items(), key=lambda x: x[0]))
    plt.bar(tags, scores, color="skyblue")
    plt.title(
        "Final Validation CpxWA per Weight Decay\nDataset: Synthetic SPR", fontsize=10
    )
    plt.ylabel("CpxWA")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "synthetic_final_cpxwa_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) Train vs Val CpxWA for best run
try:
    plt.figure()
    rec = wd_runs[best_tag]
    train_curve = [m["cpx"] for m in rec["metrics"]["train"]]
    val_curve = [m["cpx"] for m in rec["metrics"]["val"]]
    plt.plot(rec["epochs"], train_curve, marker="o", label="Train")
    plt.plot(rec["epochs"], val_curve, marker="s", label="Validation")
    plt.title(f"Train vs Val CpxWA for {best_tag}\nDataset: Synthetic SPR", fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, f"synthetic_best_{best_tag}_train_val_cpxwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best-run plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 5) Scatter of final ColorWA vs ShapeWA
try:
    plt.figure()
    for tag, rec in wd_runs.items():
        col_wa = rec["metrics"]["val"][-1]["cwa"]
        shp_wa = rec["metrics"]["val"][-1]["swa"]
        plt.scatter(col_wa, shp_wa, label=tag)
        plt.annotate(tag, (col_wa, shp_wa))
    plt.title(
        "Final Validation Color vs Shape Weighted Accuracy\nDataset: Synthetic SPR",
        fontsize=10,
    )
    plt.xlabel("Color-Weighted Acc")
    plt.ylabel("Shape-Weighted Acc")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "synthetic_color_vs_shape_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()

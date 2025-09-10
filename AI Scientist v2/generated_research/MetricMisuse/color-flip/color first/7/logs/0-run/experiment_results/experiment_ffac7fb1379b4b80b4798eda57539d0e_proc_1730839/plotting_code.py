import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = list(experiment_data.get("dropout_prob", {}).keys())
colors = plt.cm.tab10.colors  # up to 10 distinct colors

# ---------------------------------------------------------------------
# 1) Training loss curves
try:
    plt.figure()
    for i, tag in enumerate(tags):
        epochs = experiment_data["dropout_prob"][tag]["epochs"]
        loss_tr = experiment_data["dropout_prob"][tag]["losses"]["train"]
        plt.plot(epochs, loss_tr, label=tag, color=colors[i % len(colors)])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Synthetic_SPR: Training Loss vs Epoch for different Dropout p")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_spr_dropout_prob_training_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Validation CpxWA curves
try:
    plt.figure()
    for i, tag in enumerate(tags):
        epochs = experiment_data["dropout_prob"][tag]["epochs"]
        val_metrics = experiment_data["dropout_prob"][tag]["metrics"]["val"]
        cpx_vals = [m["cpx"] for m in val_metrics]
        plt.plot(epochs, cpx_vals, label=tag, color=colors[i % len(colors)])
    plt.xlabel("Epoch")
    plt.ylabel("Validation CpxWA")
    plt.title("Synthetic_SPR: Validation CpxWA vs Epoch for different Dropout p")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_spr_dropout_prob_val_cpxwa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation CpxWA plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Final Validation CpxWA comparison
try:
    plt.figure()
    ps, finals = [], []
    for i, tag in enumerate(tags):
        p_val = float(tag.split("_")[1])
        final_cpx = experiment_data["dropout_prob"][tag]["metrics"]["val"][-1]["cpx"]
        ps.append(p_val)
        finals.append(final_cpx)
    plt.plot(ps, finals, marker="o")
    plt.xlabel("Dropout probability p")
    plt.ylabel("Final Validation CpxWA")
    plt.title("Synthetic_SPR: Final Validation CpxWA vs Dropout p")
    fname = os.path.join(working_dir, "synthetic_spr_dropout_prob_final_val_cpxwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final CpxWA comparison plot: {e}")
    plt.close()

print(f"Plots saved to {working_dir}")

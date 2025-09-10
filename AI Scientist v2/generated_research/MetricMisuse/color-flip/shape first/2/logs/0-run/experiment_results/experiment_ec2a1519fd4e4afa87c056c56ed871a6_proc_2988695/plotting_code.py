import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

final_hwa = {}

# -------- per-batch-size plots (max 4) ----------
for bs_key in sorted(experiment_data.get("batch_size", {}).keys(), key=int)[:4]:
    try:
        record = experiment_data["batch_size"][bs_key]
        tr_loss = record["losses"]["train"]
        val_loss = record["losses"]["val"]
        val_metrics = record["metrics"]["val"]
        hwa_curve = [m.get("hwa", np.nan) for m in val_metrics]

        epochs = range(1, len(tr_loss) + 1)
        final_hwa[int(bs_key)] = hwa_curve[-1] if hwa_curve else np.nan

        plt.figure(figsize=(10, 4))
        # Left subplot: losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Right subplot: HWA
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hwa_curve, marker="o", label="Val HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("HWA Curve")
        plt.legend()

        plt.suptitle(f"SPR_BENCH Batch Size {bs_key} | Left: Loss, Right: HWA")
        fname = f"SPR_BENCH_bs{bs_key}_loss_hwa.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {bs_key}: {e}")
        plt.close()

# -------- bar chart comparing final HWA ----------
try:
    plt.figure()
    bs_list = list(final_hwa.keys())
    hwa_vals = [final_hwa[b] for b in bs_list]
    plt.bar([str(b) for b in bs_list], hwa_vals, color="skyblue")
    for i, v in enumerate(hwa_vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xlabel("Batch Size")
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH Final HWA by Batch Size")
    fname = "SPR_BENCH_final_HWA_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# -------- print evaluation summary ----------
print("Final HWA per batch size:")
for bs, h in final_hwa.items():
    print(f"  BS={bs}: HWA={h:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_dict = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
metrics = spr_dict.get("metrics", {})
losses = spr_dict.get("losses", {})
y_pred = np.array(spr_dict.get("predictions", []))
y_true = np.array(spr_dict.get("ground_truth", []))

# ---------- 1) HWA curves ----------
try:
    plt.figure()
    for bs, run in metrics.items():
        ep, tr_hwa, dv_hwa = zip(*run)
        plt.plot(ep, tr_hwa, "--", label=f"train bs={bs}")
        plt.plot(ep, dv_hwa, "-", label=f"dev   bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH Harmonic Weighted Accuracy\nTrain vs Validation")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------- 2) Loss curves ----------
try:
    plt.figure()
    for bs, run in losses.items():
        ep, tr_l, dv_l = zip(*run)
        plt.plot(ep, tr_l, "--", label=f"train bs={bs}")
        plt.plot(ep, dv_l, "-", label=f"dev   bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_Loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# ---------- 3) Confusion matrix ----------
try:
    if y_true.size and y_pred.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    va="center",
                    ha="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1]), ax.set_yticks([0, 1])
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_Confusion_Matrix.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Predictions or ground truth missing—skipping confusion matrix.")
except Exception as e:
    print(f"Error creating Confusion Matrix plot: {e}")
    plt.close()

print("Plotting complete. Files saved in:", working_dir)

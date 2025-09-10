import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def get_epochs(tag_dict):
    return list(range(1, len(tag_dict["losses"]["train"]) + 1))


lrs = sorted(experiment_data.get("learning_rate", {}).keys())

# ---------- Figure 1: loss curves ----------
try:
    plt.figure(figsize=(7, 5))
    for tag in lrs:
        ep = get_epochs(experiment_data["learning_rate"][tag])
        plt.plot(
            ep,
            experiment_data["learning_rate"][tag]["losses"]["train"],
            label=f"{tag}-train",
            linestyle="-",
        )
        plt.plot(
            ep,
            experiment_data["learning_rate"][tag]["losses"]["val"],
            label=f"{tag}-val",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR-BENCH: Training vs Validation Loss (Learning-Rate Sweep)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr-bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Figure 2: HWA curves ----------
try:
    plt.figure(figsize=(7, 5))
    for tag in lrs:
        ep = get_epochs(experiment_data["learning_rate"][tag])
        hwa = [
            m["hwa"] for m in experiment_data["learning_rate"][tag]["metrics"]["val"]
        ]
        plt.plot(ep, hwa, marker="o", label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR-BENCH: Validation HWA over Epochs (Learning-Rate Sweep)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr-bench_hwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- Figure 3: final HWA bar plot ----------
try:
    plt.figure(figsize=(6, 4))
    final_hwa = [
        experiment_data["learning_rate"][tag]["metrics"]["val"][-1]["hwa"]
        for tag in lrs
    ]
    plt.bar(range(len(lrs)), final_hwa, tick_label=lrs)
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR-BENCH: Final Validation HWA vs Learning-Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr-bench_final_hwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

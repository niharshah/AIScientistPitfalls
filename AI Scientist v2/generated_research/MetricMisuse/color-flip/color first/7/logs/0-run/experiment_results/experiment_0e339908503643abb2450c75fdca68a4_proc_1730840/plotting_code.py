import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

sched_data = experiment_data.get("learning_rate_scheduler", {})
names = list(sched_data.keys())

# 1) Validation CpxWA curves ---------------------------------------------------
try:
    plt.figure()
    for name in names:
        rec = sched_data[name]
        epochs = rec.get("epochs", [])
        vals = [m["cpx"] for m in rec.get("metrics", {}).get("val", [])]
        if epochs and vals:
            plt.plot(epochs, vals, marker="o", label=name)
    plt.title(
        "Validation Complexity-Weighted Accuracy\nDataset: learning_rate_scheduler"
    )
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fn = os.path.join(working_dir, "learning_rate_scheduler_val_cpxwa_curve.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating val CpxWA plot: {e}")
    plt.close()

# 2) Training loss curves ------------------------------------------------------
try:
    plt.figure()
    for name in names:
        rec = sched_data[name]
        epochs = rec.get("epochs", [])
        tr_losses = rec.get("losses", {}).get("train", [])
        if epochs and tr_losses:
            plt.plot(epochs, tr_losses, marker="o", label=name)
    plt.title("Training Loss\nDataset: learning_rate_scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fn = os.path.join(working_dir, "learning_rate_scheduler_train_loss_curve.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# 3) Learning-rate schedules ---------------------------------------------------
try:
    plt.figure()
    for name in names:
        rec = sched_data[name]
        epochs = rec.get("epochs", [])
        lrs = rec.get("lr", [])
        if epochs and lrs:
            plt.plot(epochs, lrs, marker="o", label=name)
    plt.title("Learning-Rate Schedule\nDataset: learning_rate_scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    fn = os.path.join(working_dir, "learning_rate_scheduler_lr_curve.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating LR plot: {e}")
    plt.close()

# 4) Best validation CpxWA summary --------------------------------------------
try:
    best_scores = []
    for name in names:
        vals = [m["cpx"] for m in sched_data[name]["metrics"]["val"]]
        best_scores.append(max(vals) if vals else 0)
    if best_scores:
        plt.figure()
        plt.bar(names, best_scores, color="skyblue")
        plt.title(
            "Best Validation CpxWA per Scheduler\nDataset: learning_rate_scheduler"
        )
        plt.ylabel("Best CpxWA")
        plt.xticks(rotation=45, ha="right")
        fn = os.path.join(working_dir, "learning_rate_scheduler_best_val_cpxwa_bar.png")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
except Exception as e:
    print(f"Error creating best-score plot: {e}")
    plt.close()

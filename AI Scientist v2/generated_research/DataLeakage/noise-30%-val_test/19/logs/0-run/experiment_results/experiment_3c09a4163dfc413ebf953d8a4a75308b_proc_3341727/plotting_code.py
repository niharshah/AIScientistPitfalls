import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper
variants = ["sinusoidal", "learned"]

# ------------------- loss curves -------------------
for variant in variants:
    try:
        exp = experiment_data["positional_encoding"][variant]
        train_loss = exp["losses"]["train"]
        val_loss = exp["losses"]["val"]

        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Validation")
        plt.xlabel("Updates (epochs aggregated)")
        plt.ylabel("BCE Loss")
        plt.title(f"SPR_BENCH Loss Curve\nVariant: {variant.capitalize()}")
        plt.legend()
        fname = f"spr_loss_curve_{variant}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve for {variant}: {e}")
        plt.close()

# ------------------- test metric comparison -------------------
try:
    metrics = ["test_MCC", "test_F1"]
    width = 0.35
    x = np.arange(len(variants))

    for idx, metric in enumerate(metrics):
        plt.figure(figsize=(5, 4))
        vals = [experiment_data["positional_encoding"][v][metric] for v in variants]
        plt.bar(x, vals, width, color=["steelblue", "orange"])
        plt.xticks(x, [v.capitalize() for v in variants])
        plt.ylabel(metric)
        plt.title(f"SPR_BENCH Test {metric} Comparison")
        fname = f"spr_test_{metric.lower()}_comparison.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test metric comparison plot: {e}")
    plt.close()

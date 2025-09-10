import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = ["baseline", "neurosymbolic"]

# ---- per-tag curves ----
for tag in tags:
    stats = experiment_data.get(tag, {})
    epochs = np.arange(1, len(stats.get("losses", {}).get("train", [])) + 1)

    # 1) loss curve
    try:
        plt.figure()
        plt.plot(epochs, stats["losses"]["train"], label="Train")
        plt.plot(epochs, stats["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH {tag} Loss (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"spr_{tag}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {tag}: {e}")
        plt.close()

    # 2) macro-F1 curve
    try:
        plt.figure()
        plt.plot(epochs, stats["metrics"]["train"], label="Train")
        plt.plot(epochs, stats["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"SPR_BENCH {tag} Macro-F1 (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"spr_{tag}_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {tag}: {e}")
        plt.close()

# ---- test performance bar chart ----
try:
    plt.figure()
    test_scores = [experiment_data[t]["test_macroF1"] for t in tags]
    plt.bar(tags, test_scores, color=["skyblue", "salmon"])
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Test Macro-F1 Comparison")
    for i, v in enumerate(test_scores):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
    plt.savefig(os.path.join(working_dir, "spr_test_f1_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test comparison plot: {e}")
    plt.close()

# ---- numeric summary ----
for t in tags:
    print(
        f"{t:14s}: Test Macro-F1 = {experiment_data.get(t, {}).get('test_macroF1', np.nan):.4f}"
    )

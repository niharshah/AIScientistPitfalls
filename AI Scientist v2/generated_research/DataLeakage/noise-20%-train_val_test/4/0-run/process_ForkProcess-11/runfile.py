import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dropout_dict = experiment_data.get("dropout_tuning", {})
tags = list(dropout_dict.keys())[:5]  # plot at most first 5 dropouts


# Helper to fetch arrays
def get_arr(tag, key1, key2):
    return np.asarray(dropout_dict[tag][key1][key2])


# 1) Loss curves
try:
    plt.figure(figsize=(10, 4))
    # Left subplot: training loss
    plt.subplot(1, 2, 1)
    for tag in tags:
        plt.plot(
            dropout_dict[tag]["epochs"], get_arr(tag, "losses", "train"), label=tag
        )
    plt.title("Left: Training Loss - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)
    # Right subplot: validation loss
    plt.subplot(1, 2, 2)
    for tag in tags:
        plt.plot(dropout_dict[tag]["epochs"], get_arr(tag, "losses", "val"), label=tag)
    plt.title("Right: Validation Loss - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) F1 curves
try:
    plt.figure(figsize=(10, 4))
    # Left subplot: training F1
    plt.subplot(1, 2, 1)
    for tag in tags:
        plt.plot(
            dropout_dict[tag]["epochs"], get_arr(tag, "metrics", "train_f1"), label=tag
        )
    plt.title("Left: Training Macro-F1 - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=6)
    # Right subplot: validation F1
    plt.subplot(1, 2, 2)
    for tag in tags:
        plt.plot(
            dropout_dict[tag]["epochs"], get_arr(tag, "metrics", "val_f1"), label=tag
        )
    plt.title("Right: Validation Macro-F1 - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Test-set F1 per dropout
try:
    plt.figure()
    test_f1s = [dropout_dict[tag]["metrics"]["test_f1"] for tag in tags]
    plt.bar(range(len(tags)), test_f1s, tick_label=[t.split("_")[-1] for t in tags])
    best_overall = experiment_data.get("best_overall", {})
    best_f1 = best_overall.get("test_f1", None)
    if best_f1 is not None:
        plt.axhline(
            best_f1, color="r", linestyle="--", label=f"Best Overall = {best_f1:.3f}"
        )
    plt.title("Test Macro-F1 by Dropout - SPR_BENCH")
    plt.xlabel("Dropout")
    plt.ylabel("Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test-F1 bar: {e}")
    plt.close()

# print best overall metric
bo = experiment_data.get("best_overall", {})
if bo:
    print(
        f"Best Dropout: {bo.get('dropout')} | Final Test Macro-F1: {bo.get('test_f1'):.4f}"
    )

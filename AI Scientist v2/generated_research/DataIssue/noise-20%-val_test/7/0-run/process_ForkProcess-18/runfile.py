import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# Setup & data loading
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None:
    quit()

sub = experiment_data.get("train_subsample", {})
tags = sorted(sub.keys(), key=lambda t: int(t.replace("pct", "")))
fractions = [int(t.replace("pct", "")) for t in tags]


# ------------------------------------------------------------
# Helper to fetch metric list
# ------------------------------------------------------------
def m(tag, key):
    return sub[tag]["metrics"][key]


# ------------------------------------------------------------
# 1. Accuracy curves
# ------------------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(m(tag, "train_acc"), label=f"{tag} train")
        plt.plot(m(tag, "val_acc"), label=f"{tag} val", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 2. Loss curves
# ------------------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(m(tag, "val_loss"), label=f"{tag} val")
        plt.plot(sub[tag]["losses"]["train"], label=f"{tag} train", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy Loss")
    plt.title("Training & Validation Loss (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 3. Validation RCA curves
# ------------------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(m(tag, "val_rca"), label=f"{tag}")
    plt.xlabel("Iteration")
    plt.ylabel("RCA")
    plt.title("Validation Rule-Consistency Accuracy (SPR_BENCH)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_rca_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RCA plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 4. Test accuracy bar chart
# ------------------------------------------------------------
try:
    test_accs = [sub[tag]["test_acc"] for tag in tags]
    plt.figure()
    plt.bar(range(len(tags)), test_accs, tick_label=fractions)
    plt.xlabel("Training fraction (%)")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Training Fraction (SPR_BENCH)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 5. Test RCA bar chart
# ------------------------------------------------------------
try:
    test_rca = [sub[tag]["test_rca"] for tag in tags]
    plt.figure()
    plt.bar(range(len(tags)), test_rca, tick_label=fractions, color="orange")
    plt.xlabel("Training fraction (%)")
    plt.ylabel("RCA")
    plt.title("Test Rule-Consistency Accuracy vs Training Fraction (SPR_BENCH)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_rca.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-RCA plot: {e}")
    plt.close()

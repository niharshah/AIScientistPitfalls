import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- LOAD DATA ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["dropout_prob"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}

# Helper: sorted dropout keys in numeric order
dropouts = (
    sorted(spr_data.keys(), key=lambda k: float(k.split("_")[1])) if spr_data else []
)

# ---------------- PLOT 1: Train Acc ----------------
try:
    plt.figure()
    for tag in dropouts:
        epochs = range(1, len(spr_data[tag]["metrics"]["train_acc"]) + 1)
        plt.plot(
            epochs,
            spr_data[tag]["metrics"]["train_acc"],
            label=f"p={tag.split('_')[1]}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training Accuracy vs Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train accuracy plot: {e}")
    plt.close()

# ---------------- PLOT 2: Val Acc ----------------
try:
    plt.figure()
    for tag in dropouts:
        epochs = range(1, len(spr_data[tag]["metrics"]["val_acc"]) + 1)
        plt.plot(
            epochs, spr_data[tag]["metrics"]["val_acc"], label=f"p={tag.split('_')[1]}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Validation Accuracy vs Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val accuracy plot: {e}")
    plt.close()

# ---------------- PLOT 3: Test Accuracy vs Dropout ----------------
try:
    plt.figure()
    x = [float(tag.split("_")[1]) for tag in dropouts]
    y = [spr_data[tag]["test_acc"] for tag in dropouts]
    plt.plot(x, y, marker="o")
    plt.xlabel("Dropout Probability")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Test Accuracy vs Dropout")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_vs_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()

# ---------------- PLOT 4: Fidelity & FAGM vs Dropout ----------------
try:
    plt.figure()
    x = [float(tag.split("_")[1]) for tag in dropouts]
    fidelity = [spr_data[tag]["fidelity"] for tag in dropouts]
    fagm = [spr_data[tag]["fagm"] for tag in dropouts]
    plt.plot(x, fidelity, marker="s", label="Fidelity")
    plt.plot(x, fagm, marker="^", label="FAGM")
    plt.xlabel("Dropout Probability")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Fidelity & FAGM vs Dropout")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_fidelity_fagm_vs_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating fidelity/FAGM plot: {e}")
    plt.close()

print("Finished plotting metrics.")

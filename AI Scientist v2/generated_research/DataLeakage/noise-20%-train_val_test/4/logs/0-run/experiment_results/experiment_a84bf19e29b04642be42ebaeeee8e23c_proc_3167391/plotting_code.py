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

tag = "char_count_transformer"
run = experiment_data.get(tag, {})

epochs = run.get("epochs", [])
tr_loss = run.get("losses", {}).get("train", [])
val_loss = run.get("losses", {}).get("val", [])
tr_f1 = run.get("metrics", {}).get("train_f1", [])
val_f1 = run.get("metrics", {}).get("val_f1", [])
test_f1 = run.get("metrics", {}).get("test_f1", None)

# 1) Loss curves
try:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, tr_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("Loss Curves - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) F1 curves
try:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, tr_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.title("Macro-F1 Curves - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Test-set F1
try:
    plt.figure()
    plt.bar([0], [test_f1], tick_label=["Test"])
    plt.title("Test Macro-F1 - SPR_BENCH")
    plt.ylabel("Macro-F1")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test-F1 plot: {e}")
    plt.close()

# console output
if test_f1 is not None:
    print(f"Final Test Macro-F1: {test_f1:.4f}")

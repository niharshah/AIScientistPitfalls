import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    noise_dict = experiment_data["label_noise_robustness"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    noise_dict = {}

# Gather stats
noise_levels, acc_train, acc_val, acc_test, val_loss = [], [], [], [], []
for ds_key, rec in sorted(noise_dict.items(), key=lambda x: x[1]["noise_level"]):
    n = rec["noise_level"]
    noise_levels.append(n)
    acc_train.append(rec["metrics"]["train"][0])
    acc_val.append(rec["metrics"]["val"][0])
    acc_test.append(rec["metrics"]["test"][0])
    val_loss.append(rec["losses"]["val"][0])

# Print metrics table
print("Noise  TrainAcc  ValAcc  TestAcc  ValLoss")
for n, at, av, ats, vl in zip(noise_levels, acc_train, acc_val, acc_test, val_loss):
    print(f"{n:4.1f}  {at:8.3f} {av:7.3f} {ats:8.3f} {vl:8.3f}")

# -------------------- Plot 1: Accuracy vs noise --------------------
try:
    plt.figure()
    plt.plot(noise_levels, acc_train, "-o", label="Train")
    plt.plot(noise_levels, acc_val, "-s", label="Validation")
    plt.plot(noise_levels, acc_test, "-^", label="Test")
    plt.xlabel("Label noise fraction")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Accuracy vs Label Noise")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_noise.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- Plot 2: Val loss vs noise --------------------
try:
    plt.figure()
    plt.plot(noise_levels, val_loss, "-o", color="orange")
    plt.xlabel("Label noise fraction")
    plt.ylabel("Validation log-loss")
    plt.title("SPR_BENCH – Validation Loss vs Label Noise")
    fname = os.path.join(working_dir, "SPR_BENCH_val_loss_vs_noise.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

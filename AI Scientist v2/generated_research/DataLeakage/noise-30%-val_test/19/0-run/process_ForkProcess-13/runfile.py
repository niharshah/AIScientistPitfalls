import os
import numpy as np

# ---------------- Load ----------------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# The experiments were stored under the single key "transformer"
exp = experiment_data["transformer"]

# Convenience handles
train_mcc_list = exp["metrics"]["train_MCC"]
val_mcc_list = exp["metrics"]["val_MCC"]
train_loss_list = exp["losses"]["train"]
val_loss_list = exp["losses"]["val"]

# ---------------- Report ----------------
# 1) Training set
print("Training set")
print(f"train Matthews correlation coefficient: {train_mcc_list[-1]:.4f}")
print(f"train binary cross-entropy loss: {train_loss_list[-1]:.4f}")

# 2) Validation set
print("\nValidation set")
print(f"validation Matthews correlation coefficient: {val_mcc_list[-1]:.4f}")
print(f"validation binary cross-entropy loss: {val_loss_list[-1]:.4f}")

# 3) Test set
print("\nTest set")
print(f"test Matthews correlation coefficient: {exp['test_MCC']:.4f}")
print(f"test macro F1 score: {exp['test_F1']:.4f}")

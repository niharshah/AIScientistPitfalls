import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["no_length_feature"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

loss_train = run["losses"]["train"]
loss_val = run["losses"]["val"]
swa_train = run["metrics"]["train_swa"]
swa_val = run["metrics"]["val_swa"]
test_swa = run["metrics"]["test_swa"]
y_true = np.array(run["ground_truth"])
y_pred = np.array(run["predictions"])

# ----------- evaluation print -----------
print(f"Final Test Shape-Weighted Accuracy (SWA): {test_swa:.3f}")

# Confusion matrix values
conf = np.zeros((2, 2), dtype=int)
for t, p in zip(y_true, y_pred):
    conf[t, p] += 1
print(f"Confusion matrix:\n{conf}")

# -------------- plotting ----------------
# 1) Loss curves
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) SWA curves
try:
    plt.figure()
    plt.plot(epochs, swa_train, label="Train SWA")
    plt.plot(epochs, swa_val, label="Val SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH SWA Curves (Left: Train, Right: Validation)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 3) Confusion matrix heat-map
try:
    plt.figure()
    plt.imshow(conf, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR_BENCH Confusion Matrix (True vs. Predicted)")
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 4) Correct vs Incorrect bar chart
try:
    plt.figure()
    correct = np.sum(y_true == y_pred)
    incorrect = len(y_true) - correct
    plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
    plt.ylabel("Count")
    plt.title("SPR_BENCH Test Prediction Quality (Correct vs. Incorrect)")
    plt.savefig(os.path.join(working_dir, "spr_bench_correct_incorrect.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

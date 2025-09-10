import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- Helper to compute accuracy ----------
def accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (y_true == y_pred).mean() if len(y_true) else 0.0


# ---------- Aggregate data ----------
batch_dict = experiment_data.get("batch_size", {})
epochs = None
loss_train, loss_val = {}, {}
swa, cwa, ais = {}, {}, {}
final_acc = {}

for bs, results in batch_dict.items():
    lt = results["losses"]["train"]
    lv = results["losses"]["val"]
    mt = results["metrics"]["train"]  # SWA
    mv = results["metrics"]["val"]  # CWA
    a = results["AIS"]
    loss_train[bs] = lt
    loss_val[bs] = lv
    swa[bs] = mt
    cwa[bs] = mv
    ais[bs] = a
    epochs = range(1, len(lt) + 1)
    final_acc[bs] = accuracy(results["ground_truth"], results["predictions"])

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for bs in loss_train:
        plt.plot(epochs, loss_train[bs], label=f"Train bs={bs}", linestyle="-")
        plt.plot(epochs, loss_val[bs], label=f"Val bs={bs}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR Toy Dataset – Training & Validation Loss\n( dashed = validation )")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Plot 2: Metric curves ----------
try:
    plt.figure(figsize=(6, 4))
    for bs in swa:
        plt.plot(epochs, swa[bs], label=f"SWA bs={bs}", linestyle="-")
        plt.plot(epochs, cwa[bs], label=f"CWA bs={bs}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR Toy Dataset – SWA (solid) & CWA (dashed)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_weighted_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- Plot 3: AIS curves ----------
try:
    plt.figure(figsize=(6, 4))
    for bs in ais:
        plt.plot(epochs, ais[bs], label=f"batch={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("AIS")
    plt.title("SPR Toy Dataset – Agreement Invariance Score (AIS)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_AIS_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()

# ---------- Print final validation accuracy ----------
print("Final validation accuracy by batch size:")
for bs, acc in final_acc.items():
    print(f"  Batch {bs}: {acc:.3f}")

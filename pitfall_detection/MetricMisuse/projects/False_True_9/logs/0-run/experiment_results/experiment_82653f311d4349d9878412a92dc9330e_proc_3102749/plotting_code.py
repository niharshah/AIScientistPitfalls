import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("SPR_BENCH", {})
if not data:
    print("SPR_BENCH data not found.")
    exit()

loss_train = data.get("losses", {}).get("train", [])
loss_val = data.get("losses", {}).get("val", [])
metrics = data.get("metrics", {}).get("val", [])
preds = data.get("predictions", [])
labels = data.get("ground_truth", [])


# ------------- helper ---------------------
def unzip(pairs, idx):
    return [p[idx] for p in pairs]


# ------------- plot 1: loss curves ---------
try:
    plt.figure()
    if loss_train:
        plt.plot(unzip(loss_train, 0), unzip(loss_train, 1), label="Train")
    if loss_val:
        plt.plot(unzip(loss_val, 0), unzip(loss_val, 1), "--", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: metric curves ----------
try:
    plt.figure()
    if metrics:
        epochs = unzip(metrics, 0)
        swa = unzip(metrics, 1)
        cwa = unzip(metrics, 2)
        dawa = unzip(metrics, 3)
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, dawa, label="DAWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- plot 3: final metric bars ------
try:
    plt.figure()
    if metrics:
        final = metrics[-1]
        names = ["SWA", "CWA", "DAWA"]
        vals = final[1:]
        plt.bar(names, vals, color=["steelblue", "seagreen", "salmon"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Final-Epoch Metrics")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metrics bar: {e}")
    plt.close()

# ---------- plot 4: confusion matrix -------
try:
    num_cls = len(set(labels))
    if preds and labels and num_cls <= 50:
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------ print summary ----------------
if metrics:
    print(f"Final DAWA score: {metrics[-1][3]:.4f}")

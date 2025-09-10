import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths & data loading ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

ds_name = "SPR_BENCH"
ds = experiment_data.get(ds_name, {})

# --- extract arrays ----------------------------------------------------------
train_loss = ds.get("losses", {}).get("train", [])
val_loss = ds.get("metrics", {}).get("val_loss", [])
train_acc = ds.get("metrics", {}).get("train_acc", [])
val_acc = ds.get("metrics", {}).get("val_acc", [])
preds = np.array(ds.get("predictions", []))
ground = np.array(ds.get("ground_truth", []))
rule_preds = np.array(ds.get("rule_preds", []))

# compute evaluation numbers --------------------------------------------------
test_acc = (preds == ground).mean() if ground.size else np.nan
fidelity = (rule_preds == preds).mean() if preds.size else np.nan
fagm = np.sqrt(test_acc * fidelity) if np.isfinite(test_acc * fidelity) else np.nan
print(
    f"Test accuracy: {test_acc:.4f} | Rule fidelity: {fidelity:.4f} | FAGM: {fagm:.4f}"
)

epochs = np.arange(1, len(train_loss) + 1)

# --- PLOT 1: loss curves -----------------------------------------------------
try:
    plt.figure()
    if train_loss:
        plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{ds_name} Training vs Validation Loss")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
except Exception as e:
    print(f"Error creating loss curves: {e}")
finally:
    plt.close()

# --- PLOT 2: accuracy curves -------------------------------------------------
try:
    plt.figure()
    if train_acc:
        plt.plot(epochs, train_acc, label="Train Acc")
    if val_acc:
        plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f"{ds_name} Training vs Validation Accuracy")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
finally:
    plt.close()

# --- PLOT 3: confusion matrix ------------------------------------------------
try:
    if ground.size and preds.size:
        classes = np.unique(np.concatenate([ground, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for g, p in zip(ground, preds):
            gi = np.where(classes == g)[0][0]
            pi = np.where(classes == p)[0][0]
            cm[gi, pi] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    plt.close()

# --- PLOT 4: class count comparison -----------------------------------------
try:
    if ground.size and preds.size:
        classes = np.unique(np.concatenate([ground, preds]))
        g_counts = [(ground == c).sum() for c in classes]
        p_counts = [(preds == c).sum() for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, g_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, p_counts, width, label="Predicted")
        plt.xticks(x, classes)
        plt.ylabel("Count")
        plt.title(f"{ds_name} Class Distribution\nLeft: Ground Truth, Right: Predicted")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_class_distribution.png"))
except Exception as e:
    print(f"Error creating class-distribution plot: {e}")
finally:
    plt.close()

# --- PLOT 5: summary bars (accuracy, fidelity, FAGM) -------------------------
try:
    plt.figure()
    metrics = [test_acc, fidelity, fagm]
    labels = ["Test Acc", "Fidelity", "FAGM"]
    plt.bar(labels, metrics, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.title(f"{ds_name} Summary Metrics")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_summary_metrics.png"))
except Exception as e:
    print(f"Error creating summary metrics plot: {e}")
finally:
    plt.close()

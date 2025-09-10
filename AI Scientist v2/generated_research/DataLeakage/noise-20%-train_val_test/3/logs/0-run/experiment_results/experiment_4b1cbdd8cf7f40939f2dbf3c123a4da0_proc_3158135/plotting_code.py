import matplotlib.pyplot as plt
import numpy as np
import os

# paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("nhead", {}).get("SPR_BENCH", {})
nheads = spr_data.get("values", [])
metrics = spr_data.get("metrics", {})
train_l = metrics.get("train_loss", [])
val_l = metrics.get("val_loss", [])
val_f1 = metrics.get("val_f1", [])
preds = spr_data.get("predictions", [])
gts = spr_data.get("ground_truth", [])
num_plots = 0


# helper to save plots
def save_fig(fig, name):
    fig.savefig(os.path.join(working_dir, name))
    plt.close(fig)


# 1. Train loss vs nhead
try:
    fig = plt.figure()
    plt.plot(nheads, train_l, marker="o")
    plt.xlabel("nhead")
    plt.ylabel("Train Loss")
    plt.title("SPR_BENCH: Final Train Loss vs nhead")
    save_fig(fig, "SPR_BENCH_train_loss_vs_nhead.png")
    num_plots += 1
except Exception as e:
    print(f"Error creating train-loss plot: {e}")
    plt.close()

# 2. Validation loss vs nhead
try:
    fig = plt.figure()
    plt.plot(nheads, val_l, marker="o", color="orange")
    plt.xlabel("nhead")
    plt.ylabel("Validation Loss")
    plt.title("SPR_BENCH: Final Validation Loss vs nhead")
    save_fig(fig, "SPR_BENCH_val_loss_vs_nhead.png")
    num_plots += 1
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# 3. Validation F1 vs nhead
try:
    fig = plt.figure()
    plt.plot(nheads, val_f1, marker="o", color="green")
    plt.xlabel("nhead")
    plt.ylabel("Validation Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1 vs nhead")
    save_fig(fig, "SPR_BENCH_val_f1_vs_nhead.png")
    num_plots += 1
except Exception as e:
    print(f"Error creating val-F1 plot: {e}")
    plt.close()

# 4. Confusion matrix on test set for best model
try:
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = max(gts.max(), preds.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Best nhead)")
        plt.xticks(range(n_cls))
        plt.yticks(range(n_cls))
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="red" if cm[i, j] > cm.max() * 0.7 else "black",
                )
        save_fig(fig, "SPR_BENCH_confusion_matrix.png")
        num_plots += 1
except Exception as e:
    print(f"Error creating confusion-matrix plot: {e}")
    plt.close()

# Print evaluation metric if available
if val_f1:
    best_idx = int(np.argmax(val_f1))
    print(f"Best nhead={nheads[best_idx]} | Dev Macro-F1={val_f1[best_idx]:.4f}")
if preds and gts:
    from sklearn.metrics import f1_score

    print(f'Test Macro-F1={f1_score(gts, preds, average="macro"):.4f}')

print(f"{num_plots} figure(s) saved to {working_dir}")

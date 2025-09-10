import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["embed_dim"]["SPR"]
    embed_dims = spr_data["embed_dims"]
    train_metrics = np.array(spr_data["metrics"]["train"])  # shape (n_embed, n_epoch)
    val_metrics = np.array(spr_data["metrics"]["val"])
    train_losses = np.array(spr_data["losses"]["train"])
    val_losses = np.array(spr_data["losses"]["val"])
    epochs = np.array(spr_data["epochs"])
    best_embed = spr_data["best_embed"]
    y_true = np.array(spr_data["ground_truth"])
    y_pred = np.array(spr_data["predictions"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# ---------------------------------------------------
# 1) CpxWA curves
try:
    plt.figure()
    for i, ed in enumerate(embed_dims):
        plt.plot(epochs, train_metrics[i], marker="o", label=f"Train ed={ed}")
        plt.plot(
            epochs, val_metrics[i], marker="x", linestyle="--", label=f"Val ed={ed}"
        )
    plt.title("SPR: Complexity-Weighted Accuracy across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend(fontsize="small")
    plt.savefig(os.path.join(working_dir, "SPR_cpxwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curves: {e}")
    plt.close()

# ---------------------------------------------------
# 2) Loss curves
try:
    plt.figure()
    for i, ed in enumerate(embed_dims):
        plt.plot(epochs, train_losses[i], marker="o", label=f"Train ed={ed}")
        plt.plot(
            epochs, val_losses[i], marker="x", linestyle="--", label=f"Val ed={ed}"
        )
    plt.title("SPR: Cross-Entropy Loss across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize="small")
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss curves: {e}")
    plt.close()

# ---------------------------------------------------
# 3) Final validation CpxWA bar plot
try:
    plt.figure()
    final_val = val_metrics[:, -1]
    plt.bar([str(ed) for ed in embed_dims], final_val, color="skyblue")
    plt.title("SPR: Final Validation CpxWA per embed_dim")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final Val CpxWA")
    for i, v in enumerate(final_val):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.savefig(os.path.join(working_dir, "SPR_final_val_cpxwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---------------------------------------------------
# 4) Confusion matrix for best model
try:
    n_cls = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR Confusion Matrix (best embed_dim={best_embed})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------------------------------------------
# Print evaluation metric
test_cpxwa = experiment_data["embed_dim"]["SPR"]["metrics"]["val"][
    embed_dims.index(best_embed)
][-1]
print(f"Best embed_dim={best_embed} | Test CpxWA={test_cpxwa:.4f}")

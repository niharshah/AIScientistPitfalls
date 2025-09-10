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
    ed = experiment_data["lr_tuning"]["SPR_BENCH"]
    train_hsca = np.array(ed["metrics"]["train"])
    val_hsca = np.array(ed["metrics"]["val"])
    train_loss = np.array(ed["losses"]["train"])
    val_loss = np.array(ed["losses"]["val"])
    lr_pairs = np.array(ed["lr_pairs"])  # shape (N,2)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    train_hsca = val_hsca = train_loss = val_loss = lr_pairs = np.array([])

# ------------------------------------------------------------------
# 1) HSCA curve
try:
    plt.figure()
    x = np.arange(len(train_hsca))
    plt.plot(x, train_hsca, label="Train HSCA")
    plt.plot(x, val_hsca, label="Val HSCA")
    plt.xlabel("Run index")
    plt.ylabel("HSCA")
    plt.title("SPR_BENCH – Hyper-parameter sweep HSCA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_HSCA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HSCA curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Loss curve
try:
    plt.figure()
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss, label="Train Loss")
    plt.plot(x, val_loss, label="Val Loss")
    plt.xlabel("Run index")
    plt.ylabel("Average CE Loss")
    plt.title("SPR_BENCH – Training/Validation Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_Loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Scatter of LR pairs vs Val HSCA
try:
    plt.figure()
    sc = plt.scatter(lr_pairs[:, 0], lr_pairs[:, 1], c=val_hsca, cmap="viridis", s=80)
    plt.colorbar(sc, label="Validation HSCA")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Pre-training LR")
    plt.ylabel("Supervised LR")
    plt.title("SPR_BENCH – LR grid search (colour = Val HSCA)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_LR_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating LR scatter: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print best result
if val_hsca.size:
    best_idx = val_hsca.argmax()
    best_pair = lr_pairs[best_idx]
    print(
        f"Best val HSCA={val_hsca[best_idx]:.4f} at pre_lr={best_pair[0]:.0e}, "
        f"sup_lr={best_pair[1]:.0e}"
    )

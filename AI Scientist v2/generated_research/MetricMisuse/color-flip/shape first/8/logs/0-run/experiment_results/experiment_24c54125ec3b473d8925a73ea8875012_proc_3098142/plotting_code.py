import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- setup -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    quit()

# ----------------- extract arrays -----------------
params = np.array(ed["params"])  # shape (N,2)
train_scwa = np.array(ed["metrics"]["train_SCWA"])  # (N,)
val_scwa = np.array(ed["metrics"]["val_SCWA"])  # (N,)
train_loss = np.array(ed["losses"]["train"])  # (N,)
val_loss = np.array(ed["losses"]["val"])  # (N,)

# handy labels for x-axis
labels = [f"pre={p:.0e}\nft={f:.0e}" for p, f in params]

# ----------------- 1) SCWA bar chart -----------------
try:
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, train_scwa, width, label="Train")
    plt.bar(x + width / 2, val_scwa, width, label="Validation")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("SCWA")
    plt.title("SPR_BENCH: Train vs Validation SCWA (Learning-rate Sweep)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_scwa_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCWA bar plot: {e}")
    plt.close()

# ----------------- 2) Loss bar chart -----------------
try:
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, train_loss, width, label="Train")
    plt.bar(x + width / 2, val_loss, width, label="Validation")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss (Learning-rate Sweep)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss bar plot: {e}")
    plt.close()

# ----------------- 3) SCWA correlation scatter -----------------
try:
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(train_scwa, val_scwa, c=params[:, 1], cmap="viridis", s=60)
    plt.colorbar(scatter, label="ft_lr")
    plt.xlabel("Train SCWA")
    plt.ylabel("Validation SCWA")
    plt.title("SPR_BENCH: SCWA Correlation (Color = ft_lr)")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_scwa_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCWA scatter plot: {e}")
    plt.close()

# ----------------- print best metric -----------------
best_idx = np.argmax(val_scwa)
best_val = val_scwa[best_idx]
best_pre, best_ft = params[best_idx]
print(f"Best Validation SCWA: {best_val:.4f} (pre_lr={best_pre}, ft_lr={best_ft})")

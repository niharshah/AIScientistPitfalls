import matplotlib.pyplot as plt
import numpy as np
import os

# ---- setup ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_dict = experiment_data.get("dropout_prob", {}).get("SPR_BENCH", {})
if not spr_dict:
    print("No SPR_BENCH data found")
    exit()

dropouts = sorted(spr_dict.keys(), key=float)
epochs = len(next(iter(spr_dict.values()))["losses"]["train"])

# ---- organize ----
train_losses = {p: spr_dict[p]["losses"]["train"] for p in dropouts}
val_losses = {p: spr_dict[p]["losses"]["val"] for p in dropouts}
val_cwa2 = {p: spr_dict[p]["metrics"]["val_cwa2"] for p in dropouts}

# ---- plot 1: Loss curves ----
try:
    plt.figure(figsize=(6, 4))
    for p in dropouts:
        plt.plot(range(1, epochs + 1), train_losses[p], label=f"train p={p}")
        plt.plot(
            range(1, epochs + 1), val_losses[p], linestyle="--", label=f"val p={p}"
        )
    plt.title("SPR_BENCH Training vs Validation Loss\n(lines: train, dashed: val)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize="small")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---- plot 2: Validation CWA2 curves ----
try:
    plt.figure(figsize=(6, 4))
    for p in dropouts:
        plt.plot(range(1, epochs + 1), val_cwa2[p], label=f"p={p}")
    plt.title("SPR_BENCH Validation CWA2 Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CWA2")
    plt.legend(fontsize="small")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_cwa2_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 curve plot: {e}")
    plt.close()

# ---- plot 3: Final-epoch CWA2 vs dropout ----
try:
    plt.figure(figsize=(5, 4))
    final_cwa2 = [val_cwa2[p][-1] for p in dropouts]
    x = list(map(float, dropouts))
    plt.plot(x, final_cwa2, marker="o")
    plt.title("SPR_BENCH Final Validation CWA2 vs Dropout")
    plt.xlabel("Dropout Probability")
    plt.ylabel("Final CWA2")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_final_cwa2_vs_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final CWA2 plot: {e}")
    plt.close()

# ---- print best configuration ----
best_idx = int(np.argmax([val_cwa2[p][-1] for p in dropouts]))
best_p = dropouts[best_idx]
best_cwa2 = val_cwa2[best_p][-1]
print(f"Best dropout: {best_p}, Final Validation CWA2: {best_cwa2:.4f}")

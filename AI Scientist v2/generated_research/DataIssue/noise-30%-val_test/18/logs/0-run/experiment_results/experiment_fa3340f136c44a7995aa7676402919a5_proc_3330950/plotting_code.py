import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("dropout_prob", {}).get("SPR_BENCH", {})

# quick numeric summary --------------------------------------------------------
best_f1_per_p = {}
for p_str, rec in spr_data.items():
    val_f1 = rec["metrics"]["val"]
    if val_f1:
        best_f1_per_p[float(p_str)] = max(val_f1)
print("Best validation macro-F1 per dropout value:")
for k, v in sorted(best_f1_per_p.items()):
    print(f"  p={k:>3.1f} --> {v:.4f}")

# ---------------- PLOT 1: per-epoch F1 curves ---------------------------------
try:
    plt.figure()
    for p_str, rec in spr_data.items():
        epochs = rec["epochs"]
        train_f1 = rec["metrics"]["train"]
        val_f1 = rec["metrics"]["val"]
        # Train curve (dashed)
        plt.plot(epochs, train_f1, "--", label=f"train p={p_str}")
        # Validation curve (solid)
        plt.plot(epochs, val_f1, "-", label=f"val p={p_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Train vs. Val Macro-F1 across Epochs (all dropouts)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_epochs.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves plot: {e}")
    plt.close()

# ---------------- PLOT 2: bar chart of best F1 --------------------------------
try:
    plt.figure()
    ps, bests = zip(*sorted(best_f1_per_p.items()))
    plt.bar([str(p) for p in ps], bests, color="skyblue")
    for i, v in enumerate(bests):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xlabel("Dropout probability")
    plt.ylabel("Best Val Macro-F1")
    plt.title("SPR_BENCH: Best Validation F1 vs. Dropout Probability")
    fname = os.path.join(working_dir, "SPR_BENCH_best_valF1_vs_dropout.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar chart: {e}")
    plt.close()

# -------------- PLOT 3: loss curves for best p --------------------------------
try:
    if best_f1_per_p:
        best_p = max(best_f1_per_p, key=best_f1_per_p.get)
        rec = spr_data[str(best_p)]
        epochs = rec["epochs"]
        tr_loss = rec["losses"]["train"]
        val_loss = rec["losses"]["val"]

        plt.figure()
        plt.plot(epochs, tr_loss, "--o", label="Train Loss")
        plt.plot(epochs, val_loss, "-o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH: Loss Curves (Best dropout p={best_p})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_loss_curves_best_p{best_p}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

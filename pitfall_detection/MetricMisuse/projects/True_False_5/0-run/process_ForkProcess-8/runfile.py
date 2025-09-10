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
    exit()

# ---------- helper ----------
dp_dict = experiment_data.get("dropout_prob", {})
dps, train_losses, val_losses, val_rcwas, test_rcwas = [], [], [], [], []
for k, v in dp_dict.items():  # k looks like 'p_0.1'
    try:
        dp = float(k.split("_")[1])
    except Exception:
        continue
    record = v["SPR_BENCH"]
    dps.append(dp)
    train_losses.append(record["losses"]["train"])
    val_losses.append(record["losses"]["val"])
    val_rcwas.append(record["metrics"]["val_rcwa"])
    test_rcwas.append(record["test_metrics"]["rcwa"])

# ensure consistent ordering
order = np.argsort(dps)
dps = np.array(dps)[order]
train_losses = [train_losses[i] for i in order]
val_losses = [val_losses[i] for i in order]
val_rcwas = [val_rcwas[i] for i in order]
test_rcwas = np.array(test_rcwas)[order]

best_dp = dps[np.argmax([max(r) for r in val_rcwas])]
best_rcwa = max([max(r) for r in val_rcwas])
print(f"Best val RCWA={best_rcwa:.4f} achieved with dropout={best_dp}")

# ---------- Figure 1: loss curves ----------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dp, tr_l, va_l in zip(dps, train_losses, val_losses):
        epochs = np.arange(1, len(tr_l) + 1)
        axes[0].plot(epochs, tr_l, label=f"dropout={dp}")
        axes[1].plot(epochs, va_l, label=f"dropout={dp}")
    axes[0].set_title("Left: Training Loss (SPR_BENCH)")
    axes[1].set_title("Right: Validation Loss (SPR_BENCH)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Figure 2: validation RCWA ----------
try:
    plt.figure(figsize=(5, 4))
    for dp, rcwa in zip(dps, val_rcwas):
        epochs = np.arange(1, len(rcwa) + 1)
        plt.plot(epochs, rcwa, marker="o", label=f"dropout={dp}")
    plt.title("Validation RCWA vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("RCWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_RCWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RCWA plot: {e}")
    plt.close()

# ---------- Figure 3: test RCWA bar plot ----------
try:
    plt.figure(figsize=(6, 4))
    plt.bar([str(dp) for dp in dps], test_rcwas, color="skyblue")
    plt.title("Test RCWA by Dropout Probability (SPR_BENCH)")
    plt.xlabel("Dropout")
    plt.ylabel("RCWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_RCWA_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test RCWA bar plot: {e}")
    plt.close()

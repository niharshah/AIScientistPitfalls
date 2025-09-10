import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dropout_dict = experiment_data.get("dropout", {})
if not dropout_dict:
    print("No dropout experiments found. Exiting.")
    quit()

# ---------- collect arrays ----------
epochs = None
val_losses, train_losses, val_cwa = {}, {}, {}
for key, rec in dropout_dict.items():
    train_losses[key] = rec["losses"]["train"]
    val_losses[key] = rec["losses"]["val"]
    val_cwa[key] = rec["metrics"]["val"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)

# best dropout (lowest final val loss)
best_dp = min(val_losses.items(), key=lambda kv: kv[1][-1])[0]

# ---------- PLOT 1: val loss vs epoch ----------
try:
    plt.figure()
    for k, vals in val_losses.items():
        plt.plot(epochs, vals, label=k)
    plt.title("Validation Loss vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_loss_across_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val loss plot: {e}")
    plt.close()

# ---------- PLOT 2: val CWA-2D vs epoch ----------
try:
    plt.figure()
    for k, vals in val_cwa.items():
        plt.plot(epochs, vals, label=k)
    plt.title("Validation CWA-2D vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("CWA-2D")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_cwa_across_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ---------- PLOT 3: final CWA bar chart ----------
try:
    plt.figure()
    dps = list(val_cwa.keys())
    finals = [val_cwa[k][-1] for k in dps]
    plt.bar(dps, finals)
    plt.title("Final Validation CWA-2D per Dropout (SPR_BENCH)")
    plt.xlabel("Dropout setting")
    plt.ylabel("Final CWA-2D")
    fname = os.path.join(working_dir, "SPR_BENCH_final_cwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ---------- PLOT 4: train vs val loss for best dropout ----------
try:
    plt.figure()
    plt.plot(epochs, train_losses[best_dp], label="train")
    plt.plot(epochs, val_losses[best_dp], label="val")
    plt.title(f"Best Dropout {best_dp}: Loss Curves (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_BENCH_best_{best_dp}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best-loss plot: {e}")
    plt.close()

# ---------- print summary ----------
print("\nFinal Validation CWA-2D by dropout:")
for k in sorted(val_cwa.keys()):
    print(f"{k}: {val_cwa[k][-1]:.4f}")
print(f"\nBest configuration based on final val loss: {best_dp}")

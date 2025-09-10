import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["dropout_rate"]["SPR_BENCH"]
    rates = ed["rates"]  # list(float)
    train_losses = ed["losses"]["train"]  # list(list)
    val_losses = ed["losses"]["val"]  # list(list)
    val_hwas = ed["metrics"]["val"]  # list(list)

    # keep final hwa for printing / bar plot
    final_hwas = []

    for i, drop in enumerate(rates):
        tl, vl, hwa = train_losses[i], val_losses[i], val_hwas[i]
        epochs = np.arange(1, len(tl) + 1)

        # ---- loss curve ----
        try:
            plt.figure()
            plt.plot(epochs, tl, label="Train")
            plt.plot(epochs, vl, label="Validation")
            plt.title(f"SPR_BENCH Loss Curve (dropout={drop})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"SPR_BENCH_loss_curve_dropout_{drop}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for dropout {drop}: {e}")
            plt.close()

        # ---- HWA curve ----
        try:
            plt.figure()
            plt.plot(epochs, hwa, marker="o")
            plt.title(f"SPR_BENCH Validation HWA (dropout={drop})")
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic Weighted Accuracy")
            fname = f"SPR_BENCH_hwa_curve_dropout_{drop}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating HWA plot for dropout {drop}: {e}")
            plt.close()

        final_hwas.append(hwa[-1] if len(hwa) else np.nan)

    # ---- comparative bar chart ----
    try:
        plt.figure()
        plt.bar([str(r) for r in rates], final_hwas, color="skyblue")
        plt.title("SPR_BENCH Final-Epoch HWA vs. Dropout Rate")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Harmonic Weighted Accuracy")
        fname = "SPR_BENCH_final_hwa_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating comparative HWA bar plot: {e}")
        plt.close()

    # ---- print metrics ----
    for r, h in zip(rates, final_hwas):
        print(f"Dropout={r}: Final Val HWA={h:.4f}")

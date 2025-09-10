import matplotlib.pyplot as plt
import numpy as np
import os

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
    hidden_dict = experiment_data.get("hidden_dim", {})
    # ---------- summary metrics ----------
    best_hwa = -1
    best_hdim = None
    hwa_summary = {}
    for hdim, data in hidden_dict.items():
        hwa_curve = [m["hwa"] for m in data["SPR_BENCH"]["metrics"]["val"]]
        if hwa_curve:
            max_hwa = max(hwa_curve)
            hwa_summary[hdim] = max_hwa
            if max_hwa > best_hwa:
                best_hwa, best_hdim = max_hwa, hdim

    # ---------- bar chart of final HWA ----------
    try:
        plt.figure(figsize=(6, 4))
        dims, hwas = zip(*sorted(hwa_summary.items()))
        plt.bar([str(d) for d in dims], hwas, color="skyblue")
        plt.title("SPR_BENCH: Final Dev HWA vs Hidden Dim")
        plt.ylabel("HWA")
        plt.xlabel("Hidden Dimension")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA bar chart: {e}")
        plt.close()

    # ---------- per-hidden_dim curves ----------
    plotted = 0
    for hdim, data in sorted(hidden_dict.items()):
        if plotted >= 4:  # guard: at most 4 such figs (total <=5)
            break
        try:
            losses = data["SPR_BENCH"]["losses"]
            metrics = data["SPR_BENCH"]["metrics"]
            train_loss = losses["train"]
            val_loss = losses["val"]
            hwa_curve = [m["hwa"] for m in metrics["val"]]

            epochs = range(1, len(train_loss) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # left subplot: losses
            axes[0].plot(epochs, train_loss, label="Train")
            axes[0].plot(epochs, val_loss, label="Val")
            axes[0].set_title(f"Hidden={hdim} | Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Cross-Entropy")
            axes[0].legend()

            # right subplot: HWA
            axes[1].plot(epochs, hwa_curve, marker="o", color="orange")
            axes[1].set_title(f"Hidden={hdim} | HWA")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("HWA")

            plt.suptitle(f"SPR_BENCH Results (Hidden={hdim})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = os.path.join(working_dir, f"SPR_BENCH_hidden{hdim}_loss_hwa.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
        except Exception as e:
            print(f"Error creating plot for hidden_dim={hdim}: {e}")
            plt.close()

    # ---------- print summary ----------
    print("=== Dev set best HWA by hidden_dim ===")
    for hdim, hwa in sorted(hwa_summary.items()):
        flag = "<-- best" if hdim == best_hdim else ""
        print(f"hidden_dim={hdim:<4}: best_HWA={hwa:.4f} {flag}")

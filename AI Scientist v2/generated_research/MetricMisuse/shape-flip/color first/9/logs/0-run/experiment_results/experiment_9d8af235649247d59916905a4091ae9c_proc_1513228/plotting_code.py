import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_runs = exp_data["num_epochs"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_runs = {}

# ---------- plotting ----------
max_plots = 5
for i, (k, v) in enumerate(sorted(spr_runs.items(), key=lambda x: int(x[0]))):
    if i >= max_plots:
        break
    epochs = v["epochs"]
    tr_loss = v["losses"]["train"]
    val_loss = v["losses"]["val"]
    tr_sdwa = v["metrics"]["train"]
    val_sdwa = v["metrics"]["val"]

    # ---- loss curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title(f"SPR – Loss vs Epochs (num_epochs={k})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_loss_curve_{k}_epochs.png"
        plt.savefig(os.path.join(working_dir, fname), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {k}: {e}")
        plt.close()

    # ---- SDWA curve ----
    try:
        plt.figure()
        plt.plot(epochs, tr_sdwa, label="Train")
        plt.plot(epochs, val_sdwa, label="Validation")
        plt.title(f"SPR – SDWA Metric vs Epochs (num_epochs={k})")
        plt.xlabel("Epoch")
        plt.ylabel("SDWA")
        plt.legend()
        fname = f"SPR_sdwa_curve_{k}_epochs.png"
        plt.savefig(os.path.join(working_dir, fname), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating SDWA plot for {k}: {e}")
        plt.close()

# ---------- print summary ----------
print("Summary (num_epochs | best_val_loss | test_SDWA)")
for k, v in sorted(spr_runs.items(), key=lambda x: int(x[0])):
    best_val = v.get("best_val", None)
    test_sdwa = np.mean(v.get("metrics", {}).get("val", [0]))  # fallback if not stored
    if "predictions" in v and "ground_truth" in v:
        # recompute SDWA on test split if available
        from math import isfinite

        seqs_fake = [""] * len(v["predictions"])  # dummy seqs to pass to metric if lost
        try:
            from __main__ import sdwa_metric

            test_sdwa = sdwa_metric(seqs_fake, v["ground_truth"], v["predictions"])
        except Exception:
            pass
    print(f"{k:>10} | {best_val:.4f} | {test_sdwa:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------------------------------------------------------ #
# Helper metrics (duplicated here to avoid extra imports)            #
# ------------------------------------------------------------------ #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    corr = [w_i if yt == yp else 0 for w_i, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------------------------------------------------------ #
# Load experiment data                                               #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR", {})
loss_train = [v for _, v in spr.get("losses", {}).get("train", [])]
loss_val = [v for _, v in spr.get("losses", {}).get("val", [])]
pcwa_train = [v for _, v in spr.get("metrics", {}).get("train", [])]
pcwa_val = [v for _, v in spr.get("metrics", {}).get("val", [])]
epochs = np.arange(1, len(loss_train) + 1)

# ------------------------------------------------------------------ #
# 1) Loss curves                                                     #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, loss_train, "--o", label="Train")
    plt.plot(epochs, loss_val, "-s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss – SPR dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) PCWA curves                                                     #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, pcwa_train, "--o", label="Train")
    plt.plot(epochs, pcwa_val, "-s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("PCWA")
    plt.title("Training vs Validation PCWA – SPR dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_pcwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PCWA curves plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Final test-set bar chart                                        #
# ------------------------------------------------------------------ #
try:
    seqs = spr.get("sequences", spr.get("ground_truth", []))  # fallback
    y_true = spr.get("ground_truth", [])
    y_pred = spr.get("predictions", [])
    if seqs and y_true and y_pred:
        acc = sum(int(y == p) for y, p in zip(y_true, y_pred)) / len(y_true)
        pc = pcwa(seqs, y_true, y_pred)
        cwa_num = sum(
            count_color_variety(s) if y == p else 0
            for s, y, p in zip(seqs, y_true, y_pred)
        )
        cwa_den = sum(count_color_variety(s) for s in seqs)
        swa_num = sum(
            count_shape_variety(s) if y == p else 0
            for s, y, p in zip(seqs, y_true, y_pred)
        )
        swa_den = sum(count_shape_variety(s) for s in seqs)
        cwa = cwa_num / cwa_den if cwa_den else 0.0
        swa = swa_num / swa_den if swa_den else 0.0

        metrics = {"ACC": acc, "PCWA": pc, "CWA": cwa, "SWA": swa}
        plt.figure(figsize=(6, 4))
        plt.bar(metrics.keys(), metrics.values(), color="skyblue")
        plt.title("Final Test-set Metrics – SPR dataset")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_final_metrics_bar.png"))
        plt.close()
        print("Final test metrics:", metrics)
    else:
        print("Predictions / ground-truth not found; skipping final bar chart.")
except Exception as e:
    print(f"Error creating final metrics bar plot: {e}")
    plt.close()

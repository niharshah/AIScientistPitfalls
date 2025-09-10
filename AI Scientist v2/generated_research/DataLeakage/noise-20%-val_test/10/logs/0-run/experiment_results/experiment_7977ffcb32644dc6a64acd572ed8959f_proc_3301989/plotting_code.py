import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_scores = {}

# ---------- per-run curves ----------
for tag, data in experiment_data.items():
    ed = data.get("SPR_BENCH", {})
    losses = ed.get("losses", {})
    metrics = ed.get("metrics", {})
    # Loss curve
    try:
        plt.figure()
        if losses:
            plt.plot(losses.get("train", []), label="Train Loss")
            plt.plot(losses.get("val", []), label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{tag} Loss Curves (SPR_BENCH)")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_{tag}_loss_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {tag}: {e}")
        plt.close()
    # F1 curve
    try:
        plt.figure()
        if metrics:
            plt.plot(metrics.get("train_f1", []), label="Train F1")
            plt.plot(metrics.get("val_f1", []), label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{tag} F1 Curves (SPR_BENCH)")
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_{tag}_f1_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve for {tag}: {e}")
        plt.close()

    # store best val F1
    if metrics.get("val_f1"):
        best_scores[tag] = max(metrics["val_f1"])

# ---------- summary bar chart ----------
try:
    plt.figure()
    tags = list(best_scores.keys())
    scores = [best_scores[t] for t in tags]
    plt.bar(tags, scores, color="skyblue")
    plt.ylabel("Best Dev Macro-F1")
    plt.title("Best Validation F1 by Num Epochs (SPR_BENCH)")
    plt.xticks(rotation=15)
    fname = os.path.join(working_dir, "SPR_BENCH_best_dev_f1_summary.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()

# ---------- print metrics ----------
for t, s in best_scores.items():
    print(f"{t}: Best Dev F1 = {s:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- load data -----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_scores = {}  # ablation -> (best_f1, epoch)

# ----------- per-ablation plots -----------
for ablation_name, runs in experiment_data.items():
    run = runs.get("SPR-BENCH", {})
    epochs = run.get("epochs", [])
    train_loss = run.get("losses", {}).get("train", [])
    val_loss = run.get("losses", {}).get("val", [])
    val_f1 = run.get("metrics", {}).get("val_f1", [])

    # cache best score
    if val_f1:
        best_epoch = int(np.argmax(val_f1)) + 1
        best_scores[ablation_name] = (float(np.max(val_f1)), best_epoch)

    # --- Loss curve ---
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR-BENCH Loss Curve ({ablation_name})")
        plt.legend()
        fname = f"spr_bench_loss_{ablation_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ablation_name}: {e}")
        plt.close()

    # --- Validation F1 curve ---
    try:
        plt.figure()
        plt.plot(epochs, val_f1, label="Val Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"SPR-BENCH Validation F1 ({ablation_name})")
        plt.legend()
        fname = f"spr_bench_f1_{ablation_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {ablation_name}: {e}")
        plt.close()

# ----------- aggregated bar chart -----------
try:
    if best_scores:
        names = list(best_scores.keys())
        scores = [best_scores[n][0] for n in names]
        plt.figure()
        plt.bar(names, scores, color="skyblue")
        for i, s in enumerate(scores):
            plt.text(i, s + 0.005, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        plt.ylabel("Best Val Macro F1")
        plt.title("SPR-BENCH: Best Validation F1 Across Ablations")
        plt.savefig(os.path.join(working_dir, "spr_bench_best_f1_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ----------- print summary -----------
if best_scores:
    print("=== Best Validation F1 per Ablation ===")
    for k, (score, ep) in best_scores.items():
        print(f"{k:15s}  F1={score:.4f} at epoch {ep}")

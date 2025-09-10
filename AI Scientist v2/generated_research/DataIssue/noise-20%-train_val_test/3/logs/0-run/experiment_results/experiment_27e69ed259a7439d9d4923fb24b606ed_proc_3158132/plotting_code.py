import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_results = experiment_data.get("d_model_tuning", {}).get("SPR_BENCH", {})

# -------- print and collect summary --------
summary = []
for dm, data in spr_results.items():
    summary.append((int(dm), data["test_loss"], data["test_f1"]))
summary.sort(key=lambda x: x[0])
print("d_model | test_loss | test_f1")
for dm, tl, tf1 in summary:
    print(f"{dm:7d} | {tl:9.4f} | {tf1:7.4f}")

# -------- plot loss curves for each d_model (max 4) --------
for i, (dm, data) in enumerate(summary):
    if i >= 4:  # safeguard although only 4 exist
        break
    try:
        metrics = spr_results[str(dm)]["metrics"]
        epochs = np.arange(1, len(metrics["train_loss"]) + 1)
        plt.figure()
        plt.plot(epochs, metrics["train_loss"], label="Train Loss")
        plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss Curves (d_model={dm})\nTraining vs Validation Loss")
        plt.legend()
        fname = f"SPR_BENCH_dmodel{dm}_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for d_model={dm}: {e}")
        plt.close()

# -------- bar chart of final test F1 across d_model --------
try:
    d_models = [dm for dm, _, _ in summary]
    f1_scores = [tf1 for _, _, tf1 in summary]
    plt.figure()
    plt.bar(range(len(d_models)), f1_scores, tick_label=d_models)
    plt.xlabel("d_model size")
    plt.ylabel("Test F1")
    plt.title("SPR_BENCH Final Test F1 across d_model settings")
    fname = "SPR_BENCH_testF1_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating final F1 bar plot: {e}")
    plt.close()

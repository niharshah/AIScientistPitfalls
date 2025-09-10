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
    experiment_data = {}

dm_data = experiment_data.get("d_model_tuning", {})
if not dm_data:
    print("No d_model_tuning data found.")
    exit()

dmodels = sorted(map(int, dm_data.keys()))
colors = plt.cm.tab10(np.linspace(0, 1, len(dmodels)))

# ---------- PLOT 1: loss curves ----------
try:
    plt.figure()
    for c, dm in zip(colors, dmodels):
        epochs = dm_data[str(dm)]["epochs"]
        plt.plot(
            epochs,
            dm_data[str(dm)]["losses"]["train"],
            color=c,
            linestyle="-",
            label=f"{dm}-train",
        )
        plt.plot(
            epochs,
            dm_data[str(dm)]["losses"]["val"],
            color=c,
            linestyle="--",
            label=f"{dm}-val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves\nSolid: Train, Dashed: Val")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------- PLOT 2: F1 curves ----------
try:
    plt.figure()
    for c, dm in zip(colors, dmodels):
        epochs = dm_data[str(dm)]["epochs"]
        plt.plot(
            epochs,
            dm_data[str(dm)]["metrics"]["train_f1"],
            color=c,
            linestyle="-",
            label=f"{dm}-train",
        )
        plt.plot(
            epochs,
            dm_data[str(dm)]["metrics"]["val_f1"],
            color=c,
            linestyle="--",
            label=f"{dm}-val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH – Macro-F1 Curves\nSolid: Train, Dashed: Val")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves plot: {e}")
    plt.close()

# ---------- PLOT 3: Test F1 bar chart ----------
try:
    plt.figure()
    test_f1s = [dm_data[str(dm)]["test_f1"] for dm in dmodels]
    plt.bar(range(len(dmodels)), test_f1s, tick_label=dmodels, color=colors)
    plt.ylabel("Test Macro F1")
    plt.xlabel("d_model")
    plt.title("SPR_BENCH – Test F1 vs d_model")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---------- identify best model ----------
best_dm = max(dmodels, key=lambda d: dm_data[str(d)]["best_val_f1"])
best_entry = dm_data[str(best_dm)]
print(
    f'Best d_model = {best_dm} | Val F1 = {best_entry["best_val_f1"]:.4f} | '
    f'Test F1 = {best_entry["test_f1"]:.4f}'
)

# ---------- PLOT 4: Confusion matrix for best model ----------
try:
    preds = np.array(best_entry["predictions"])
    gts = np.array(best_entry["ground_truth"])
    num_cls = len(np.unique(gts))
    conf = np.zeros((num_cls, num_cls), dtype=int)
    for p, g in zip(preds, gts):
        conf[g, p] += 1
    plt.figure()
    im = plt.imshow(conf, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH – Confusion Matrix (d_model={best_dm})")
    for i in range(num_cls):
        for j in range(num_cls):
            plt.text(
                j,
                i,
                conf[i, j],
                ha="center",
                va="center",
                color="white" if conf[i, j] > conf.max() / 2 else "black",
                fontsize=8,
            )
    fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_dmodel{best_dm}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

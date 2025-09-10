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

drop_dict = experiment_data.get("dropout_rate", {})
dropouts = sorted(drop_dict.keys())

# -------- helper to find best dropout --------
best_rate, best_val = None, -1.0
for r in dropouts:
    val_f1 = drop_dict[r]["metrics"]["val_macro_f1"][-1]  # last epoch val F1
    if val_f1 > best_val:
        best_val, best_rate = val_f1, r
best_test = drop_dict[best_rate]["test_macro_f1"] if best_rate is not None else None

# -------- plot 1: test macro-F1 summary --------
try:
    plt.figure(figsize=(6, 4))
    test_scores = [drop_dict[r]["test_macro_f1"] for r in dropouts]
    plt.bar([str(r) for r in dropouts], test_scores, color="skyblue")
    plt.ylabel("Macro-F1")
    plt.xlabel("Dropout rate")
    plt.title("SPR_BENCH: Test Macro-F1 vs Dropout")
    for i, v in enumerate(test_scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_test_F1_vs_dropout.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# -------- per-dropout F1 curves (limit to 4 = total 5 plots) --------
for idx, r in enumerate(dropouts):
    if idx >= 4:  # ensure at most 5 plots total (1 summary + 4 curves)
        break
    try:
        rec = drop_dict[r]
        epochs = rec["epochs"]
        tr_f1 = rec["metrics"]["train_macro_f1"]
        val_f1 = rec["metrics"]["val_macro_f1"]

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_f1, label="train")
        plt.plot(epochs, val_f1, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.title(f"SPR_BENCH Macro-F1 Curves (dropout={r})")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_macro_F1_dropout_{r}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for dropout={r}: {e}")
        plt.close()

print(
    f"Best dropout rate: {best_rate} | Val-F1: {best_val:.4f} | Test-F1: {best_test:.4f}"
    if best_rate is not None
    else "No data found."
)

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_runs = exp_data["nhead"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_runs = {}

# ---------- helper to collect series ----------
epochs = None
heads, test_losses, test_f1s = [], [], []
loss_curves, f1_curves = {}, {}

for key, run in spr_runs.items():
    h = int(key.split("_")[1])
    heads.append(h)
    test_losses.append(run["test_loss"])
    test_f1s.append(run["test_macroF1"])
    loss_curves[h] = (run["losses"]["train"], run["losses"]["val"])
    f1_curves[h] = (run["metrics"]["train"], run["metrics"]["val"])
    if epochs is None:
        epochs = run["epochs"]

# ---------- Figure 1: loss curves ----------
try:
    plt.figure()
    for h in sorted(loss_curves):
        tr, va = loss_curves[h]
        plt.plot(epochs, tr, label=f"train nhead={h}")
        plt.plot(epochs, va, "--", label=f"val nhead={h}")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Figure 2: macro-F1 curves ----------
try:
    plt.figure()
    for h in sorted(f1_curves):
        tr, va = f1_curves[h]
        plt.plot(epochs, tr, label=f"train nhead={h}")
        plt.plot(epochs, va, "--", label=f"val nhead={h}")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ---------- Figure 3: test macro-F1 bar chart ----------
try:
    plt.figure()
    x = np.arange(len(heads))
    plt.bar(x, test_f1s, color="skyblue")
    plt.xticks(x, [str(h) for h in heads])
    plt.title("SPR_BENCH: Test Macro-F1 vs nhead")
    plt.xlabel("nhead")
    plt.ylabel("Test Macro-F1")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_macroF1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar plot: {e}")
    plt.close()

# ---------- Figure 4: confusion matrix for best model ----------
try:
    best_idx = int(np.argmax(test_f1s))
    best_head = heads[best_idx]
    preds = spr_runs[f"nhead_{best_head}"]["predictions"]
    gts = spr_runs[f"nhead_{best_head}"]["ground_truth"]
    cm = confusion_matrix(gts, preds)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR_BENCH Confusion Matrix (best nhead={best_head})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(
        os.path.join(working_dir, f"SPR_BENCH_confusion_best_nhead_{best_head}.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print summary ----------
print("=== Test Results ===")
for h, loss, f1 in zip(heads, test_losses, test_f1s):
    print(f"nhead={h:2d}: test_loss={loss:.4f}  test_macroF1={f1:.4f}")

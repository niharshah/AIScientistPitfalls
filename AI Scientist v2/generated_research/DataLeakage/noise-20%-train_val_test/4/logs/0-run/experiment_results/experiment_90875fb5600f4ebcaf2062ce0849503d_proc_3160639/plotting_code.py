import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("d_model_tuning", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data found.")
    exit()

d_models = sorted(spr_data.keys())
epochs = spr_data[d_models[0]]["epochs"]

# Gather metrics
train_losses = {dm: spr_data[dm]["losses"]["train"] for dm in d_models}
val_losses = {dm: spr_data[dm]["losses"]["val"] for dm in d_models}
train_f1s = {dm: spr_data[dm]["metrics"]["train"] for dm in d_models}
val_f1s = {dm: spr_data[dm]["metrics"]["val"] for dm in d_models}
best_val_f1 = {dm: spr_data[dm]["best_val_f1"] for dm in d_models}
test_f1 = {dm: spr_data[dm]["test_f1"] for dm in d_models}

# -------- 1) Loss curves --------
try:
    plt.figure()
    for dm in d_models:
        plt.plot(epochs, train_losses[dm], label=f"train d_model={dm}", linestyle="--")
        plt.plot(epochs, val_losses[dm], label=f"val d_model={dm}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Char Transformer)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- 2) F1 curves --------
try:
    plt.figure()
    for dm in d_models:
        plt.plot(epochs, train_f1s[dm], label=f"train d_model={dm}", linestyle="--")
        plt.plot(epochs, val_f1s[dm], label=f"val d_model={dm}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves (Char Transformer)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# -------- 3) Bar chart: best val F1 --------
try:
    plt.figure()
    plt.bar(range(len(d_models)), [best_val_f1[dm] for dm in d_models])
    plt.xticks(range(len(d_models)), d_models)
    plt.ylabel("Best Validation Macro-F1")
    plt.title("SPR_BENCH Best Validation F1 vs d_model")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_best_val_f1.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best val F1 plot: {e}")
    plt.close()

# -------- 4) Bar chart: test F1 --------
try:
    plt.figure()
    plt.bar(range(len(d_models)), [test_f1[dm] for dm in d_models])
    plt.xticks(range(len(d_models)), d_models)
    plt.ylabel("Test Macro-F1")
    plt.title("SPR_BENCH Test F1 vs d_model")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test F1 plot: {e}")
    plt.close()

# -------- 5) Confusion Matrix for best model --------
try:
    best_dm = max(test_f1, key=test_f1.get)
    preds = np.array(spr_data[best_dm]["predictions"])
    gts = np.array(spr_data[best_dm]["ground_truth"])
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        f"SPR_BENCH Confusion Matrix (d_model={best_dm})\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.tight_layout()
    fname = os.path.join(
        working_dir, f"SPR_BENCH_confusion_matrix_dmodel_{best_dm}.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------- print summary --------
print("\n=== Summary of Results ===")
for dm in d_models:
    print(
        f"d_model {dm}:  best_val_F1={best_val_f1[dm]:.4f} | test_F1={test_f1[dm]:.4f}"
    )
print(f"\nBest test model: d_model={max(test_f1, key=test_f1.get)}")

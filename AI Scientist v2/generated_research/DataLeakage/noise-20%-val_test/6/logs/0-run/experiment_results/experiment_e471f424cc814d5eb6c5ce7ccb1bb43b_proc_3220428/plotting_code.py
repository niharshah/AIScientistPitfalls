import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["rare_ngram_pruning"]["SPR_BENCH"]
    cfgs = ed["configs"]
    train_acc = ed["metrics"]["train_acc"]
    val_acc = ed["metrics"]["val_acc"]
    rule_fid = ed["metrics"]["rule_fidelity"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    preds = ed["predictions"]
    gts = ed["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    train_acc = val_acc = rule_fid = train_loss = val_loss = preds = gts = cfgs = []

epochs = np.arange(1, len(train_acc[0]) + 1) if len(train_acc) else np.array([])

# helper palette
colors = plt.cm.tab10.colors

# -------------------- 1) accuracy curves --------------------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(
            epochs, train_acc[i], label=f"{cfg}-train", color=colors[i], linestyle="-"
        )
        plt.plot(
            epochs, val_acc[i], label=f"{cfg}-val", color=colors[i], linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training vs Validation Accuracy (rare_ngram_pruning)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_acc_curves_rare_ngram_pruning.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- 2) loss curves --------------------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(
            epochs, train_loss[i], label=f"{cfg}-train", color=colors[i], linestyle="-"
        )
        plt.plot(
            epochs, val_loss[i], label=f"{cfg}-val", color=colors[i], linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss (rare_ngram_pruning)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_loss_curves_rare_ngram_pruning.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- 3) rule fidelity curves --------------------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        plt.plot(epochs, rule_fid[i], label=cfg, color=colors[i])
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Rule-Fidelity Across Epochs (rare_ngram_pruning)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_rule_fid_curves_rare_ngram_pruning.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# -------------------- 4) confusion matrix / bar chart --------------------
try:
    if len(preds) and len(gts):
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_cfg.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Plotting complete.")

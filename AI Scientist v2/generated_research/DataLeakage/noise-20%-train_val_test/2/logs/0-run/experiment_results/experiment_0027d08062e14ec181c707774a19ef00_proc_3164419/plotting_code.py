import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------- helper
ds_key = "dual_stream"
ed = experiment_data.get(ds_key, {})
epochs = ed.get("epochs", [])
tr_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
pre_loss = ed.get("losses", {}).get("pretrain", [])
tr_f1 = ed.get("metrics", {}).get("train_macro_f1", [])
val_f1 = ed.get("metrics", {}).get("val_macro_f1", [])
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))
test_f1 = ed.get("test_macro_f1", None)

# ------------------------------------------------------------------- colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 1) macro-F1 curves ----------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_f1, "--", color=colors[0], label="Train")
    plt.plot(epochs, val_f1, "-", color=colors[1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        "SPR_BENCH (dual_stream) Macro-F1 Curves\nLeft: Train (dashed), Right: Validation (solid)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dual_stream_macro_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# 2) loss curves --------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, "--", color=colors[2], label="Train")
    plt.plot(epochs, val_loss, "-", color=colors[3], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH (dual_stream) Loss Curves\nLeft: Train (dashed), Right: Validation (solid)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dual_stream_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) pre-training loss --------------------------------------------------------
try:
    if pre_loss:
        plt.figure()
        plt.plot(range(1, len(pre_loss) + 1), pre_loss, "-o", color=colors[4])
        plt.xlabel("Pre-train Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH (dual_stream) Pre-Training LM Loss")
        plt.savefig(os.path.join(working_dir, "dual_stream_pretrain_loss.png"))
        plt.close()
except Exception as e:
    print(f"Error creating pre-train loss plot: {e}")
    plt.close()

# 4) ground truth vs predictions bar chart -----------------------------------
try:
    if preds.size and gts.size:
        n_cls = int(max(preds.max(), gts.max()) + 1)
        gt_cnt = np.bincount(gts, minlength=n_cls)
        pr_cnt = np.bincount(preds, minlength=n_cls)
        idx = np.arange(n_cls)

        plt.figure(figsize=(8, 4))
        w = 0.35
        plt.bar(idx - w / 2, gt_cnt, width=w, label="Ground Truth")
        plt.bar(idx + w / 2, pr_cnt, width=w, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_BENCH (dual_stream)\nLeft: Ground Truth, Right: Predictions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dual_stream_gt_vs_pred_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating GT vs Pred plot: {e}")
    plt.close()

# ------------------------------------------------------------------- summary
print(f"Stored TEST Macro-F1 (dual_stream): {test_f1}")

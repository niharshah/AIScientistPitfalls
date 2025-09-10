import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("num_epochs_tuning", {}).get("SPR_BENCH", {})

hparams = spr_data.get("hparam_values", [])
epochs_run = spr_data.get("epochs_run", [])
loss_tr = spr_data.get("losses", {}).get("train", [])
loss_val = spr_data.get("losses", {}).get("val", [])
val_cwa = spr_data.get("metrics", {}).get("val_compwa", [])
test_cwa = spr_data.get("metrics", {}).get("test_compwa", [])
predictions = spr_data.get("predictions", [])
ground_truth = spr_data.get("ground_truth", [])

# ---------- Identify best configuration ----------
best_idx = int(np.argmax(val_cwa)) if val_cwa else None

# ---- 1. Best loss curve ----
try:
    if best_idx is not None:
        plt.figure()
        ep_range = range(1, len(loss_tr[best_idx]) + 1)
        plt.plot(ep_range, loss_tr[best_idx], label="Train")
        plt.plot(ep_range, loss_val[best_idx], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curve (Best num_epochs={})".format(hparams[best_idx]))
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best loss plot: {e}")
    plt.close()

# ---- 2. Validation CompWA per hyper-param ----
try:
    if val_cwa:
        plt.figure()
        plt.bar([str(h) for h in hparams], val_cwa, color="skyblue")
        plt.xlabel("num_epochs")
        plt.ylabel("Validation CompWA")
        plt.title("SPR_BENCH Validation CompWA vs. num_epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_compwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val CompWA bar: {e}")
    plt.close()

# ---- 3. Test CompWA per hyper-param ----
try:
    if test_cwa:
        plt.figure()
        plt.bar([str(h) for h in hparams], test_cwa, color="lightgreen")
        plt.xlabel("num_epochs")
        plt.ylabel("Test CompWA")
        plt.title("SPR_BENCH Test CompWA vs. num_epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_compwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test CompWA bar: {e}")
    plt.close()

# ---- 4. Epochs actually run ----
try:
    if epochs_run:
        plt.figure()
        plt.bar([str(h) for h in hparams], epochs_run, color="salmon")
        plt.xlabel("num_epochs setting")
        plt.ylabel("Epochs Run")
        plt.title("SPR_BENCH Early-Stopping Effect (Epochs Run)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_epochs_run.png"))
    plt.close()
except Exception as e:
    print(f"Error creating epochs-run bar: {e}")
    plt.close()

# ---- 5. Confusion matrix for best config ----
try:
    if best_idx is not None and predictions:
        preds = np.array(predictions[best_idx])
        truth = np.array(ground_truth)
        classes = np.unique(np.concatenate([truth, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(truth, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH Confusion Matrix (Best num_epochs={})".format(hparams[best_idx])
        )
        plt.xticks(classes)
        plt.yticks(classes)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

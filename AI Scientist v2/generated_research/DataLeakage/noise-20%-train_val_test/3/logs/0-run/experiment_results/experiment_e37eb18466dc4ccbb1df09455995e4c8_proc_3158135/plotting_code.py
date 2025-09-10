import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

# ---------- load experiment data ----------
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_data = experiment_data.get("weight_decay", {})
if not wd_data:
    print("No weight_decay data found; nothing to plot.")
    exit()

wds, train_losses, val_losses, val_f1s, test_f1s = [], [], [], [], []

# ---------- extract metrics ----------
for k, v in wd_data.items():
    wds.append(float(k.split("_")[-1]))
    train_losses.append(v["metrics"]["train_loss"])
    val_losses.append(v["metrics"]["val_loss"])
    val_f1s.append(v["metrics"]["val_f1"])
    preds, gts = v["predictions"], v["ground_truth"]
    try:
        test_f1s.append(f1_score(gts, preds, average="macro"))
    except Exception:
        test_f1s.append(np.nan)

epochs = range(1, len(train_losses[0]) + 1)

# ---------- training loss curves ----------
try:
    plt.figure()
    for wd, tl in zip(wds, train_losses):
        plt.plot(epochs, tl, label=f"wd={wd}")
    plt.title("SPR_BENCH Training Loss vs Epoch (weight decay sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_weight_decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ---------- validation loss curves ----------
try:
    plt.figure()
    for wd, vl in zip(wds, val_losses):
        plt.plot(epochs, vl, label=f"wd={wd}")
    plt.title("SPR_BENCH Validation Loss vs Epoch (weight decay sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_weight_decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# ---------- validation F1 curves ----------
try:
    plt.figure()
    for wd, vf1 in zip(wds, val_f1s):
        plt.plot(epochs, vf1, label=f"wd={wd}")
    plt.title("SPR_BENCH Validation Macro-F1 vs Epoch (weight decay sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_f1_weight_decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-F1 plot: {e}")
    plt.close()

# ---------- bar chart of final test F1 ----------
try:
    plt.figure()
    x_pos = np.arange(len(wds))
    plt.bar(x_pos, test_f1s, tick_label=[f"{wd}" for wd in wds])
    plt.title("SPR_BENCH Final Test Macro-F1 by Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Macro-F1")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_f1_weight_decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-F1 bar chart: {e}")
    plt.close()

# ---------- print summary ----------
for wd, vf1, tf1 in zip(wds, [v[-1] for v in val_f1s], test_f1s):
    print(f"wd={wd:>6}: final_val_f1={vf1:.4f}  test_f1={tf1:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- set up paths ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data --------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp, exp["weight_decay"] = {}, {}  # empty fallback to avoid NameError

wds = sorted(exp.get("weight_decay", {}).keys())
dataset = "SPR_BENCH"

# ---------------- plot 1: val loss ------------
try:
    plt.figure()
    for wd in wds:
        epochs = exp["weight_decay"][wd][dataset]["epochs"]
        val_loss = exp["weight_decay"][wd][dataset]["losses"]["val"]
        plt.plot(epochs, val_loss, label=f"wd={wd}")
    plt.title("SPR_BENCH Validation Loss across Weight Decays")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_loss_weight_decays.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- plot 2: val F1 --------------
try:
    plt.figure()
    for wd in wds:
        epochs = exp["weight_decay"][wd][dataset]["epochs"]
        val_f1 = exp["weight_decay"][wd][dataset]["metrics"]["val_f1"]
        plt.plot(epochs, val_f1, label=f"wd={wd}")
    plt.title("SPR_BENCH Validation Macro-F1 across Weight Decays")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_f1_weight_decays.png"))
    plt.close()
except Exception as e:
    print(f"Error creating f1 curve plot: {e}")
    plt.close()

# ---------------- plot 3: final F1 bar --------
try:
    final_f1 = [exp["weight_decay"][wd][dataset]["metrics"]["val_f1"][-1] for wd in wds]
    plt.figure()
    plt.bar([str(wd) for wd in wds], final_f1, color="skyblue")
    plt.title("SPR_BENCH Final Validation Macro-F1 vs Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Macro F1")
    plt.savefig(os.path.join(working_dir, "spr_bench_final_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final f1 bar plot: {e}")
    plt.close()

# ---------------- print evaluation metric -----
print("\nFinal Validation Macro-F1 scores")
for wd, f1 in zip(wds, final_f1):
    print(f"weight_decay={wd}: {f1:.4f}")

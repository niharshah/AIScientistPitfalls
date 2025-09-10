import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = exp.get("batch_size", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


# helper to gather lists aligned by epoch
def gather(metric_key, bsz):
    return exp[bsz]["metrics" if "f1" in metric_key else "losses"][metric_key]


# ---------- Figure 1: loss curves ----------
try:
    plt.figure()
    for bsz in sorted(exp):
        epochs = exp[bsz]["epochs"]
        plt.plot(epochs, gather("train", bsz), label=f"train bsz{bsz}")
        plt.plot(epochs, gather("val", bsz), label=f"val   bsz{bsz}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Figure 2: F1 curves ----------
try:
    plt.figure()
    for bsz in sorted(exp):
        epochs = exp[bsz]["epochs"]
        plt.plot(epochs, gather("train_f1", bsz), label=f"train bsz{bsz}")
        plt.plot(epochs, gather("val_f1", bsz), label=f"val   bsz{bsz}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---------- Figure 3: Test F1 by batch size ----------
try:
    plt.figure()
    bsizes = sorted(exp)
    test_f1 = [exp[b]["test_f1"] for b in bsizes]
    plt.bar([str(b) for b in bsizes], test_f1, color="skyblue")
    for i, v in enumerate(test_f1):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.xlabel("Batch Size")
    plt.ylabel("Best Test Macro-F1")
    plt.title("SPR_BENCH Test Performance vs. Batch Size")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_macroF1_vs_batchsize.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Test F1 plot: {e}")
    plt.close()

# ---------- quick console summary ----------
if exp:
    for bsz in sorted(exp):
        print(
            f"Batch {bsz}: best dev F1={max(exp[bsz]['metrics']['val_f1']):.3f}, "
            f"test F1={exp[bsz]['test_f1']:.3f}"
        )

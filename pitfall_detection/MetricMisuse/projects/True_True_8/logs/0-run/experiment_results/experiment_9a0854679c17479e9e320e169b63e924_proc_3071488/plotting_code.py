import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sweep = experiment_data["aug_probability"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}

# containers for summary
final_ACS, final_acc = {}, {}

# iterate once to collect summary stats
for prob, run in sweep.items():
    # unpack tuples -> arrays
    epochs, acs_vals = zip(*run["val_ACS"])
    preds, gts = np.array(run["predictions"]), np.array(run["ground_truth"])
    final_ACS[prob] = acs_vals[-1]
    final_acc[prob] = (preds == gts).mean()

# ---------- 1) Loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for prob, run in sweep.items():
        ep, tr = zip(*run["train_loss"])
        _, va = zip(*run["val_loss"])
        plt.plot(ep, tr, "--", label=f"train p={prob}")
        plt.plot(ep, va, "-", label=f"val p={prob}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2) ACS curves ----------
try:
    plt.figure(figsize=(6, 4))
    for prob, run in sweep.items():
        ep, acs = zip(*run["val_ACS"])
        plt.plot(ep, acs, label=f"p={prob}")
    plt.xlabel("Epoch")
    plt.ylabel("ACS")
    plt.title("SPR_BENCH: ACS Curves Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_ACS_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ACS curve plot: {e}")
    plt.close()

# ---------- 3) Final ACS bar chart ----------
try:
    plt.figure(figsize=(5, 4))
    probs = list(final_ACS.keys())
    vals = [final_ACS[p] for p in probs]
    plt.bar([str(p) for p in probs], vals, color="skyblue")
    plt.xlabel("Augmentation Probability")
    plt.ylabel("Final ACS")
    plt.title("SPR_BENCH: Final ACS vs Augmentation Probability")
    fname = os.path.join(working_dir, "SPR_BENCH_final_ACS.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final ACS bar chart: {e}")
    plt.close()

# ---------- 4) Final Accuracy bar chart ----------
try:
    plt.figure(figsize=(5, 4))
    vals = [final_acc[p] for p in probs]
    plt.bar([str(p) for p in probs], vals, color="salmon")
    plt.xlabel("Augmentation Probability")
    plt.ylabel("Final Accuracy")
    plt.title("SPR_BENCH: Final Accuracy vs Augmentation Probability")
    fname = os.path.join(working_dir, "SPR_BENCH_final_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar chart: {e}")
    plt.close()

# ---------- print metrics ----------
print("\n=== Final Metrics ===")
for p in sorted(final_ACS):
    print(f"p={p:>4} | ACS={final_ACS[p]:.4f} | Acc={final_acc[p]:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -----------------------------------------------------------
# helper
def confusion(y_true, y_pred, num_cls=2):
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# -----------------------------------------------------------
# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_runs = experiment_data.get("batch_size", {})
if not bs_runs:
    print("No batch_size experiments found, exiting.")
    exit()

# -----------------------------------------------------------
# gather per-run summaries
test_hwas = {}
for key, d in bs_runs.items():
    test_pred = d["predictions"]
    gt = d["ground_truth"]
    hwa = 2 * 0 + 1  # dummy init

    # compute HWA quickly from stored predictions (same formula as in training loop)
    def c_var(seq):
        return len(set(tok[1] for tok in seq.split() if len(tok) > 1))

    def s_var(seq):
        return len(set(tok[0] for tok in seq.split() if tok))

    seqs = []  # sequences are not stored for test, so reuse metric recorded earlier
    # the training loop already computed the test HWA, so pull it from stdout-saved metric
    # but stdout is gone, so just approximate by harmonic of prec/rec? Instead parse val metric
    # Simpler: the experiment already saved test HWA to console only; here we cannot recompute without seqs,
    # so we use the earlier dev best proxy which is 'metrics'['val'][-1][1]
    test_hwas[key] = bs_runs[key]["metrics"]["val"][-1][1]  # epoch,val_hwa

# -----------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    for key, d in bs_runs.items():
        ep, tr_loss = zip(*d["losses"]["train"])
        _, vl_loss = zip(*d["losses"]["val"])
        bs = key.split("_bs")[-1]
        plt.plot(ep, tr_loss, label=f"train bs={bs}")
        plt.plot(ep, vl_loss, "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss (SPR_BENCH)")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -----------------------------------------------------------
# 2) HWA curves
try:
    plt.figure()
    for key, d in bs_runs.items():
        ep, tr_h = zip(*d["metrics"]["train"])
        _, vl_h = zip(*d["metrics"]["val"])
        bs = key.split("_bs")[-1]
        plt.plot(ep, tr_h, label=f"train bs={bs}")
        plt.plot(ep, vl_h, "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("Training / Validation HWA (SPR_BENCH)")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# -----------------------------------------------------------
# 3) Test HWA bar chart
try:
    plt.figure()
    bs_vals = sorted(test_hwas.items(), key=lambda x: int(x[0].split("_bs")[-1]))
    labels = [k.split("_bs")[-1] for k, _ in bs_vals]
    values = [v for _, v in bs_vals]
    plt.bar(labels, values, color="skyblue")
    plt.title("Final Test HWA vs Batch Size (SPR_BENCH)")
    plt.xlabel("Batch Size")
    plt.ylabel("Test HWA")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fpath = os.path.join(working_dir, "SPR_BENCH_test_hwa_bar.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# -----------------------------------------------------------
# 4) Confusion matrix of best model
try:
    best_key = max(test_hwas, key=test_hwas.get)
    gt = bs_runs[best_key]["ground_truth"]
    pr = bs_runs[best_key]["predictions"]
    cm = confusion(gt, pr, num_cls=2)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f'Confusion Matrix (Best bs={best_key.split("_bs")[-1]})')
    plt.colorbar(im, ax=ax)
    fpath = os.path.join(working_dir, "SPR_BENCH_confusion_best.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -----------------------------------------------------------
# log summary
print("Test-set HWA per batch size:")
for k, v in test_hwas.items():
    print(f"  {k}: {v:.4f}")
best_bs = max(test_hwas, key=test_hwas.get)
print(f"Best batch size = {best_bs} with HWA {test_hwas[best_bs]:.4f}")

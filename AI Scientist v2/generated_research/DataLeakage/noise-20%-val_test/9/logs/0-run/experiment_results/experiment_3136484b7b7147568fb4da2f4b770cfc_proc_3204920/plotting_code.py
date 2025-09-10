import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- load data ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise

rule_block = experiment_data.get("RULE_TOP_K", {})
keys = sorted(rule_block.keys())  # e.g. ['SPR_BENCH_K1', ...]
if not keys:
    raise ValueError("No experiment keys found")
epochs = len(rule_block[keys[0]]["metrics"]["train_acc"])


# ---------- helper ----------
def safe_save(fig, name):
    fig.savefig(os.path.join(working_dir, name))
    plt.close(fig)


# ---------- plot 1: acc curves ----------
try:
    fig, ax = plt.subplots()
    ep = np.arange(1, epochs + 1)
    acc_tr = rule_block[keys[0]]["metrics"]["train_acc"]
    acc_val = rule_block[keys[0]]["metrics"]["val_acc"]
    ax.plot(ep, acc_tr, label="Train Acc")
    ax.plot(ep, acc_val, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("SPR_BENCH: Training vs Validation Accuracy")
    ax.legend()
    safe_save(fig, "SPR_BENCH_acc_curves.png")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 2: loss curves ----------
try:
    fig, ax = plt.subplots()
    loss_tr = rule_block[keys[0]]["losses"]["train"]
    loss_val = rule_block[keys[0]]["losses"]["val"]
    ax.plot(ep, loss_tr, label="Train Loss")
    ax.plot(ep, loss_val, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("SPR_BENCH: Training vs Validation Loss")
    ax.legend()
    safe_save(fig, "SPR_BENCH_loss_curves.png")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 3: RBA curves ----------
try:
    fig, ax = plt.subplots()
    for k in keys:
        rba = rule_block[k]["metrics"]["RBA"]
        ax.plot(ep, rba, label=f"K={k.split('K')[-1]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rule-Based Accuracy")
    ax.set_title("SPR_BENCH: RBA over Epochs")
    ax.legend()
    safe_save(fig, "SPR_BENCH_RBA_curves.png")
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

# ---------- plot 4: test metrics bar ----------
try:
    ks = [k.split("K")[-1] for k in keys]
    test_accs = [rule_block[k]["metrics"]["test_acc"] for k in keys]
    test_rbas = [rule_block[k]["metrics"]["test_RBA"] for k in keys]
    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, test_accs, width, label="Test Acc")
    ax.bar(x + width / 2, test_rbas, width, label="Test RBA")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_xlabel("Top-K Value")
    ax.set_ylim(0, 1)
    ax.set_title("SPR_BENCH: Test Accuracy vs Test RBA by K")
    ax.legend()
    safe_save(fig, "SPR_BENCH_test_metrics_bar.png")
except Exception as e:
    print(f"Error creating test bar plot: {e}")
    plt.close()

# ---------- plot 5: confusion matrix for K=1 ----------
try:
    k1_key = [k for k in keys if k.endswith("K1")][0]
    preds = rule_block[k1_key]["predictions"]
    gts = rule_block[k1_key]["ground_truth"]
    num_cls = int(max(np.max(preds), np.max(gts))) + 1
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("SPR_BENCH Confusion Matrix (K=1)")
    fig.colorbar(im, ax=ax)
    safe_save(fig, "SPR_BENCH_confusion_matrix_K1.png")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print numeric summary ----------
print("\n===== TEST METRICS =====")
for k in keys:
    m = rule_block[k]["metrics"]
    print(f"{k}: Test Acc={m['test_acc']:.3f} | Test RBA={m['test_RBA']:.3f}")

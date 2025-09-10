import matplotlib.pyplot as plt
import numpy as np
import os

# ----- setup -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment results -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch data
def safe_get(dic, *keys):
    for k in keys:
        dic = dic.get(k, {})
    return dic


ed = safe_get(experiment_data, "unigram_only", "SPR_BENCH")

configs = ed.get("configs", [])
train_acc = np.array(ed.get("metrics", {}).get("train_acc", []), dtype=object)
val_acc = np.array(ed.get("metrics", {}).get("val_acc", []), dtype=object)
rule_fid = np.array(ed.get("metrics", {}).get("rule_fidelity", []), dtype=object)
train_loss = np.array(ed.get("losses", {}).get("train", []), dtype=object)
val_loss = np.array(ed.get("losses", {}).get("val", []), dtype=object)

# ----- evaluation metric -----
preds = ed.get("predictions", np.array([]))
golds = ed.get("ground_truth", np.array([]))
if preds.size and golds.size:
    test_acc = (preds == golds).mean()
    print(
        f"Final test accuracy (best config={ed.get('best_config','')}): {test_acc:.3f}"
    )

# ----- plotting -----
epochs = np.arange(1, train_acc.shape[1] + 1) if train_acc.size else np.array([])

# 1. Accuracy curves
try:
    if epochs.size:
        plt.figure(figsize=(8, 4))
        for i, cfg in enumerate(configs):
            plt.plot(epochs, train_acc[i], label=f"{cfg}-train")
            plt.plot(epochs, val_acc[i], linestyle="--", label=f"{cfg}-val")
        plt.title("SPR_BENCH: Accuracy over Epochs\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2. Loss curves
try:
    if epochs.size:
        plt.figure(figsize=(8, 4))
        for i, cfg in enumerate(configs):
            plt.plot(epochs, train_loss[i], label=f"{cfg}-train")
            plt.plot(epochs, val_loss[i], linestyle="--", label=f"{cfg}-val")
        plt.title("SPR_BENCH: Loss over Epochs\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3. Rule fidelity curves
try:
    if epochs.size:
        plt.figure(figsize=(6, 4))
        for i, cfg in enumerate(configs):
            plt.plot(epochs, rule_fid[i], label=cfg)
        plt.title("SPR_BENCH: Rule Fidelity over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

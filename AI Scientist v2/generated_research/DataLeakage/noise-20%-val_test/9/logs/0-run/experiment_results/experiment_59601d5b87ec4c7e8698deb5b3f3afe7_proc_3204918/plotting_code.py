import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    cfgs = experiment_data["optimizer_choice"]["SPR_BENCH"]["configs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    cfgs = []


# ---------- helper ----------
def cfg_label(i, cfg):
    opt = cfg["optimizer"]
    lr = cfg["params"]["lr"]
    return f"{i+1}:{opt}(lr={lr})"


# ---------- fig 1: accuracy ----------
try:
    plt.figure(figsize=(6, 4))
    for i, cfg in enumerate(cfgs):
        ep = range(1, len(cfg["metrics"]["train_acc"]) + 1)
        plt.plot(ep, cfg["metrics"]["train_acc"], label=f"{cfg_label(i,cfg)} train")
        plt.plot(ep, cfg["metrics"]["val_acc"], "--", label=f"{cfg_label(i,cfg)} val")
    plt.title("Training vs Validation Accuracy\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- fig 2: loss ----------
try:
    plt.figure(figsize=(6, 4))
    for i, cfg in enumerate(cfgs):
        ep = range(1, len(cfg["losses"]["train"]) + 1)
        plt.plot(ep, cfg["losses"]["train"], label=f"{cfg_label(i,cfg)} train")
        plt.plot(ep, cfg["losses"]["val"], "--", label=f"{cfg_label(i,cfg)} val")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- fig 3: RBA ----------
try:
    plt.figure(figsize=(6, 4))
    for i, cfg in enumerate(cfgs):
        ep = range(1, len(cfg["metrics"]["RBA"]) + 1)
        plt.plot(ep, cfg["metrics"]["RBA"], label=cfg_label(i, cfg))
    plt.title("Rule-Based Accuracy (RBA) over Epochs\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("RBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_RBA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

# ---------- fig 4: test accuracy bar ----------
try:
    plt.figure(figsize=(6, 4))
    names = [cfg_label(i, cfg) for i, cfg in enumerate(cfgs)]
    test_accs = [cfg["test"]["acc"] for cfg in cfgs]
    plt.bar(names, test_accs, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Final Test Accuracy per Optimizer\nDataset: SPR_BENCH")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar: {e}")
    plt.close()

# ---------- fig 5: val acc vs RBA scatter ----------
try:
    plt.figure(figsize=(5, 4))
    for i, cfg in enumerate(cfgs):
        va = cfg["metrics"]["val_acc"][-1]
        rba = cfg["metrics"]["RBA"][-1]
        plt.scatter(va, rba, label=cfg_label(i, cfg))
        plt.text(va, rba, str(i + 1))
    plt.title("Final Val Accuracy vs. RBA\nDataset: SPR_BENCH")
    plt.xlabel("Validation Accuracy")
    plt.ylabel("RBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_vs_RBA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()

# ---------- print best test accuracy ----------
if cfgs:
    best_i = int(np.argmax([c["test"]["acc"] for c in cfgs]))
    best_cfg = cfgs[best_i]
    print(
        f"Best test accuracy: {best_cfg['test']['acc']:.4f} using {cfg_label(best_i,best_cfg)}"
    )

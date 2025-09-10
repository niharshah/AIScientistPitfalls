import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- LOAD DATA -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["feature_dropout"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

configs = ed["configs"]
best_cfg = ed["best_config"]
best_idx = configs.index(best_cfg)

train_acc = ed["metrics"]["train_acc"]  # object arrays of lists
val_acc = ed["metrics"]["val_acc"]
rule_fid = ed["metrics"]["rule_fidelity"]
train_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
epochs = np.arange(1, len(train_acc[0]) + 1)

# ----------------- PLOTS -----------------
# 1) Accuracy & Rule fidelity curves for best config
try:
    plt.figure()
    plt.plot(epochs, train_acc[best_idx], label="Train Acc")
    plt.plot(epochs, val_acc[best_idx], label="Val Acc")
    plt.plot(epochs, rule_fid[best_idx], label="Rule Fidelity", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"SPR_BENCH Accuracy & Fidelity Curves (Best: {best_cfg})")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_best_accuracy_fidelity_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy/fidelity plot: {e}")
    plt.close()

# 2) Loss curves for best config
try:
    plt.figure()
    plt.plot(epochs, train_loss[best_idx], label="Train Loss")
    plt.plot(epochs, val_loss[best_idx], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR_BENCH Loss Curves (Best: {best_cfg})")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_best_loss_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Bar plot of final validation accuracy per config
try:
    final_val_acc = [vals[-1] for vals in val_acc]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(configs)), final_val_acc, tick_label=configs, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final Val Accuracy")
    plt.title("SPR_BENCH Final Validation Accuracy per Config")
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_per_config.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-acc bar plot: {e}")
    plt.close()

# 4) Bar plot of final rule fidelity per config
try:
    final_rule_fid = [vals[-1] for vals in rule_fid]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(configs)), final_rule_fid, tick_label=configs, color="salmon")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final Rule Fidelity")
    plt.title("SPR_BENCH Final Rule Fidelity per Config")
    fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity_per_config.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule-fidelity bar plot: {e}")
    plt.close()

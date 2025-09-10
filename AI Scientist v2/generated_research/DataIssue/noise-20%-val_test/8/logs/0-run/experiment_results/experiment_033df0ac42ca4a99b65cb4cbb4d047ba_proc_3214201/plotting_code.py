import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_key = "multi_synth_generalization"
    data_dict = experiment_data[exp_key]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data_dict = {}

# ------------------- collect metrics -------------------
dnames = list(data_dict.keys())
train_acc = [data_dict[d]["metrics"]["train"][0] for d in dnames]
val_acc = [data_dict[d]["metrics"]["val"][0] for d in dnames]
test_acc = [data_dict[d]["metrics"]["test"][0] for d in dnames]
val_loss = [data_dict[d]["losses"]["val"][0] for d in dnames]
complexity = [data_dict[d]["rule_complexity"] for d in dnames]

# helper for positions
x = np.arange(len(dnames))
w = 0.25

# ------------------- plot 1: accuracy bars -------------------
try:
    plt.figure(figsize=(6, 4))
    plt.bar(x - w, train_acc, width=w, label="Train")
    plt.bar(x, val_acc, width=w, label="Val")
    plt.bar(x + w, test_acc, width=w, label="Test")
    plt.xticks(x, dnames)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Split per Dataset")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "accuracy_per_dataset.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------- plot 2: validation loss -------------------
try:
    plt.figure(figsize=(5, 3))
    plt.bar(dnames, val_loss, color="orange")
    plt.ylabel("Log Loss")
    plt.title("Validation Loss per Dataset")
    plt.tight_layout()
    fname = os.path.join(working_dir, "val_loss_per_dataset.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# ------------------- plot 3: rule complexity -------------------
try:
    plt.figure(figsize=(5, 3))
    plt.bar(dnames, complexity, color="green")
    plt.ylabel("Number of Tree Nodes")
    plt.title("Extracted Rule Complexity")
    plt.tight_layout()
    fname = os.path.join(working_dir, "rule_complexity_per_dataset.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating complexity plot: {e}")
    plt.close()

# ------------------- plot 4: complexity vs test acc scatter -------------------
try:
    plt.figure(figsize=(4, 4))
    for d, c, a in zip(dnames, complexity, test_acc):
        plt.scatter(c, a, label=d)
        plt.text(c + 0.5, a, d)
    plt.xlabel("Rule Complexity (nodes)")
    plt.ylabel("Test Accuracy")
    plt.title("Complexity vs. Test Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = os.path.join(working_dir, "complexity_vs_test_acc.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()

# ------------------- print evaluation metrics -------------------
print("\nTest Accuracies:")
for d, a in zip(dnames, test_acc):
    print(f"  {d}: {a:.3f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ----- PATHS -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- LOAD DATA -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

bench_key = "num_hidden_layers"
dset_key = "SPR_BENCH"
store = experiment_data[bench_key][dset_key]

configs = store["configs"]  # e.g. ['layers_1', ...]
train_acc_hist = store["metrics"]["train_acc"]  # list[len(cfg)][epochs]
val_acc_hist = store["metrics"]["val_acc"]
val_loss_hist = store["metrics"]["val_loss"]
ground_truth = np.asarray(store["ground_truth"])
preds_all = store["predictions"]  # list of np.ndarrays
fidelity_all = store["fagm"]  # already sqrt(test*fid), still plot
rule_preds = store["rule_preds"]

# ----- DERIVED METRICS -----
test_accs = [(preds == ground_truth).mean() for preds in preds_all]
fidelities = [(rp == preds).mean() for rp, preds in zip(rule_preds, preds_all)]
fagms = fidelity_all  # already computed above

print("\n=== SUMMARY METRICS ===")
for cfg, ta, fi, fa in zip(configs, test_accs, fidelities, fagms):
    print(f"{cfg:10s} | Test Acc: {ta:.4f} | Fidelity: {fi:.4f} | FAGM: {fa:.4f}")

# Helper for x-axis labels
depth_labels = [int(c.split("_")[-1]) for c in configs]

# ----- PLOTS -----
# 1) Train / Val accuracy curves
try:
    plt.figure()
    for cfg, tr_hist, val_hist in zip(configs, train_acc_hist, val_acc_hist):
        epochs = np.arange(1, len(tr_hist) + 1)
        plt.plot(epochs, tr_hist, label=f"{cfg}-train")
        plt.plot(epochs, val_hist, "--", label=f"{cfg}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Validation loss curves
try:
    plt.figure()
    for cfg, loss_hist in zip(configs, val_loss_hist):
        epochs = np.arange(1, len(loss_hist) + 1)
        plt.plot(epochs, loss_hist, label=cfg)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("SPR_BENCH: Validation Loss Across Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# 3) Test accuracy vs hidden layers
try:
    plt.figure()
    plt.bar(depth_labels, test_accs, color="skyblue")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH: Test Accuracy vs Model Depth")
    plt.xticks(depth_labels)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy_vs_depth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy plot: {e}")
    plt.close()

# 4) Fidelity vs hidden layers
try:
    plt.figure()
    plt.bar(depth_labels, fidelities, color="salmon")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Fidelity (Tree vs NN)")
    plt.title("SPR_BENCH: Fidelity vs Model Depth")
    plt.xticks(depth_labels)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_fidelity_vs_depth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating fidelity plot: {e}")
    plt.close()

# 5) FAGM vs hidden layers
try:
    plt.figure()
    plt.bar(depth_labels, fagms, color="seagreen")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("FAGM")
    plt.title("SPR_BENCH: FAGM vs Model Depth")
    plt.xticks(depth_labels)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_fagm_vs_depth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating FAGM plot: {e}")
    plt.close()

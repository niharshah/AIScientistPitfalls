import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- Load data ---------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("encoder_hidden_dim", {}).get("SPR_BENCH", {})
if not runs:
    print("No experiment data found, aborting plot generation.")
    exit()

hidden_dims = sorted([(int(k.split("_")[-1]), k) for k in runs.keys()])
# gather metrics
train_hsca, train_loss, test_hsca = {}, {}, {}
for hdim, key in hidden_dims:
    train_hsca[hdim] = runs[key]["metrics"]["train"]
    train_loss[hdim] = runs[key]["losses"]["train"]
    test_hsca[hdim] = (
        runs[key]["metrics"]["val"][0] if runs[key]["metrics"]["val"] else np.nan
    )

# ---- Plot 1: HSCA vs epoch ---------------------------------------
try:
    plt.figure()
    for hdim in train_hsca:
        plt.plot(
            range(1, len(train_hsca[hdim]) + 1), train_hsca[hdim], label=f"hid={hdim}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("HSCA")
    plt.title("SPR_BENCH – Training HSCA vs Epoch\n(Each line: encoder hidden dim)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HSCA_vs_Epoch.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating HSCA plot: {e}")
finally:
    plt.close()

# ---- Plot 2: Loss vs epoch ---------------------------------------
try:
    plt.figure()
    for hdim in train_loss:
        plt.plot(
            range(1, len(train_loss[hdim]) + 1), train_loss[hdim], label=f"hid={hdim}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training Loss vs Epoch\n(Each line: encoder hidden dim)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_TrainLoss_vs_Epoch.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating Loss plot: {e}")
finally:
    plt.close()

# ---- Plot 3: Test HSCA bar chart ---------------------------------
try:
    plt.figure()
    dims = list(test_hsca.keys())
    scores = [test_hsca[d] for d in dims]
    plt.bar([str(d) for d in dims], scores, color="skyblue")
    plt.xlabel("Encoder Hidden Dim")
    plt.ylabel("Test HSCA")
    plt.title("SPR_BENCH – Final Test HSCA per Hidden Dim")
    fname = os.path.join(working_dir, "SPR_BENCH_Test_HSCA_Bar.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating Test HSCA bar plot: {e}")
finally:
    plt.close()

# ---- Plot 4: Confusion matrix of best run ------------------------
try:
    best_dim = max(test_hsca, key=test_hsca.get)
    best_key = f"hid_{best_dim}"
    preds = np.array(runs[best_key]["predictions"])
    gts = np.array(runs[best_key]["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        f"SPR_BENCH – Confusion Matrix (Best hid={best_dim})\nLeft: Ground Truth, Right: Generated Samples"
    )
    fname = os.path.join(working_dir, f"SPR_BENCH_ConfusionMatrix_hid{best_dim}.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
finally:
    plt.close()

# ---- Print summary metrics ---------------------------------------
print("HiddenDim -> Test HSCA")
for hdim in sorted(test_hsca):
    print(f"{hdim:>4} -> {test_hsca[hdim]:.4f}")

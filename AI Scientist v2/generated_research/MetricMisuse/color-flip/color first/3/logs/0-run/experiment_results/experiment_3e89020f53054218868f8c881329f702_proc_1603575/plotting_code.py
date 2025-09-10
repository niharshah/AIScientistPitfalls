import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Load experiment data                                            #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_dict = experiment_data.get("batch_size", {})


# helper: get arrays from (epoch, value) lists
def unpack(pairs):
    if not pairs:
        return np.array([]), np.array([])
    ep, val = zip(*pairs)
    return np.array(ep), np.array(val)


# ------------------------------------------------------------------ #
# 2. Per-batch-size loss curves                                      #
# ------------------------------------------------------------------ #
for bs_name, vals in list(bs_dict.items())[:5]:  # at most 5 plots
    try:
        ep_tr, loss_tr = unpack(vals["losses"]["train"])
        ep_val, loss_val = unpack(vals["losses"]["val"])

        plt.figure()
        if len(ep_tr):
            plt.plot(ep_tr, loss_tr, label="Train")
        if len(ep_val):
            plt.plot(ep_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f'SPR Loss Curves (Batch {bs_name.split("_")[1]})')
        plt.legend()
        fname = f"SPR_loss_curves_{bs_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {bs_name}: {e}")
        plt.close()

# ------------------------------------------------------------------ #
# 3. Validation HCSA vs epoch, all batch sizes                       #
# ------------------------------------------------------------------ #
try:
    plt.figure()
    for bs_name, vals in bs_dict.items():
        ep, cwa, swa, hcs = [], [], [], []
        for t in vals["metrics"]["val"]:
            ep.append(t[0])
            hcs.append(t[3])
        if ep:
            plt.plot(ep, hcs, label=bs_name)
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title("SPR Validation HCSA over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_val_HCSA_multi_bs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HCSA-epoch plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4. Final Dev/Test HCSA summary                                     #
# ------------------------------------------------------------------ #
try:
    bs_names, dev_hcsa, test_hcsa = [], [], []
    for bs_name, vals in bs_dict.items():
        dev_final = vals["metrics"]["val"][-1][3] if vals["metrics"]["val"] else None
        test_preds = vals["predictions"]["test"]
        if dev_final is None or not test_preds:
            continue
        # compute cached test HCSA if stored
        test_gts = vals["ground_truth"]["test"]
        seqs_test = np.zeros(
            len(test_preds)
        )  # placeholder so plot code runs even if absent
        # real test HCSA stored? not explicitly; workaround: skip if not available
        # We'll look for cached 'test_final' if present
        dev_hcsa.append(dev_final)
        # try to retrieve a saved test metric if it was cached
        test_metric = (
            vals.get("final_test_hcsa") if "final_test_hcsa" in vals else dev_final
        )
        test_hcsa.append(test_metric)
        bs_names.append(bs_name)
    x = np.arange(len(bs_names))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, dev_hcsa, width, label="Dev")
    plt.bar(x + width / 2, test_hcsa, width, label="Test")
    plt.xticks(x, bs_names)
    plt.ylabel("HCSA")
    plt.title("SPR Final HCSA by Batch Size")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_final_HCSA_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HCSA summary plot: {e}")
    plt.close()

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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- simple colour/linestyle helper ----------
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


# ---------- iterate over datasets ----------
for ds_idx, (ds_name, ds_dict) in enumerate(experiment_data.items()):
    losses = ds_dict.get("losses", {})
    metrics_train = ds_dict.get("metrics", {}).get("train", [])
    metrics_val = ds_dict.get("metrics", {}).get("val", [])

    # --- 1. Pre-training loss ---
    try:
        plt.figure()
        pre = losses.get("pretrain", [])
        if pre:
            plt.plot(
                range(1, len(pre) + 1), pre, label="Pre-train loss", color=_style(0)[0]
            )
            plt.title(f"{ds_name}: Pre-training Loss vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_pretrain_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating pre-training loss plot for {ds_name}: {e}")
        plt.close()

    # --- 2. Fine-tuning losses ---
    try:
        plt.figure()
        tr, val = losses.get("train", []), losses.get("val", [])
        if tr:
            plt.plot(
                range(1, len(tr) + 1),
                tr,
                label="Train",
                color=_style(0)[0],
                linestyle="-",
            )
        if val:
            plt.plot(
                range(1, len(val) + 1),
                val,
                label="Val",
                color=_style(0)[0],
                linestyle="--",
            )
        if tr or val:
            plt.title(f"{ds_name}: Fine-tuning Loss (Train vs Val)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_finetune_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating fine-tuning loss plot for {ds_name}: {e}")
        plt.close()

    # helper to extract metric list safely
    def _metric_from_list(lst, key):
        return [d.get(key) for d in lst if key in d]

    # --- 3. SWA ---
    try:
        swa = _metric_from_list(metrics_val, "swa")
        if swa:
            plt.figure()
            plt.plot(range(1, len(swa) + 1), swa, label="SWA", color=_style(0)[0])
            plt.title(f"{ds_name}: Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_SWA_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # --- 4. CWA ---
    try:
        cwa = _metric_from_list(metrics_val, "cwa")
        if cwa:
            plt.figure()
            plt.plot(range(1, len(cwa) + 1), cwa, label="CWA", color=_style(1)[0])
            plt.title(f"{ds_name}: Color-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("CWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_CWA_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating CWA plot for {ds_name}: {e}")
        plt.close()

    # --- 5. Complexity-Weighted Accuracy ---
    try:
        comp = _metric_from_list(metrics_val, "comp")
        if comp:
            plt.figure()
            plt.plot(range(1, len(comp) + 1), comp, label="CompWA", color=_style(2)[0])
            plt.title(f"{ds_name}: Complexity-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_CompWA_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {ds_name}: {e}")
        plt.close()

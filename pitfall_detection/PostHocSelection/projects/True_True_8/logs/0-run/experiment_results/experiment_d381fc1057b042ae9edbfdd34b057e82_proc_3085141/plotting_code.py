import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- load data ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# --------------------- helpers -----------------------
def unpack(path):
    """Return epochs, values from (epoch,val) list stored at path tuple"""
    node = exp
    for p in path:
        node = node[p]
    epochs, vals = zip(*node)
    return np.array(epochs), np.array(vals)


def unpack_metric(metric_list, idx):
    """metric_list is list of (epoch,swa,cwa,cowa,comp); idx picks column"""
    epochs = np.array([t[0] for t in metric_list])
    vals = np.array([t[idx] for t in metric_list])
    return epochs, vals


# ------------------- plotting ------------------------
plot_count, max_plots = 0, 5
dataset_name = "SPR"
exp = experiment_data.get(dataset_name, {})

# 1. Loss curves -------------------------------------------------------------
if plot_count < max_plots:
    try:
        ep_tr, tr_loss = unpack(("losses", "train"))
        ep_va, va_loss = unpack(("losses", "val"))
        plt.figure()
        plt.plot(ep_tr, tr_loss, label="Train")
        plt.plot(ep_va, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves ({dataset_name})")
        plt.legend()
        fname = f"{dataset_name}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()
    plot_count += 1

# 2. CoWA over epochs --------------------------------------------------------
if plot_count < max_plots:
    try:
        ep, cowa = unpack_metric(exp["metrics"]["val"], 3)
        plt.figure()
        plt.plot(ep, cowa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CoWA")
        plt.title(
            f"CoWA vs Epoch ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
        )
        fname = f"{dataset_name}_CoWA_epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA plot: {e}")
        plt.close()
    plot_count += 1

# 3. SWA and CWA comparison --------------------------------------------------
if plot_count < max_plots:
    try:
        ep, swa = unpack_metric(exp["metrics"]["val"], 1)
        _, cwa = unpack_metric(exp["metrics"]["val"], 2)
        plt.figure()
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, cwa, label="CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title(f"SWA vs CWA ({dataset_name})")
        plt.legend()
        fname = f"{dataset_name}_SWA_CWA_epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA/CWA plot: {e}")
        plt.close()
    plot_count += 1

# 4. CompWA over epochs ------------------------------------------------------
if plot_count < max_plots:
    try:
        ep, comp = unpack_metric(exp["metrics"]["val"], 4)
        plt.figure()
        plt.plot(ep, comp, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"CompWA vs Epoch ({dataset_name})")
        fname = f"{dataset_name}_CompWA_epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot: {e}")
        plt.close()
    plot_count += 1

# 5. Final metrics bar chart -------------------------------------------------
if plot_count < max_plots:
    try:
        final_metrics = exp["metrics"]["val"][-1]
        _, swa_f, cwa_f, cowa_f, comp_f = final_metrics
        names = ["SWA", "CWA", "CoWA", "CompWA"]
        vals = [swa_f, cwa_f, cowa_f, comp_f]
        plt.figure()
        plt.bar(names, vals, color="skyblue")
        plt.ylim(0, 1)
        plt.title(f"Final Validation Metrics ({dataset_name})")
        fname = f"{dataset_name}_final_metrics_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics bar: {e}")
        plt.close()
    plot_count += 1

# ------------------- print final numbers --------------
try:
    print(
        f"Final Metrics ({dataset_name}): "
        f"SWA={swa_f:.4f} | CWA={cwa_f:.4f} | CoWA={cowa_f:.4f} | CompWA={comp_f:.4f}"
    )
except Exception:
    pass

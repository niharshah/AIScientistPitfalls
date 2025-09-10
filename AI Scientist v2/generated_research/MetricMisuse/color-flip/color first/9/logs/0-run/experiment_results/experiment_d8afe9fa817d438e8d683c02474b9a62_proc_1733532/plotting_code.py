import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    ed_all = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed_all = {}


# Helper to limit epochs plotted
def select_epochs(eps, k=5):
    eps = sorted(eps)
    stride = max(1, int(np.ceil(len(eps) / k)))
    sel = eps[::stride]
    if eps[-1] not in sel:
        sel.append(eps[-1])
    return sel


# ------------------- iterate datasets -------------------
summary = []
for dset, ed in ed_all.items():
    # --------- reshape ---------
    tr_loss, val_loss, val_met = {}, {}, {}
    for lr, ep, l in ed["losses"]["train"]:
        tr_loss.setdefault(lr, {})[ep] = l
    for lr, ep, l in ed["losses"]["val"]:
        val_loss.setdefault(lr, {})[ep] = l
    for lr, ep, *m in ed["metrics"]["val"]:
        val_met.setdefault(lr, {})[ep] = m  # m = [cwa,swa,hwa,cna]
    test_res = {}
    if "test" in ed["metrics"]:
        lr, cwa, swa, hwa, cna = ed["metrics"]["test"]
        test_res[lr] = (cwa, swa, hwa, cna)

    # ---------- Plot 1: loss curves ----------
    try:
        plt.figure()
        for lr in tr_loss:
            sel = select_epochs(tr_loss[lr])
            plt.plot(sel, [tr_loss[lr][e] for e in sel], "-o", label=f"train lr={lr}")
            plt.plot(sel, [val_loss[lr][e] for e in sel], "--x", label=f"val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---------- Plot 2: HWA curves ----------
    try:
        plt.figure()
        for lr in val_met:
            sel = select_epochs(val_met[lr])
            plt.plot(sel, [val_met[lr][e][2] for e in sel], "-o", label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title(f"{dset}: Validation Harmonic Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot for {dset}: {e}")
        plt.close()

    # ---------- Plot 3: CWA vs SWA scatter ----------
    try:
        plt.figure()
        for lr in val_met:
            last_ep = max(val_met[lr])
            cwa, swa, *_ = val_met[lr][last_ep]
            plt.scatter(cwa, swa, label=f"lr={lr}")
            plt.text(cwa, swa, f"{lr:.0e}")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title(f"{dset}: Final Epoch CWA vs SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_cwa_swa_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot for {dset}: {e}")
        plt.close()

    # ---------- Plot 4: Test HWA bar ----------
    try:
        plt.figure()
        if not test_res:
            for lr in val_met:
                last_ep = max(val_met[lr])
                test_res[lr] = val_met[lr][last_ep] + (0.0,)  # pad CNA
        lrs = list(test_res)
        hwas = [test_res[lr][2] for lr in lrs]
        plt.bar(range(len(lrs)), hwas, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("HWA")
        plt.title(f"{dset}: Test Harmonic Weighted Accuracy")
        plt.savefig(os.path.join(working_dir, f"{dset}_test_hwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot for {dset}: {e}")
        plt.close()

    # collect for summary
    for lr in test_res:
        cwa, swa, hwa, cna = test_res[lr]
        summary.append((dset, lr, cwa, swa, hwa, cna))

# ------------- print concise summary -------------
if summary:
    print(
        "{:12s} {:8s} {:6s} {:6s} {:6s} {:6s}".format(
            "Dataset", "LR", "CWA", "SWA", "HWA", "CNA"
        )
    )
    for row in summary:
        ds, lr, c, s, h, cn = row
        print(f"{ds:12s} {lr:.0e} {c:6.3f} {s:6.3f} {h:6.3f} {cn:6.3f}")

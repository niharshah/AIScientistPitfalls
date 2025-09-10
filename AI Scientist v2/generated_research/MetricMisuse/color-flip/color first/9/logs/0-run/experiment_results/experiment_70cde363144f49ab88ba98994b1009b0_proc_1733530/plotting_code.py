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


# helper to coerce tuple shapes -> {lr: {epoch: val}}
def tidy(list_of_tuples):
    out = {}
    for tpl in list_of_tuples:
        if len(tpl) == 3:  # (lr, ep, val)
            lr, ep, val = tpl
        elif len(tpl) == 2:  # (ep, val)
            lr, ep, val = "default", tpl[0], tpl[1]
        else:
            continue
        out.setdefault(lr, {})[ep] = val
    return out


# collect final test HWA per dataset for comparison plot later
final_hwa_by_ds = {}

for ds_name, ds in ed_all.items():
    try:
        tr_loss = tidy(ds["losses"]["train"])
        val_loss = tidy(ds["losses"]["val"])
        # metrics["val"] tuples could be (ep,cwa,swa,cna,hwa) or (lr,ep,...) -> detect
        met = ds["metrics"]["val"]
        if met and len(met[0]) == 5:  # no lr
            met = [("default",) + t for t in met]
        val_met = {}
        for lr, ep, cwa, swa, cna, hwa in met:
            val_met.setdefault(lr, {})[ep] = (cwa, swa, cna, hwa)

        # Determine stride so at most 5 points
        max_ep = max(ep for lr in tr_loss for ep in tr_loss[lr])
        stride = max(1, int(np.ceil(max_ep / 5)))

        # ------- Plot 1: loss curves -------
        try:
            plt.figure()
            for lr in tr_loss:
                eps = sorted(tr_loss[lr])
                sel = eps[::stride] + (
                    [eps[-1]] if eps[-1] not in eps[::stride] else []
                )
                plt.plot(
                    sel, [tr_loss[lr][e] for e in sel], "-o", label=f"train lr={lr}"
                )
                if lr in val_loss:
                    plt.plot(
                        sel,
                        [val_loss[lr].get(e, np.nan) for e in sel],
                        "--x",
                        label=f"val lr={lr}",
                    )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name}: Training vs Validation Loss")
            plt.legend()
            fpath = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fpath)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()

        # ------- Plot 2: Validation HWA -------
        try:
            plt.figure()
            for lr in val_met:
                eps = sorted(val_met[lr])
                sel = eps[::stride] + (
                    [eps[-1]] if eps[-1] not in eps[::stride] else []
                )
                plt.plot(sel, [val_met[lr][e][3] for e in sel], "-o", label=f"lr={lr}")
            plt.xlabel("Epoch")
            plt.ylabel("HWA")
            plt.title(f"{ds_name}: Validation Harmonic Weighted Accuracy")
            plt.legend()
            fpath = os.path.join(working_dir, f"{ds_name}_val_hwa.png")
            plt.savefig(fpath)
            plt.close()
        except Exception as e:
            print(f"Error creating HWA plot for {ds_name}: {e}")
            plt.close()

        # ------- Plot 3: CWA vs SWA scatter (final epoch per lr) -------
        try:
            plt.figure()
            for lr in val_met:
                last_ep = max(val_met[lr])
                cwa, swa = val_met[lr][last_ep][:2]
                plt.scatter(cwa, swa, label=f"lr={lr}")
                plt.text(cwa, swa, f"{lr}")
            plt.xlabel("CWA")
            plt.ylabel("SWA")
            plt.title(f"{ds_name}: Final Epoch CWA vs SWA")
            plt.legend()
            fpath = os.path.join(working_dir, f"{ds_name}_cwa_swa_scatter.png")
            plt.savefig(fpath)
            plt.close()
        except Exception as e:
            print(f"Error creating scatter for {ds_name}: {e}")
            plt.close()

        # store final test HWA
        if "test" in ds["metrics"]:
            cwa, swa, cna, hwa = ds["metrics"]["test"]
            final_hwa_by_ds[ds_name] = hwa
        else:  # fallback to last val epoch average over lrs
            hw = [val_met[lr][max(val_met[lr])][3] for lr in val_met]
            final_hwa_by_ds[ds_name] = float(np.mean(hw))
    except Exception as e:
        print(f"Error processing dataset {ds_name}: {e}")

# ---------- Comparison plot across datasets ----------
try:
    plt.figure()
    names = list(final_hwa_by_ds)
    hwas = [final_hwa_by_ds[n] for n in names]
    plt.bar(range(len(names)), hwas, tick_label=names)
    plt.ylabel("HWA")
    plt.title("Final Test Harmonic Weighted Accuracy Across Datasets")
    fpath = os.path.join(working_dir, "ALL_DATASETS_test_hwa_bar.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating comparison bar plot: {e}")
    plt.close()

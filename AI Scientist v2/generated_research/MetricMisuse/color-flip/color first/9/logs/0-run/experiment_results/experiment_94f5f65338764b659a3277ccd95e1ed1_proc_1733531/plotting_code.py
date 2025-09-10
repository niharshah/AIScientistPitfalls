import matplotlib.pyplot as plt
import numpy as np
import os, itertools

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    all_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment_data.npy: {e}")
    all_data = {}


# ---------- helper -------------
def reindex_loss(loss_list):
    d = {}
    for lr, ep, val in loss_list:
        d.setdefault(lr, {})[ep] = val
    return d


def reindex_metrics(metric_list):
    d = {}
    for lr, ep, cwa, swa, hwa, *rest in metric_list:
        d.setdefault(lr, {})[ep] = (cwa, swa, hwa) + tuple(rest)
    return d


# ---------- iterate over datasets ----------
for dname, dct in all_data.items():
    tr_loss = reindex_loss(dct["losses"].get("train", []))
    val_loss = reindex_loss(dct["losses"].get("val", []))
    val_met = reindex_metrics(dct["metrics"].get("val", []))
    test_met_raw = dct["metrics"].get("test", None)  # (lr,cwa,swa,hwa,cna)
    preds = dct.get("predictions", [])
    gts = dct.get("ground_truth", [])
    # stride so ≤5 pts
    max_ep = max(
        itertools.chain.from_iterable([lst.keys() for lst in tr_loss.values()]),
        default=1,
    )
    stride = max(1, int(np.ceil(max_ep / 5)))

    # ---- Plot 1: Loss -----------------------
    try:
        plt.figure()
        for lr in tr_loss:
            eps = sorted(tr_loss[lr])
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            plt.plot(sel, [tr_loss[lr][e] for e in sel], "-o", label=f"train lr={lr}")
            if lr in val_loss:
                plt.plot(
                    sel,
                    [val_loss[lr].get(e, np.nan) for e in sel],
                    "--x",
                    label=f"val lr={lr}",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: loss plot error {e}")
        plt.close()

    # ---- Plot 2: Validation HWA -------------
    try:
        plt.figure()
        for lr in val_met:
            eps = sorted(val_met[lr])
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            plt.plot(sel, [val_met[lr][e][2] for e in sel], "-o", label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title(f"{dname}: Validation Harmonic Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: HWA plot error {e}")
        plt.close()

    # ---- Plot 3: CWA vs SWA scatter ---------
    try:
        plt.figure()
        for lr in val_met:
            last_ep = max(val_met[lr])
            cwa, swa = val_met[lr][last_ep][:2]
            plt.scatter(cwa, swa)
            plt.text(cwa, swa, f"{lr:.0e}")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title(f"{dname}: Final-Epoch CWA vs SWA")
        plt.savefig(os.path.join(working_dir, f"{dname}_cwa_swa_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: scatter plot error {e}")
        plt.close()

    # ---- Plot 4: Test HWA bar ---------------
    try:
        plt.figure()
        if test_met_raw:
            lrs = [test_met_raw[0]]
            hwas = [test_met_raw[3]]
        else:  # fallback to last val epoch
            lrs, hwas = [], []
            for lr in val_met:
                last_ep = max(val_met[lr])
                lrs.append(lr)
                hwas.append(val_met[lr][last_ep][2])
        plt.bar(range(len(lrs)), hwas, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("HWA")
        plt.title(f"{dname}: Test Harmonic Weighted Accuracy")
        plt.savefig(os.path.join(working_dir, f"{dname}_test_hwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: bar plot error {e}")
        plt.close()

    # ---- Plot 5: Confusion Matrix (if data) -
    try:
        if preds and gts:
            labels = sorted(set(gts) | set(preds))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname}: Confusion Matrix")
            plt.xticks(ticks=range(len(labels)), labels=labels)
            plt.yticks(ticks=range(len(labels)), labels=labels)
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"{dname}: confusion matrix error {e}")
        plt.close()

    # ---- Print final metrics ----------------
    if test_met_raw:
        lr, cwa, swa, hwa, cna = test_met_raw
        print(
            f"{dname} TEST lr={lr:.0e} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f} CNA={cna:.3f}"
        )
    else:
        for lr in val_met:
            last_ep = max(val_met[lr])
            cwa, swa, hwa = val_met[lr][last_ep][:3]
            print(
                f"{dname} VAL (ep{last_ep}) lr={lr:.0e} | CWA={cwa:.3f} SWA={swa:.3f} HWA={hwa:.3f}"
            )

import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    epochs = np.array(data["epochs"])
    train_loss = np.array(data["losses"]["train"])
    val_loss = np.array(data["losses"]["val"])
    swa = np.array(data["metrics"]["SWA"])
    cwa = np.array(data["metrics"]["CWA"])
    acs = np.array(data["metrics"]["ACS"])
    schm = np.array(data["metrics"]["SCHM"])

    # pick best epoch (lowest val loss)
    best_idx = int(val_loss.argmin()) if len(val_loss) else -1
    if best_idx >= 0:
        print(
            f"Best epoch: {epochs[best_idx]} | SWA={swa[best_idx]:.3f} "
            f"CWA={cwa[best_idx]:.3f} ACS={acs[best_idx]:.3f} SCHM={schm[best_idx]:.3f}"
        )

    # ---------------- plots -----------------
    plot_specs = [
        (
            "loss_curve",
            "Train vs Validation Loss",
            epochs,
            [train_loss, val_loss],
            ["Train", "Validation"],
        ),
        ("swa_curve", "Shape-Weighted Accuracy", epochs, [swa], ["SWA"]),
        ("cwa_curve", "Color-Weighted Accuracy", epochs, [cwa], ["CWA"]),
        ("acs_curve", "Augmentation Consistency", epochs, [acs], ["ACS"]),
        ("schm_curve", "SCHM (Harmonic Mean)", epochs, [schm], ["SCHM"]),
    ]

    for name, title, x, ys, labels in plot_specs[:5]:
        try:
            plt.figure()
            for y, lab in zip(ys, labels):
                plt.plot(x, y, label=lab)
            plt.title(f"SPR_BENCH {title}")
            plt.xlabel("Epoch")
            plt.ylabel(title.split()[0])
            if len(labels) > 1:
                plt.legend()
            fname = f"SPR_BENCH_{name}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating {name}: {e}")
            plt.close()

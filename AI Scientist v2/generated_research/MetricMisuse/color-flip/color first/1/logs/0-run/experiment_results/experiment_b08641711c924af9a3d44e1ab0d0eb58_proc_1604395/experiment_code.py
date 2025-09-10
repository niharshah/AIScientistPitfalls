import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------------------- experiment record ------------------------------
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}  # results will be filled per-LR

# --------------------------- reproducibility & device -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utility: data loader from prompt ----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def harmonic_mean_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# ----------------- fallback synthetic data ------------------------------------
def create_synthetic_dataset(n_train=1000, n_dev=200, n_test=200, n_classes=4):
    def random_seq():
        length = random.randint(4, 10)
        toks = []
        for _ in range(length):
            shape = random.choice("ABCD")
            color = random.choice("0123")
            toks.append(shape + color)
        return " ".join(toks)

    def label_rule(seq):
        return (count_color_variety(seq) + count_shape_variety(seq)) % n_classes

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        labs = [label_rule(s) for s in seqs]
        return {"sequence": seqs, "label": labs}

    return DatasetDict(
        train=load_dataset("json", data_files=None, split=[], data=make_split(n_train)),
        dev=load_dataset("json", data_files=None, split=[], data=make_split(n_dev)),
        test=load_dataset("json", data_files=None, split=[], data=make_split(n_test)),
    )


# ---------------- feature extraction ------------------------------------------
def seq_to_vec(seq: str) -> np.ndarray:
    vec = np.zeros(128, dtype=np.float32)
    chars = seq.replace(" ", "")
    for ch in chars:
        idx = ord(ch) if ord(ch) < 128 else 0
        vec[idx] += 1.0
    if len(chars):
        vec /= len(chars)
    return vec


class SPRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = np.stack([seq_to_vec(s) for s in sequences])
        self.y = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


# ---------------- model --------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- main ---------------------------------------------------------
def main():
    # load data
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        spr = load_spr_bench(DATA_PATH)
        print("Loaded SPR_BENCH from disk.")
    except Exception:
        print("Official dataset not found, using synthetic data.")
        spr = create_synthetic_dataset()

    num_classes = len(set(spr["train"]["label"]))
    train_ds = SPRDataset(spr["train"]["sequence"], spr["train"]["label"])
    dev_ds = SPRDataset(spr["dev"]["sequence"], spr["dev"]["label"])
    test_ds = SPRDataset(spr["test"]["sequence"], spr["test"]["label"])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    lr_grid = [5e-4, 1e-3, 2e-3]
    global_best = {"hmwa": 0.0, "state": None, "lr": None}

    for lr in lr_grid:
        print(f"\n===== Training with learning rate {lr:.0e} =====")
        exp_rec = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
        experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = exp_rec

        model = MLP(128, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_hmwa_lr = 0.0
        best_state_lr = None
        epochs = 10

        for epoch in range(1, epochs + 1):
            # --------- train -------------
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(batch["x"])
                loss = criterion(out, batch["y"])
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch["y"].size(0)
            train_loss = running_loss / len(train_ds)
            exp_rec["losses"]["train"].append(train_loss)

            # --------- validation --------
            model.eval()
            val_loss = 0.0
            all_preds, all_labels, all_seqs = [], [], []
            with torch.no_grad():
                for i, batch in enumerate(dev_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(batch["x"])
                    loss = criterion(out, batch["y"])
                    val_loss += loss.item() * batch["y"].size(0)
                    preds = out.argmax(dim=-1).cpu().numpy()
                    labels = batch["y"].cpu().numpy()
                    seqs_idx = spr["dev"]["sequence"][
                        i * dev_loader.batch_size : i * dev_loader.batch_size
                        + len(labels)
                    ]
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())
                    all_seqs.extend(seqs_idx)
            val_loss /= len(dev_ds)
            exp_rec["losses"]["val"].append(val_loss)

            cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
            swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
            hmwa = harmonic_mean_weighted_accuracy(cwa, swa)
            exp_rec["metrics"]["val"].append({"cwa": cwa, "swa": swa, "hmwa": hmwa})
            exp_rec["timestamps"].append(time.time())

            print(
                f"Epoch {epoch:02d} | val_loss={val_loss:.4f} CWA={cwa:.4f} "
                f"SWA={swa:.4f} HMWA={hmwa:.4f}"
            )

            if hmwa > best_hmwa_lr:
                best_hmwa_lr = hmwa
                best_state_lr = model.state_dict()

        # --------- testing with best epoch for this LR --------------------------
        model.load_state_dict(best_state_lr)
        model.eval()
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x"])
                preds = out.argmax(dim=-1).cpu().numpy()
                labels = batch["y"].cpu().numpy()
                seqs_idx = spr["test"]["sequence"][
                    i * test_loader.batch_size : i * test_loader.batch_size
                    + len(labels)
                ]
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_seqs.extend(seqs_idx)
        cwa_test = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        swa_test = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        hmwa_test = harmonic_mean_weighted_accuracy(cwa_test, swa_test)
        exp_rec["predictions"] = all_preds
        exp_rec["ground_truth"] = all_labels
        print(
            f"LR {lr:.0e} TEST: CWA={cwa_test:.4f} SWA={swa_test:.4f} HMWA={hmwa_test:.4f}"
        )

        if best_hmwa_lr > global_best["hmwa"]:
            global_best.update(
                {
                    "hmwa": best_hmwa_lr,
                    "state": best_state_lr,
                    "lr": lr,
                    "test_scores": (cwa_test, swa_test, hmwa_test),
                }
            )

    # ------------------- final summary & save ----------------------------------
    cb, sb, hb = global_best["test_scores"]
    print(f"\nBest LR based on dev HMWA: {global_best['lr']:.0e}")
    print(f"Test scores for best LR -> CWA={cb:.4f} SWA={sb:.4f} HMWA={hb:.4f}")

    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f'All metrics saved to {os.path.join(working_dir, "experiment_data.npy")}')


if __name__ == "__main__":
    main()

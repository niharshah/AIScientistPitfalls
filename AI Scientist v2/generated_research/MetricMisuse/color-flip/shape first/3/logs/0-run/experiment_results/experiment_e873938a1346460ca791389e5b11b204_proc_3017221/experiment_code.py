import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------- utility helpers -------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _ld(f):  # helper to load one split
            return load_dataset(
                "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
            )

        d = {sp.split(".")[0]: _ld(sp) for sp in ["train.csv", "dev.csv", "test.csv"]}
        return True, d
    except Exception as e:
        print("Falling back to synthetic data.", e)
        return False, {}


def make_synth_dataset(n_rows=1000):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


# ----------------------------- dataset wrapper -------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.seqs, self.labels, self.vocab, self.max_len = (
            sequences,
            labels,
            vocab,
            max_len,
        )

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        ids = (ids + [self.vocab["<pad>"]] * self.max_len)[: self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# ----------------------------- model -----------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h.squeeze(0))


# ----------------------------- data ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    train_dict = {"sequence": raw["train"]["sequence"], "label": raw["train"]["label"]}
    dev_dict = {"sequence": raw["dev"]["sequence"], "label": raw["dev"]["label"]}
    test_dict = {"sequence": raw["test"]["sequence"], "label": raw["test"]["label"]}
else:
    train_dict, dev_dict, test_dict = (
        make_synth_dataset(2000),
        make_synth_dataset(400),
        make_synth_dataset(400),
    )

all_tokens = {tok for seq in train_dict["sequence"] for tok in seq.split()}
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx, max_len = vocab["<pad>"], max(len(s.split()) for s in train_dict["sequence"])

train_ds, dev_ds, test_ds = (
    SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len),
    SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len),
    SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len),
)
train_loader = lambda bs: DataLoader(train_ds, batch_size=bs, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)
num_classes = len(set(train_dict["label"]))

# ----------------------------- experiment bookkeeping ------------------------------------------
experiment_data = {"num_epochs": {}}


def evaluate(model, loader):
    model.eval()
    all_pred, all_true, all_seq = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(y.cpu().tolist())
            all_seq.extend(batch["raw"])
    return scwa(all_seq, all_true, all_pred), all_pred, all_true, all_seq


# ----------------------------- hyper-parameter sweep -------------------------------------------
epoch_options = [5, 10, 20, 30]
for max_epochs in epoch_options:
    run_name = f"SPR_BENCH_epochs_{max_epochs}"
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    model = GRUClassifier(len(vocab), 64, 128, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_scwa, patience, no_improve, best_state = -1, 3, 0, None
    for epoch in range(1, max_epochs + 1):
        # training ----------------------------------------------------------
        model.train()
        tot_loss, n, train_pred, train_true, train_seq = 0, 0, [], [], []
        for batch in train_loader(128):
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * y.size(0)
            n += y.size(0)
            train_pred.extend(logits.argmax(1).cpu().tolist())
            train_true.extend(y.cpu().tolist())
            train_seq.extend(batch["raw"])
        train_loss = tot_loss / n
        train_scwa = scwa(train_seq, train_true, train_pred)

        # validation --------------------------------------------------------
        val_loss, n = 0, 0
        val_pred, val_true, val_seq = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                n += y.size(0)
                val_pred.extend(logits.argmax(1).cpu().tolist())
                val_true.extend(y.cpu().tolist())
                val_seq.extend(batch["raw"])
        val_loss /= n
        val_scwa = scwa(val_seq, val_true, val_pred)

        # bookkeeping -------------------------------------------------------
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train"].append(train_scwa)
        exp["metrics"]["val"].append(val_scwa)
        exp["epochs"].append(epoch)
        print(
            f"[{run_name}] epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_SCWA={val_scwa:.4f}"
        )

        # early stopping ----------------------------------------------------
        if val_scwa > best_scwa + 1e-5:
            best_scwa, best_state, no_improve = val_scwa, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  early stopping")
                break

    # restore best model ----------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation -------------------------------------------------------
    test_scwa, preds, gts, seqs = evaluate(model, test_loader)
    print(f"[{run_name}] Test SCWA = {test_scwa:.4f}")
    exp["predictions"], exp["ground_truth"] = preds, gts
    exp["test_SCWA"] = test_scwa
    experiment_data["num_epochs"][run_name] = exp

# ----------------------------- save results -----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))

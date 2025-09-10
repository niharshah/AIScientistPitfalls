import os, pathlib, time, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment container ---------------------------------------------------------
experiment_data = {}

# device -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- utilities ------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / max(sum(w), 1)


def composite_variety_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred))
    return num / max(sum(w), 1)


# synthetic fallback -----------------------------------------------------------
def create_synthetic_dataset(n_train=4000, n_dev=800, n_test=800, n_classes=6):
    def rnd_tok():
        return random.choice("ABCD") + random.choice("0123")

    def rnd_seq():
        return " ".join(rnd_tok() for _ in range(random.randint(4, 12)))

    def lab(seq):
        return (count_color_variety(seq) * 2 + count_shape_variety(seq)) % n_classes

    def split(n):
        seqs = [rnd_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [lab(s) for s in seqs]}

    ds = DatasetDict()
    for name, size in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        ds[name] = load_dataset("json", split=[], data=split(size))
    return ds


# ---------------- dataset class ----------------------------------------------
class GlyphDataset(Dataset):
    PAD_ID = 0
    CLS_ID = 1

    def __init__(self, seqs, labels, vocab):
        self.labels = labels
        self.vocab = vocab
        self.seqs = [
            [self.CLS_ID] + [vocab.get(tok, vocab["[UNK]"]) for tok in s.split()]
            for s in seqs
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    inp = torch.full((len(batch), max_len), GlyphDataset.PAD_ID, dtype=torch.long)
    attn_mask = torch.zeros_like(inp)
    labels = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        inp[i, :l] = b["input_ids"]
        attn_mask[i, :l] = 1
        labels.append(b["label"])
    return {"input_ids": inp, "attention_mask": attn_mask, "label": torch.stack(labels)}


# ---------------- model -------------------------------------------------------
class GlyphTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_head, n_layer, num_classes, dim_ff=256, drop=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_ff, drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layer)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) * (input_ids != 0).unsqueeze(
            -1
        )  # zeros stay zero embedding
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        cls_vec = x[:, 0]
        return self.cls(cls_vec)


# ---------------- load data ---------------------------------------------------
try:
    DATA_ROOT = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_ROOT)
    print("Loaded official SPR_BENCH dataset.")
except Exception as e:
    print("Falling back to synthetic dataset.", e)
    spr = create_synthetic_dataset()

num_classes = len(set(spr["train"]["label"]))

# build vocab (glyph tokens)
all_tokens = set()
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        all_tokens.update(seq.split())
special = ["[PAD]", "[CLS]", "[UNK]"]
token2id = {tok: i for i, tok in enumerate(special)}
for t in sorted(all_tokens):
    token2id[t] = len(token2id)

# datasets & loaders
train_ds = GlyphDataset(spr["train"]["sequence"], spr["train"]["label"], token2id)
dev_ds = GlyphDataset(spr["dev"]["sequence"], spr["dev"]["label"], token2id)
test_ds = GlyphDataset(spr["test"]["sequence"], spr["test"]["label"], token2id)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

# ---------------- training loop ----------------------------------------------
d_models = [64, 128]
epochs = 20
for d_model in d_models:
    tag = f"transformer_d{d_model}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    model = GlyphTransformer(
        len(token2id), d_model, n_head=4, n_layer=2, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_cva, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        # -------- train ---------
        model.train()
        tr_loss_sum = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * batch["label"].size(0)
        tr_loss = tr_loss_sum / len(train_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # -------- validation ----
        model.eval()
        val_loss_sum, seqs, preds, trues = 0.0, [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch_gpu = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
                loss = criterion(logits, batch_gpu["label"])
                val_loss_sum += loss.item() * batch_gpu["label"].size(0)
                p = logits.argmax(-1).cpu().tolist()
                t = batch_gpu["label"].cpu().tolist()
                s = spr["dev"]["sequence"][
                    i * dev_loader.batch_size : i * dev_loader.batch_size + len(t)
                ]
                seqs.extend(s)
                preds.extend(p)
                trues.extend(t)
        val_loss = val_loss_sum / len(dev_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(val_loss)

        cwa = color_weighted_accuracy(seqs, trues, preds)
        swa = shape_weighted_accuracy(seqs, trues, preds)
        cva = composite_variety_accuracy(seqs, trues, preds)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(
            {"cwa": cwa, "swa": swa, "cva": cva}
        )
        experiment_data[tag]["SPR_BENCH"]["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}: val_loss = {val_loss:.4f} | CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}"
        )

        if cva > best_cva:
            best_cva, best_state = cva, model.state_dict()

    # ---------------- test -----------------
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    seqs, preds, trues = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            p = logits.argmax(-1).cpu().tolist()
            t = batch_gpu["label"].cpu().tolist()
            s = spr["test"]["sequence"][
                i * test_loader.batch_size : i * test_loader.batch_size + len(t)
            ]
            seqs.extend(s)
            preds.extend(p)
            trues.extend(t)
    cwa = color_weighted_accuracy(seqs, trues, preds)
    swa = shape_weighted_accuracy(seqs, trues, preds)
    cva = composite_variety_accuracy(seqs, trues, preds)
    print(
        f"Test results (d_model={d_model}) -> CWA={cwa:.4f} | SWA={swa:.4f} | CVA={cva:.4f}"
    )

    ed = experiment_data[tag]["SPR_BENCH"]
    ed["predictions"], ed["ground_truth"] = preds, trues
    ed["metrics"]["test"] = {"cwa": cwa, "swa": swa, "cva": cva}

    del model
    torch.cuda.empty_cache()

# -------------- save ----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all experiment data to", os.path.join(working_dir, "experiment_data.npy"))

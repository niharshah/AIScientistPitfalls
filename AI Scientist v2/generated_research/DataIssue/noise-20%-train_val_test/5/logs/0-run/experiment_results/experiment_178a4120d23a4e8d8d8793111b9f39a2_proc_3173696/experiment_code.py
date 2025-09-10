# UseCLS token ablation vs. mean-pooling – single-file script
import os, pathlib, random, numpy as np, torch, math, time, json
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ------------------ EXPERIMENT DATA STRUCTURE ---------------------- #
experiment_data = {
    "mean_pooling": {"SPR_BENCH": {"results": {}}},
    "cls_token": {"SPR_BENCH": {"results": {}}},
}

# ----------------------- MISC SETUP -------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------- DATA UTILS ------------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_vocab(dataset: DatasetDict, seq_field: str = "sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    vocab["<cls>"] = idx  # reserve slot for CLS token (not used in sequences)
    return vocab


def encode_sequence(seq: str, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()]
    return toks[:max_len] if max_len else toks


# --------------------- SYNTHETIC PARITY DATA ----------------------- #
def build_synthetic(n_train=500, n_dev=100, n_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def _gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            lbl = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(lbl)
        return Dataset.from_dict(data)

    return DatasetDict(train=_gen(n_train), dev=_gen(n_dev), test=_gen(n_test))


# --------------------------- MODEL --------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_layers,
        num_classes,
        pad_idx,
        use_cls: bool = False,
    ):
        super().__init__()
        self.use_cls = use_cls
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        bs, seq_len = x.size()
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        if self.use_cls:
            cls_tok = self.cls_token.expand(bs, -1, -1)  # (B,1,D)
            h = torch.cat([cls_tok, h], dim=1)
            zeros = torch.zeros(bs, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([zeros, mask], dim=1)  # cls never masked
            pos = torch.arange(0, h.size(1), device=x.device).unsqueeze(0)
            h[:, 1:] += self.pos_embed(pos[:, 1:])  # add pos to non-cls part
        h = self.encoder(h, src_key_padding_mask=mask)
        if self.use_cls:
            pooled = h[:, 0, :]
        else:
            mask_flt = (~mask).unsqueeze(-1)  # (B,L,1)  True for valid tokens
            h_sum = (h * mask_flt).sum(1)
            lengths = mask_flt.sum(1).clamp(min=1)
            pooled = h_sum / lengths
        return self.classifier(pooled)


# ----------------------- DATALOADER -------------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    pad_id = vocab["<pad>"]
    padded = [s + [pad_id] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_id  # True where padding
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, loader, crit):
    model.eval()
    loss_tot, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = crit(logits, batch["labels"])
            loss_tot += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
    return loss_tot / count, correct / count


# ----------------------- LOAD DATASET ------------------------------ #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, using synthetic parity data:", e)
    datasets_dict = build_synthetic()

vocab = build_vocab(datasets_dict["train"])
num_classes = len(set(datasets_dict["train"]["label"]))
print(f"Vocab size: {len(vocab)}, num_classes: {num_classes}")

batch_size = 64
train_dl = DataLoader(
    datasets_dict["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    datasets_dict["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    datasets_dict["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)

# ------------------- HYPER-PARAMETER SWEEP ------------------------- #
nhead_values = [2, 4, 8, 16]
epochs, embed_dim = 5, 128

for variant in ["mean_pooling", "cls_token"]:
    print(f"\n##### Variant: {variant} #####")
    use_cls_flag = variant == "cls_token"
    for nhead in nhead_values:
        if embed_dim % nhead:
            print(f"Skip nhead={nhead} (embed_dim not divisible).")
            continue
        print(f"\n=== Training ({variant}) with nhead={nhead} ===")
        model = SimpleTransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=2,
            num_classes=num_classes,
            pad_idx=vocab["<pad>"],
            use_cls=use_cls_flag,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        metrics = {"train_acc": [], "val_acc": []}
        losses = {"train_loss": [], "val_loss": []}

        for ep in range(1, epochs + 1):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            for batch in train_dl:
                batch = {
                    k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                }
                optimizer.zero_grad()
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * batch["labels"].size(0)
                preds = logits.argmax(-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
            train_loss, train_acc = ep_loss / total, correct / total
            val_loss, val_acc = evaluate(model, dev_dl, criterion)
            print(
                f"Epoch {ep}/{epochs} | nhead={nhead} | train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f}"
            )
            metrics["train_acc"].append(train_acc)
            metrics["val_acc"].append(val_acc)
            losses["train_loss"].append(train_loss)
            losses["val_loss"].append(val_loss)

        # -------------------- TEST EVAL ------------------------------ #
        test_loss, test_acc = evaluate(model, test_dl, criterion)
        print(f"Finished nhead={nhead} | Test accuracy: {test_acc:.4f}")

        # gather predictions
        preds_all, gts_all = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_dl:
                gts_all.extend(batch["labels"].tolist())
                batch_gpu = {
                    k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                }
                logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
                preds_all.extend(logits.argmax(-1).cpu().tolist())

        # store results
        experiment_data[variant]["SPR_BENCH"]["results"][str(nhead)] = {
            "metrics": metrics,
            "losses": losses,
            "test_acc": test_acc,
            "predictions": preds_all,
            "ground_truth": gts_all,
        }

# -------------------- SAVE EXPERIMENT DATA ------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))

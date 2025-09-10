import os, pathlib, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# --------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- gpu / cpu ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------- experiment data container ----------
experiment_data = {
    "run": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# --------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s if s != "dev" else "dev"] = _load(f"{s}.csv")
    return d


# --------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.pad_id, self.cls_id, self.max_len = (
            vocab,
            vocab["<pad>"],
            vocab["<cls>"],
            max_len,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.cls_id] + [
            self.vocab.get(ch, self.vocab["<unk>"]) for ch in seq[: self.max_len - 1]
        ]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# --------- relative positional encoding ----------
def build_rel_pos(max_len, d_model, device):
    positions = torch.arange(-max_len + 1, max_len, device=device).float()
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, d_model, 2, device=device).float() / d_model)
    )
    sinusoid = torch.einsum("i,j->ij", positions, inv_freq)  # (2L-1, d/2)
    emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # (2L-1, d)
    return emb


# --------- transformer with rel pos ----------
class RelPosTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1, self.dropout, self.linear2 = (
            nn.Linear(d_model, dim_ff),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src, rel_pos_emb):
        # src: (B,L,D)  rel_pos_emb: (L,L,D)
        q = k = v = src
        attn_output, _ = self.self_attn(
            q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        return self.norm2(src)


class SPRModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        max_len,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.cls_id = 1  # as defined below
        self.max_len, self.d_model = max_len, d_model
        self.layers = nn.ModuleList(
            [
                RelPosTransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        # build / cache relative pos emb
        if (
            not hasattr(self, "rel_cache")
            or self.rel_cache.size(0) != 2 * self.max_len - 1
        ):
            self.rel_cache = build_rel_pos(self.max_len, self.d_model, x.device)
        idxs = torch.arange(L, device=x.device)
        rel_matrix = self.rel_cache[
            (idxs.unsqueeze(0) - idxs.unsqueeze(1)) + self.max_len - 1
        ]  # (L,L,D)
        for layer in self.layers:
            x = layer(x, rel_matrix)
        cls_vec = x[:, 0, :]  # use [CLS]
        return self.head(self.pool(cls_vec))


# --------- focal loss ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, targets):
        logp = nn.functional.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        focal = (1 - p) ** self.gamma
        loss = nn.functional.nll_loss(focal * logp, targets, weight=self.weight)
        return loss


# --------- train / eval ----------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train:
            optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        y_true.extend(batch["labels"].cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        f1_score(y_true, y_pred, average="macro"),
        y_pred,
        y_true,
    )


# --------- main pipeline ----------
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    # vocab with special tokens
    chars = set("".join(spr["train"]["sequence"]))
    vocab = {"<pad>": 0, "<cls>": 1, "<unk>": 2}
    vocab.update({ch: i + 3 for i, ch in enumerate(sorted(chars))})
    max_len = min(128, max(len(s) for s in spr["train"]["sequence"]) + 1)  # +1 for CLS

    train_ds = SPRTorchDataset(spr["train"], vocab, max_len)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, max_len)
    test_ds = SPRTorchDataset(spr["test"], vocab, max_len)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=256)
    test_dl = DataLoader(test_ds, batch_size=256)

    num_classes = len(set(spr["train"]["label"]))
    # compute class frequencies for weighted Focal loss
    cls_freq = np.bincount(spr["train"]["label"], minlength=num_classes) + 1
    cls_weight = torch.tensor(cls_freq.max() / cls_freq, dtype=torch.float32).to(device)

    model = SPRModel(len(vocab), num_classes, max_len).to(device)
    criterion = FocalLoss(gamma=2.0, weight=cls_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val, patience, wait, best_state = 0.0, 5, 0, None
    max_epochs = 20
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
        scheduler.step()

        experiment_data["run"]["epochs"].append(epoch)
        experiment_data["run"]["losses"]["train"].append(tr_loss)
        experiment_data["run"]["losses"]["val"].append(val_loss)
        experiment_data["run"]["metrics"]["train_macro_f1"].append(tr_f1)
        experiment_data["run"]["metrics"]["val_macro_f1"].append(val_f1)

        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}  val_macro_f1 = {val_f1:.4f}")

        if val_f1 > best_val:
            best_val, wait, best_state = val_f1, 0, model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stopping")
            break

    # test
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, test_dl, criterion)

    experiment_data["run"]["predictions"], experiment_data["run"]["ground_truth"] = (
        preds,
        gts,
    )
    experiment_data["run"]["test_macro_f1"], experiment_data["run"]["test_loss"] = (
        test_f1,
        test_loss,
    )
    print(f"Test Macro-F1: {test_f1:.4f}")

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# execute immediately
main()

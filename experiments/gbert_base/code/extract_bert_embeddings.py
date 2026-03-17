"""
extract_bert_embeddings.py
===========================
One-time offline script: loads all orders from the dataset,
extracts mean-pooled embeddings from deepset/gbert-base via the
transformers library (BertTokenizer + BertModel), and saves them.

deepset/gbert-base is a standard BERT model whose config.json lacks
the `model_type` key needed by the transformers AutoModel API.
We load it explicitly with BertTokenizer + BertModel to bypass this.

Run this ONCE before training:
    python gbert_base/code/extract_bert_embeddings.py

Output:
    gbert_base/data/bert_embeddings.npy   — float32 array, shape (N, 768)
    gbert_base/data/bert_order_index.json — list of original record indices
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("[ERROR] transformers is not installed.")
    print("        Run: pip install transformers torch")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = Path(r"C:\Users\Administrator\baum\data\orders_simplified_sample.json")
OUT_EMB   = ROOT / "data" / "bert_embeddings.npy"
OUT_IDX   = ROOT / "data" / "bert_order_index.json"

MODEL_ID   = "deepset/gbert-base"
BATCH_SIZE = 16
MAX_LEN    = 256   # max tokens per order; BERT limit is 512


# ── Helper: mean-pool last hidden state (ignoring padding) ────────────────────

def mean_pool(hidden_states, attention_mask):
    """
    Mean-pool token embeddings, correctly ignoring padding tokens.
    hidden_states : (batch, seq_len, hidden_dim)
    attention_mask: (batch, seq_len)
    Returns       : (batch, hidden_dim)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, S, 1)
    summed     = (hidden_states * mask_expanded).sum(dim=1)       # (B, H)
    count      = mask_expanded.sum(dim=1).clamp(min=1e-9)         # (B, 1)
    return summed / count                                          # (B, H)


# ── Feature helpers (must match model_gbert.py exactly) ──────────────────────

def preprocess_positions(positions):
    cleaned = []
    for p in positions:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter") or "unknown_cc"
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({"text": text, "totalPrice": price,
                         "totalTime": time_, "genericCostCenter": cc})
    return cleaned


def build_order_text(positions):
    parts = []
    for p in positions:
        t = p["text"].strip()
        if t and (p["totalPrice"] > 0 or p["totalTime"] > 0):
            parts.append(t.lower())
    return " ".join(parts)


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n  Loading data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

valid   = [r for r in raw
           if "input" in r and "output" in r
           and "calculatedPositions" in r["input"]]
print(f"  Valid records: {len(valid)}")

order_texts   = []
order_indices = []
for i, r in enumerate(valid):
    positions = preprocess_positions(r["input"]["calculatedPositions"])
    text      = build_order_text(positions)
    order_texts.append(text if text.strip() else "[leer]")
    order_indices.append(i)

print(f"  Order texts built: {len(order_texts)}")
print(f"  Sample text[0]  : {order_texts[0][:120]!r}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n  Loading tokenizer: {MODEL_ID}")
print("  (First run downloads ~440 MB — please wait…)\n")
tokenizer = BertTokenizer.from_pretrained(MODEL_ID)

print(f"  Loading model: {MODEL_ID}")
model = BertModel.from_pretrained(MODEL_ID)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Device: {device}")

# ── Encode in batches ──────────────────────────────────────────────────────────
print(f"\n  Encoding {len(order_texts)} orders (batch_size={BATCH_SIZE}, max_len={MAX_LEN})…")
all_embeddings = []

for start in range(0, len(order_texts), BATCH_SIZE):
    batch_texts = order_texts[start:start + BATCH_SIZE]
    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    all_embeddings.append(pooled.cpu().numpy())

    pct = min(start + BATCH_SIZE, len(order_texts))
    print(f"    {pct}/{len(order_texts)} encoded…", end="\r", flush=True)

print()
embeddings = np.vstack(all_embeddings).astype(np.float32)  # (N, 768)
print(f"\n  Embedding shape: {embeddings.shape}")

# L2-normalise for cosine compatibility
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-10)
norms_check = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms_check, 1.0, atol=1e-4), "Norm check failed"
print("  ✓ L2 norm check passed (all rows ≈ 1.0)")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(OUT_EMB, embeddings)
print(f"  ✓ Embeddings saved → {OUT_EMB}")

with open(OUT_IDX, "w", encoding="utf-8") as f:
    json.dump(order_indices, f)
print(f"  ✓ Index saved      → {OUT_IDX}")

print(f"\n  Done. You can now run: python gbert_base/code/model_gbert.py\n")

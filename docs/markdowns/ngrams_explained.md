# N-Grams — What They Are and Why We Use Them

An **n-gram** is a sequence of **N consecutive units** from a text. Depending on what the
unit is, there are two types used in this project.

---

## Type 1 — Word N-Grams

The unit is a **word**.

Example sentence:
```
"STOSSFAENGER VORNE ERSETZEN"
```

| N | Name | Result |
|---|---|---|
| 1 | Unigram | `stossfaenger`, `vorne`, `ersetzen` |
| 2 | Bigram | `stossfaenger vorne`, `vorne ersetzen` |
| 3 | Trigram | `stossfaenger vorne ersetzen` |

**Why bigrams matter:**
- `lackieren` alone could mean many things
- `stossfaenger lackieren` is much more specific → likely `paintingSpraying`
- `spotrep stossfaenger` → likely `paintingPreparation`

We use `ngram_range=(1,2)` — all unigrams **and** bigrams are included as features.

---

## Type 2 — Character N-Grams

The unit is a **single character**.

Example word:
```
"FRONTSCHEIBE"
```

| N | Result |
|---|---|
| 3 | `FRO`, `RON`, `ONT`, `NTS`, `TSC`, `SCH`, `CHE`, `HEI`, `EIB`, `IBE` |
| 4 | `FRON`, `RONT`, `ONTS`, `NTSC`, … |

The word is split into many **overlapping character sequences**.

**Why this helps:**

| Problem | Example | How char n-grams solve it |
|---|---|---|
| Abbreviations | `A+E` | Characters `A`, `+`, `E` are individually retained |
| Compound words | `Windschutzscheibe` contains `scheibe` | `CHEI`, `HEIB`, `EIBE` all match |
| Spelling variants | `Stossfänger` vs. `Stossfaenger` | Many shared subsequences |
| Unknown words | New car part name | Subpatterns known even when the word is unseen |

We use `ngram_range=(3,5)` — character sequences of length 3, 4, and 5.

---

## How Both Types Work Together in Our Model

```
Position text: "FRONTSCHEIBE ERSETZEN"
                        │
           ┌────────────┴────────────┐
           │                         │
   Word N-Grams              Character N-Grams
   ─────────────              ────────────────
   "frontscheibe"             "fron", "ront", "onts"...
   "ersetzen"                 "sche", "chei", "heib"...
   "frontscheibe ersetzen"    "tsch", "sche"...
           │                         │
           └────────────┬────────────┘
                        │
              ~3,000 TF-IDF features
                        │
                   Classifier
                        │
              glas → high probability
```

Both types are complementary:
- **Word n-grams** capture meaning and semantic context
- **Character n-grams** capture morphology — useful for abbreviations, compound nouns, and spelling variants

---

## Summary

| Type | Unit | Strength | Configuration |
|---|---|---|---|
| Word n-grams | Word | Semantic meaning, phrases | `ngram_range=(1,2)`, `max_features=2000` |
| Char n-grams | Character | Abbreviations, compounds, variants | `ngram_range=(3,5)`, `max_features=1000` |

> **Original German version:** [`markdowns/de/ngramme_erklaerung.md`](de/ngramme_erklaerung.md)

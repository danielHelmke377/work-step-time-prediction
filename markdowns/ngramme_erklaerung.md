# N-Gramme — Erklärung

Ein **N-Gramm** ist eine Sequenz von **N aufeinanderfolgenden Einheiten** aus einem Text. Je nachdem, was die Einheit ist, unterscheidet man zwei Typen:

---

## Typ 1 — Wort-N-Gramme (word n-grams)

Die Einheit ist ein **Wort**.

Beispielsatz:
```
"STOSSFAENGER VORNE ERSETZEN"
```

| N | Name | Ergebnis |
|---|---|---|
| 1 | Unigram | `stossfaenger`, `vorne`, `ersetzen` |
| 2 | Bigram | `stossfaenger vorne`, `vorne ersetzen` |
| 3 | Trigram | `stossfaenger vorne ersetzen` |

**Warum Bigramme nützlich sind:**
- `lackieren` allein kann vieles bedeuten
- `stossfaenger lackieren` ist viel spezifischer → `paintingSpraying` wahrscheinlich
- `spotrep stossfaenger` → `paintingPreparation`

Wir haben `ngram_range=(1,2)` verwendet — d. h. alle Unigramme **und** alle Bigramme werden als Features aufgenommen.

---

## Typ 2 — Zeichen-N-Gramme (character n-grams)

Die Einheit ist ein **einzelnes Zeichen**.

Beispielwort:
```
"FRONTSCHEIBE"
```

| N | Ergebnis |
|---|---|
| 3 | `FRO`, `RON`, `ONT`, `NTS`, `TSC`, `SCH`, `CHE`, `HEI`, `EIB`, `IBE` |
| 4 | `FRON`, `RONT`, `ONTS`, `NTSC`, ... |

Das Wort wird in **viele überlappende Zeichenketten** zerlegt.

**Warum das hilft:**

| Problem | Beispiel | Char N-Gram löst es |
|---|---|---|
| Abkürzungen | `A+E` | Zeichen `A`, `+`, `E` werden erkannt |
| Komposita | `Windschutzscheibe` enthält `scheibe` | `CHEI`, `HEIB`, `EIBE` matchen |
| Schreibvarianten | `Stossfänger` vs. `Stossfaenger` | Viele gemeinsame Teilfolgen |
| Unbekannte Wörter | Neues Fahrzeugteil | Teilmuster bekannt, auch wenn Wort fehlt |

Wir haben `ngram_range=(3,5)` verwendet — Zeichenfolgen der Länge 3, 4 und 5.

---

## Zusammenspiel beider Typen in unserem Modell

```
Position-Text: "FRONTSCHEIBE ERSETZEN"
                        │
           ┌────────────┴────────────┐
           │                         │
   Wort-N-Gramme              Zeichen-N-Gramme
   ─────────────              ────────────────
   "frontscheibe"             "fron", "ront", "onts"...
   "ersetzen"                 "sche", "chei", "heib"...
   "frontscheibe ersetzen"    "tsch", "sche"...
           │                         │
           └────────────┬────────────┘
                        │
              3.000 TF-IDF Features
                        │
                   Classifier
                        │
              glas → sehr wahrscheinlich
```

Beide Typen ergänzen sich:
- **Wort-N-Gramme** erkennen Bedeutung und inhaltliche Zusammenhänge
- **Zeichen-N-Gramme** erkennen Form und Struktur — auch bei Abkürzungen und Komposita

---

## Zusammenfassung

| Typ | Einheit | Stärke | Konfiguration |
|---|---|---|---|
| Word n-grams | Wort | Bedeutung, Phrasen | `ngram_range=(1,2)`, `max_features=2000` |
| Char n-grams | Zeichen | Abkürzungen, Komposita, Varianten | `ngram_range=(3,5)`, `max_features=1000` |

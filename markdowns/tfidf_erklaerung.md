# TF-IDF — Erklärung

**TF-IDF** steht für **Term Frequency – Inverse Document Frequency**. Es ist eine Methode, um Text in Zahlen umzuwandeln, die ein Modell verarbeiten kann.

---

## Das Grundproblem

Ein Computer versteht keinen Text. Wir müssen jeden Reparaturauftrag als **Zahlenvektor** darstellen. Die Frage ist: Welche Zahlen fassen den Text am besten zusammen?

---

## Bestandteil 1 — Term Frequency (TF)

*"Wie oft kommt ein Wort in diesem Dokument vor?"*

Beispiel für einen Auftrag:
```
"STOSSFAENGER A+E  STOSSFAENGER LACKIEREN  FELGE A+E"
```

| Wort | Häufigkeit (TF) |
|---|---|
| stossfaenger | 2 |
| a+e | 2 |
| lackieren | 1 |
| felge | 1 |

Häufige Wörter in einem Dokument sind tendenziell wichtig — **aber nicht immer.**

---

## Bestandteil 2 — Inverse Document Frequency (IDF)

*"Wie selten ist dieses Wort über alle Dokumente hinweg?"*

Wenn ein Wort in **jedem** Auftrag vorkommt (z. B. `fahrzeugreinigung`), sagt es über diesen spezifischen Auftrag nichts aus. Wenn ein Wort nur in **wenigen** Aufträgen vorkommt (z. B. `frontscheibe`), ist es hochinformativ.

```
IDF = log( Gesamtzahl Aufträge / Anzahl Aufträge mit diesem Wort )
```

| Wort | Aufträge mit diesem Wort | IDF |
|---|---|---|
| `a+e` | 480 von 491 | niedrig → unwichtig |
| `frontscheibe` | 38 von 491 | hoch → wichtig |
| `hagelschadenreparatur` | 15 von 491 | sehr hoch → sehr wichtig |

---

## Das Endgewicht: TF × IDF

```
TF-IDF(Wort, Auftrag) = TF × IDF
```

Ein Wort bekommt ein **hohes Gewicht**, wenn es:
- **oft in diesem Auftrag** vorkommt (hohe TF)
- **selten in anderen Aufträgen** vorkommt (hohe IDF)

Ein Wort bekommt ein **niedriges Gewicht**, wenn es:
- überall vorkommt (`a+e`, `fahrzeugreinigung`) → IDF zieht es runter
- in diesem Auftrag nur einmal vorkommt → TF ist niedrig

---

## Warum wir es hier verwenden

Jeder Auftrag wird zu einem **Vektor mit 2.000 Zahlen** (eine pro Wort/Bigram im Vokabular). Der Wert an jeder Stelle ist das TF-IDF-Gewicht dieses Terms für diesen Auftrag.

Daraus lernt der Classifier z. B.:
- Aufträge mit hohem Gewicht für `frontscheibe` → `glas`-Arbeitsschritt sehr wahrscheinlich
- Aufträge mit hohem Gewicht für `hagelschadenreparatur` → `hailrepair` sehr wahrscheinlich
- Aufträge mit hohem Gewicht für `kalibriersysteme` → `calibration` sehr wahrscheinlich

---

## Zusatz: Warum auch Zeichen-N-Gramme (char n-grams)?

Wir haben zusätzlich **character n-grams** (Zeichenketten der Länge 3–5) verwendet. Das hilft bei:

| Problem | Beispiel | Lösung durch char n-grams |
|---|---|---|
| Abkürzungen | `A+E`, `ERS.`, `V.R.` | Teilmuster werden erkannt |
| Zusammengesetzte Wörter | `stossfaenger` | Enthält `stoss`, `faeng`, `aenge` |
| Schreibvariationen | `stossfänger` vs. `stossfaenger` | Überlappende Zeichen matchen |
| Unbekannte Wörter | Neues Modell, neue Teilebezeichnung | Zeichenfolge ist trotzdem partiell bekannt |

---

## Zusammenfassung

| Komponente | Fragt | Belohnt |
|---|---|---|
| TF | Wie oft im Auftrag? | Häufige Wörter im Dokument |
| IDF | Wie selten insgesamt? | Seltene, informative Wörter |
| TF-IDF | Beides zusammen | Wörter, die für *diesen* Auftrag typisch sind |

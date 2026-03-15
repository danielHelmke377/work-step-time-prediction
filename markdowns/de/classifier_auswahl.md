# Getestete Classifier — Auswahl und Begründung

## Überblick

In Stufe 1 des Zwei-Stufen-Modells wird für jede der 14 Zielgrößen ein binärer Classifier trainiert, der entscheidet ob ein Arbeitsschritt in einem Reparaturauftrag vorkommt oder nicht. Wir haben zwei Algorithmen verglichen.

---

## 1. Logistic Regression

**Was es ist:** Ein lineares Modell, das die Wahrscheinlichkeit berechnet, dass ein Arbeitsschritt vorkommt — basierend auf einer gewichteten Summe der Features.

**Warum getestet:**
- **Stark mit TF-IDF:** Lineare Modelle und TF-IDF sind eine klassisch bewährte Kombination für Textklassifikation
- **Interpretierbar:** Die Gewichte der Features zeigen direkt, welche Wörter das Modell „überzeugen" — z. B. hohes Gewicht für `frontscheibe` bei `glas`
- **Robust bei wenig Daten:** Mit nur 294 Trainingsaufträgen ist ein komplexes Modell anfällig für Overfitting; Logistic Regression ist sehr stabil
- **Schnell:** Kein Tuning nötig, trainiert in Millisekunden

**Konfiguration:**
- `C = 1.0` (Standard-Regularisierung)
- Class weights automatisch gesetzt anhand des Klassenungleichgewichts pro Ziel

---

## 2. LightGBM Classifier

**Was es ist:** Ein Gradient-Boosting-Modell auf Entscheidungsbäumen — lernt nicht-lineare Zusammenhänge und Interaktionen zwischen Features.

**Warum getestet:**
- **Erkennt Kombinationen:** Z. B. „wenn Hailreparatur-Kostenstelle UND `DELLEN`-Keyword UND Gesamtzeit > 500 min → sehr wahrscheinlich `hailrepair`" — das kann Logistic Regression nicht direkt lernen
- **Tabular-Champion:** LightGBM ist auf strukturierten/tabellarischen Daten oft das stärkste Modell
- **Gut bei Ungleichgewichten:** `scale_pos_weight` skaliert den Trainingsverlust für seltene positive Klassen hoch
- **Schnell trotz Komplexität:** Deutlich schneller als Random Forest oder XGBoost

**Konfiguration:**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 31`
- `scale_pos_weight` automatisch aus Klassenungleichgewicht berechnet

---

## Warum nicht mehr Modelle?

| Modell | Warum nicht getestet |
|---|---|
| Random Forest | LightGBM ist in der Regel schneller und besser |
| SVM | Bei 3.065 Features und Sparse-Matrizen langsam; Logistic Regression ist äquivalent und schneller |
| Neural Networks | 294 Trainingsaufträge sind zu wenig — würde overfitten |
| Naive Bayes | Für reine Textklassifikation geeignet, aber kein sinnvoller Umgang mit numerischen Features |

---

## Ergebnis: Wer hat gewonnen?

**LightGBM in 11 von 14 Fällen** — vor allem bei seltenen oder numerisch-dominierten Zielen:

| Ziel | Gewinner | Valdierungs-F1 |
|---|---|---|
| `calibration` | LightGBM | 0.790 |
| `wheelmeasurement` | LightGBM | 0.837 |
| `bodymeasurement` | LightGBM | 0.800 |
| `bodyrepair` | LightGBM | 0.875 |
| `assembly` | LightGBM | 0.965 |
| `plasticrepair` | LightGBM | 0.851 |
| `paintingPreparation` | LightGBM | 0.986 |
| `paintingFinish` | LightGBM | 0.986 |
| `hailrepair` | LightGBM | 1.000 |
| `glas` | LightGBM | 0.923 |
| `allTiresService` | LightGBM | 0.000 ⚠️ |
| `dismounting` | Logistic Regression | 0.978 |
| `cleaning` | Logistic Regression | 1.000 |
| `paintingSpraying` | Logistic Regression | 0.986 |

> ⚠️ `allTiresService`: Nur 13 positive Beispiele im gesamten Datensatz — zu selten für zuverlässiges maschinelles Lernen.

---

## Interpretation

Das ist ein typisches Muster:

- **Klare Textsignale** (`FAHRZEUGREINIGUNG`, `LACKIERUNG`) → **Logistic Regression reicht**, ist stabiler und weniger anfällig für Overfitting
- **Komplexe, seltene oder numerisch getriebene Ziele** (`calibration`, `hailrepair`, `glas`) → **LightGBM punktet**, weil es nichtlineare Kombinationen aus Text- und Zahlfeatures lernen kann

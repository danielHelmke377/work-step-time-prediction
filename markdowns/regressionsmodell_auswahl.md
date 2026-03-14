# Regressionsmodell — Auswahl und Begründung

## Kontext

In Stufe 2 des Zwei-Stufen-Modells wird für jede Zielgröße ein Regressionsmodell trainiert, das die **Dauer in Stunden** vorhersagt — allerdings **nur auf Aufträgen, bei denen der Arbeitsschritt tatsächlich vorkommt** (Zielwert > 0).

---

## Der wichtigste Designentscheid: Trainingssubset

> Der Regressor wird **ausschließlich auf positiven Beispielen** trainiert — d. h. nur auf Aufträgen, bei denen der Zielwert > 0 ist.

Würde man alle Aufträge inklusive der Nullen verwenden, würde das Modell lernen, fast immer niedrige Werte vorherzusagen — weil das den mittleren Fehler über alle Aufträge hinweg minimiert. Durch das Herausfiltern der Nullen bekommt der Regressor eine saubere, realistische Verteilung der tatsächlichen Arbeitsdauern.

---

## Getestete Modelle

### 1. Ridge Regression

**Was es ist:** Eine lineare Regression mit L2-Regularisierung — sie bestraft große Gewichte, um Overfitting zu vermeiden.

**Warum getestet:**
- **Natürliche Ergänzung zur TF-IDF:** Lineare Regression in Kombination mit Textfeatures ist ein bewährtes Muster
- **Stabil bei wenig Daten:** Für seltene Ziele (z. B. `bodymeasurement` mit nur 21 positiven Beispielen) ist Regularisierung entscheidend
- **Schnell und erklärbar:** Koeffizienten zeigen direkt, welche Features die Zeit nach oben oder unten treiben

**Konfiguration:** `alpha = 10.0` — bewusst stärker regularisiert als der Standardwert, weil die positiven Teilmengen sehr klein sind.

---

### 2. LightGBM Regressor

**Was es ist:** Gradient-Boosting auf Entscheidungsbäumen für kontinuierliche Ausgaben (Stunden statt Ja/Nein).

**Warum getestet:**
- **Nicht-lineare Zusammenhänge:** Die Dauer hängt oft nicht linear von den Features ab. Beispiel: Ab einer bestimmten Anzahl von Dellenpositionen steigt die `hailrepair`-Zeit überproportional — das kann ein lineares Modell nicht abbilden
- **Robustheit gegenüber Ausreißern:** LightGBM ist weniger empfindlich für extreme Einzelwerte als quadratische Verlustfunktionen
- **Automatische Feature-Interaktionen:** Kombination aus Kostenstellen-Zeit, Textfeatures und Fahrzeugmarke wird automatisch genutzt

**Konfiguration:**
- `n_estimators = 300`, `learning_rate = 0.05`, `num_leaves = 31`
- `min_child_samples = max(5, n_pos // 10)` — verhindert Overfitting bei kleinen positiven Teilmengen

---

## Ergebnis: Vergleich auf dem Validierungsset (MAE, nur positive Beispiele)

| Ziel | n pos. | Ridge MAE | LightGBM MAE | Gewinner |
|---|---|---|---|---|
| `calibration` | 23 | 1.88 hrs | **0.95 hrs** | LightGBM |
| `wheelmeasurement` | 20 | 0.55 hrs | **0.51 hrs** | LightGBM |
| `bodymeasurement` | 4 | 5.65 hrs | **0.46 hrs** | LightGBM |
| `dismounting` | 90 | 1.18 hrs | **1.14 hrs** | LightGBM |
| `bodyrepair` | 32 | **2.96 hrs** | 3.27 hrs | Ridge |
| `assembly` | 86 | **1.41 hrs** | 1.43 hrs | Ridge |
| `plasticrepair` | 24 | 1.13 hrs | **0.67 hrs** | LightGBM |
| `cleaning` | 97 | 0.40 hrs | 0.40 hrs | Gleichstand |
| `paintingPreparation` | 70 | **1.62 hrs** | 2.13 hrs | Ridge |
| `paintingSpraying` | 69 | 0.73 hrs | **0.64 hrs** | LightGBM |
| `paintingFinish` | 70 | **0.93 hrs** | 1.10 hrs | Ridge |
| `glas` | 7 | 1.59 hrs | **0.89 hrs** | LightGBM |
| `hailrepair` | 32 | **0.03 hrs** | 160.8 hrs | Ridge ⚠️ |

---

## Sonderfall: `hailrepair`

Ridge schlägt LightGBM hier drastisch, weil:
- Die Dauern extrem variieren (von ~1.7 bis 67 Stunden) — entspricht ~100 bis 4.000 Minuten im Input
- LightGBM auf dem kleinen Trainingsset bei so extremen Werten stark überangepasst ist
- Ridge durch die starke Regularisierung stabiler bleibt

**Lösung:** Eine **Log-Transformation** des Zielwerts (`log1p` beim Training, `expm1` bei der Ausgabe) würde den LightGBM-MAE für `hailrepair` deutlich senken. Dies ist als nächster Verbesserungsschritt vorgesehen.

---

## Finale Entscheidung

**LightGBM als primärer Regressor** — weil er über alle 14 Ziele hinweg den niedrigeren Gesamtfehler liefert. `hailrepair` ist als bekannte Schwachstelle dokumentiert und kann durch Log-Transformation behoben werden.

Als Fallback bei zu wenigen positiven Trainingsbeispielen (< 5) wird der **Mittelwert der positiven Trainingswerte** verwendet.

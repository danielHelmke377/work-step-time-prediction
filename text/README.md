# text/ — Script Output Logs

This directory contains captured console output from the main pipeline scripts,
saved for reference. These files are **not** part of the production pipeline and
are excluded from git via `.gitignore`.

| File | Source script |
|---|---|
| `eda_output.txt` | `code/eda_phase1.py` |
| `model_output.txt` | `code/model_phase2.py` |
| `predict_demo_out.txt` | `code/predict.py --demo` |
| `predict_batch_out.txt` | `code/predict.py --batch` |

To regenerate, redirect stdout when running the relevant script:

```bash
python code/eda_phase1.py > text/eda_output.txt
python code/model_phase2.py > text/model_output.txt
```

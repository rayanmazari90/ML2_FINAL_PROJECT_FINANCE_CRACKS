# ML2 — Cross-sectional equity ML framework

Final project: point-in-time features, triple-barrier labels, CPCV validation, meta-labeling, risk neutralization, and execution-aware backtesting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure `config/default.yaml` and optional `FRED_API_KEY` for macro series.

## Data cache

Large intermediate files under `data/cache/` are **not** tracked. Run the ingestion / feature pipeline or the main notebook to rebuild them locally.

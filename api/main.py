"""FastAPI service for fraud prediction. Reuses the saved pipeline and threshold."""
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitoring import log_prediction, read_log, summarize  # noqa: E402
from src.predict import load_artifact, predict_one  # noqa: E402

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = load_artifact()
    return _artifact


class Transaction(BaseModel):
    features: Dict[str, float]  # Time, V1..V28, Amount


PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Credit Card Fraud Detection API</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #666; margin-top: 0; }}
  .card {{ background: #f5f7fa; border-radius: 10px; padding: 20px 24px; margin: 20px 0; }}
  code {{ background: #eef1f5; padding: 2px 6px; border-radius: 4px; }}
  a.btn {{ display: inline-block; background: #0d6efd; color: white; padding: 10px 18px;
           border-radius: 6px; text-decoration: none; margin-top: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  td, th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #ddd; }}
</style>
</head>
<body>
  <h1>Credit Card Fraud Detection API</h1>
  <p class="sub">Model: {model} &nbsp;|&nbsp; Threshold: {threshold} &nbsp;|&nbsp; Status: online</p>

  <div class="card">
    <h3>Endpoints</h3>
    <table>
      <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
      <tr><td>GET</td><td><code>/health</code></td><td>Service health check</td></tr>
      <tr><td>GET</td><td><code>/model-info</code></td><td>Model name, threshold, test metrics</td></tr>
      <tr><td>POST</td><td><code>/predict</code></td><td>Predict fraud for one transaction</td></tr>
      <tr><td>GET</td><td><code>/stats</code></td><td>Prediction log summary</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>Try it</h3>
    <p>Interactive API docs (Swagger UI) are available for testing requests directly:</p>
    <a class="btn" href="/docs">Open API docs</a>
  </div>

  <p class="sub">Part of an end-to-end MLOps pipeline: data validation &rarr; EDA &rarr; preprocessing &rarr;
  class-imbalance handling &rarr; model comparison &rarr; threshold optimization &rarr; MLflow tracking &rarr;
  testing &rarr; CI &rarr; deployment.</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    a = get_artifact()
    return PAGE.format(model=a["model_name"], threshold=a["threshold"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    a = get_artifact()
    return {
        "model": a["model_name"],
        "threshold": a["threshold"],
        "trained_at": a["trained_at"],
        "test_metrics": a["test_metrics"],
        "features": a["features"],
    }


@app.post("/predict")
def predict(tx: Transaction):
    try:
        result = predict_one(get_artifact(), tx.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    log_prediction(result)
    return result


@app.get("/stats")
def stats():
    return summarize(read_log())

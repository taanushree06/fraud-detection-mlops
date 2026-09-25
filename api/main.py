"""FastAPI service for fraud prediction, with an accessible built-in web UI."""
import json
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES  # noqa: E402
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
    features: Dict[str, float]


V_FEATURES = [f for f in FEATURES if f.startswith("V")]


def build_v_inputs():
    rows = []
    for f in V_FEATURES:
        rows.append(f'''
        <div class="field">
          <label for="{f}">{f}</label>
          <input type="number" step="any" id="{f}" name="{f}" value="0" inputmode="decimal">
        </div>''')
    return "\n".join(rows)


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Card Fraud Detection</title>
<style>
  :root {{
    --bg: #0f1420; --panel: #171d2c; --panel-2: #1e2536; --border: #2a3244;
    --text: #e8ecf4; --muted: #9aa5b8; --accent: #6ea8fe; --accent-2: #8b7cf6;
    --danger: #ff6b6b; --danger-bg: #3a1a1e; --safe: #51cf82; --safe-bg: #14301f;
    --radius: 14px;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "Segoe UI", system-ui, Arial, sans-serif;
    background: radial-gradient(1200px 600px at 20% -10%, #1c2540 0%, var(--bg) 55%);
    color: var(--text); min-height: 100vh; line-height: 1.5;
  }}
  a {{ color: var(--accent); }}
  .skip-link {{
    position: absolute; left: -999px; top: 0; background: var(--accent); color: #0b1020;
    padding: 8px 14px; border-radius: 0 0 8px 0; z-index: 100;
  }}
  .skip-link:focus {{ left: 0; }}
  header {{ max-width: 980px; margin: 0 auto; padding: 40px 20px 10px; }}
  .badge {{
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
    background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
    padding: 5px 12px; border-radius: 999px; margin-right: 8px;
  }}
  .badge .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--safe); }}
  h1 {{
    margin: 14px 0 4px; font-size: 30px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .sub {{ color: var(--muted); margin: 0 0 8px; font-size: 15px; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 10px 20px 60px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 24px; margin-bottom: 22px;
  }}
  .card h2 {{ margin-top: 0; font-size: 18px; }}
  .row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .field {{ display: flex; flex-direction: column; gap: 6px; }}
  .field label {{ font-size: 12px; color: var(--muted); font-weight: 600; letter-spacing: 0.02em; }}
  input[type=number] {{
    background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
    padding: 9px 10px; border-radius: 8px; font-size: 14px; width: 100%;
  }}
  input[type=number]:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  .row.top .field {{ min-width: 200px; flex: 1; }}
  fieldset {{ border: none; padding: 0; margin: 18px 0 0; }}
  legend {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; padding: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 10px; }}
  .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
  button {{
    border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    padding: 10px 18px; border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600;
  }}
  button:hover {{ border-color: var(--accent); }}
  button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  button.primary {{
    background: linear-gradient(90deg, var(--accent), var(--accent-2)); color: #0b1020; border: none;
  }}
  button:disabled {{ opacity: 0.6; cursor: progress; }}
  #result {{
    margin-top: 20px; padding: 18px 20px; border-radius: var(--radius); display: none;
  }}
  #result.fraud {{ background: var(--danger-bg); border: 1px solid #6b2b2f; }}
  #result.legit {{ background: var(--safe-bg); border: 1px solid #2b6b46; }}
  .verdict {{ font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
  .verdict.fraud-text {{ color: var(--danger); }}
  .verdict.legit-text {{ color: var(--safe); }}
  .metrics {{ display: flex; gap: 32px; margin-top: 14px; flex-wrap: wrap; }}
  .metrics div span {{ display: block; font-size: 12px; color: var(--muted); }}
  .metrics div b {{ font-size: 20px; }}
  .meter {{ height: 10px; border-radius: 999px; background: #2a3244; margin-top: 14px; overflow: hidden; }}
  .meter-fill {{ height: 100%; border-radius: 999px; transition: width 0.4s ease; }}
  .meter-fill.fraud {{ background: var(--danger); }}
  .meter-fill.legit {{ background: var(--safe); }}
  footer {{ color: var(--muted); font-size: 13px; text-align: center; padding-bottom: 40px; }}
  .visually-hidden {{
    position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap;
  }}
</style>
</head>
<body>
  <a href="#main" class="skip-link">Skip to main content</a>

  <header>
    <span class="badge"><span class="dot" aria-hidden="true"></span>Service online</span>
    <span class="badge">Model: {model}</span>
    <span class="badge">Threshold: {threshold}</span>
    <h1>Credit Card Fraud Detection</h1>
    <p class="sub">Enter transaction features or load a real example, then predict.
       See <a href="/docs">API docs</a> or <a href="/stats">live stats</a>.</p>
  </header>

  <main id="main">
    <form class="card" id="predict-form" aria-describedby="form-help">
      <h2>Transaction details</h2>
      <p id="form-help" class="sub" style="margin-top:-6px;">
        Time and Amount are raw values. V1&ndash;V28 are anonymized PCA components from the original dataset.
      </p>

      <div class="row top">
        <div class="field">
          <label for="Time">Time (seconds since first transaction)</label>
          <input type="number" step="any" id="Time" name="Time" value="0" inputmode="decimal">
        </div>
        <div class="field">
          <label for="Amount">Amount</label>
          <input type="number" step="any" id="Amount" name="Amount" value="0" inputmode="decimal">
        </div>
      </div>

      <fieldset>
        <legend>V1&ndash;V28 (anonymized features)</legend>
        <div class="grid" id="vgrid">
          {v_inputs}
        </div>
      </fieldset>

      <div class="actions">
        <button type="button" onclick="loadExample('fraud_example')">Load fraud example</button>
        <button type="button" onclick="loadExample('legit_example')">Load legitimate example</button>
        <button type="button" onclick="resetFields()">Reset</button>
        <button type="submit" class="primary" id="predict-btn">Predict</button>
      </div>

      <div id="result" role="status" aria-live="polite">
        <div class="verdict" id="verdict-text"></div>
        <div class="meter" aria-hidden="true"><div class="meter-fill" id="meter-fill" style="width:0%"></div></div>
        <div class="metrics">
          <div><span>Fraud probability</span><b id="metric-prob">-</b></div>
          <div><span>Decision threshold</span><b id="metric-threshold">-</b></div>
        </div>
      </div>
    </form>
  </main>

  <footer>
    Part of an end-to-end MLOps pipeline &mdash; data validation, EDA, preprocessing, class-imbalance
    handling, model comparison, threshold optimization, MLflow tracking, automated tests, CI and deployment.
  </footer>

<script>
const FEATURES = {features_json};
let SAMPLES = null;

async function getSamples() {{
  if (!SAMPLES) {{
    const r = await fetch('/samples');
    SAMPLES = await r.json();
  }}
  return SAMPLES;
}}

async function loadExample(name) {{
  const s = await getSamples();
  const values = s[name];
  for (const f of FEATURES) {{
    document.getElementById(f).value = values[f];
  }}
  hideResult();
}}

function resetFields() {{
  for (const f of FEATURES) {{
    document.getElementById(f).value = 0;
  }}
  hideResult();
}}

function hideResult() {{
  document.getElementById('result').style.display = 'none';
}}

document.getElementById('predict-form').addEventListener('submit', async function (e) {{
  e.preventDefault();
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  const features = {{}};
  for (const f of FEATURES) {{
    features[f] = parseFloat(document.getElementById(f).value);
  }}

  const box = document.getElementById('result');
  const verdict = document.getElementById('verdict-text');
  const meter = document.getElementById('meter-fill');
  const probEl = document.getElementById('metric-prob');
  const threshEl = document.getElementById('metric-threshold');

  try {{
    const r = await fetch('/predict', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{features}})
    }});
    const data = await r.json();

    if (!r.ok) {{
      box.className = 'fraud';
      box.style.display = 'block';
      verdict.className = 'verdict fraud-text';
      verdict.textContent = 'Error: ' + (data.detail || 'invalid input');
      meter.style.width = '0%';
      probEl.textContent = '-';
      threshEl.textContent = '-';
    }} else {{
      const pct = (data.fraud_probability * 100).toFixed(1);
      box.className = data.is_fraud ? 'fraud' : 'legit';
      box.style.display = 'block';
      verdict.className = 'verdict ' + (data.is_fraud ? 'fraud-text' : 'legit-text');
      verdict.innerHTML = (data.is_fraud ? '&#9888; ' : '&#10003; ') + data.prediction;
      meter.className = 'meter-fill ' + (data.is_fraud ? 'fraud' : 'legit');
      meter.style.width = pct + '%';
      probEl.textContent = pct + '%';
      threshEl.textContent = data.threshold;
    }}
    box.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
  }} catch (err) {{
    box.className = 'fraud';
    box.style.display = 'block';
    verdict.className = 'verdict fraud-text';
    verdict.textContent = 'Request failed: ' + err;
  }} finally {{
    btn.disabled = false;
    btn.textContent = 'Predict';
  }}
}});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    a = get_artifact()
    return PAGE.format(
        model=a["model_name"],
        threshold=a["threshold"],
        v_inputs=build_v_inputs(),
        features_json=json.dumps(FEATURES),
    )


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


@app.get("/samples")
def samples():
    path = ROOT / "models" / "sample_inputs.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="sample_inputs.json not found")
    return json.loads(path.read_text())


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

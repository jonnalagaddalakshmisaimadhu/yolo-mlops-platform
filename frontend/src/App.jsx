import { useEffect, useRef, useState } from "react";

const PALETTE = ["#ff5a36", "#1f6feb", "#12a37f", "#b44fe0", "#e0a100"];

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const canvasRef = useRef(null);

  const loadHistory = () =>
    fetch("/api/predictions?limit=8").then((r) => r.json()).then(setHistory).catch(() => {});

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => setHealth(false));
    loadHistory();
  }, []);

  // Draw the uploaded image, then the boxes on top of it.
  useEffect(() => {
    if (!file || !canvasRef.current) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const c = canvasRef.current;
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      const lw = Math.max(2, Math.round(c.width / 300));
      const fs = Math.max(14, Math.round(c.width / 45));
      g.lineWidth = lw;
      g.font = `700 ${fs}px "Schibsted Grotesk", sans-serif`;
      const labels = [...new Set((result?.detections ?? []).map((d) => d.label))];
      (result?.detections ?? []).forEach((d) => {
        const color = PALETTE[labels.indexOf(d.label) % PALETTE.length];
        const [x1, y1, x2, y2] = d.box;
        g.strokeStyle = color;
        g.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${d.label} ${Math.round(d.confidence * 100)}%`;
        const tw = g.measureText(text).width + 10;
        g.fillStyle = color;
        g.fillRect(x1, Math.max(0, y1 - fs - 6), tw, fs + 6);
        g.fillStyle = "#fff";
        g.fillText(text, x1 + 5, Math.max(fs, y1 - 6));
      });
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, [file, result]);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
  };

  const detect = async () => {
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Detection failed.");
      setResult(json);
      loadHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <header>
        <h1>Detector</h1>
        <p className={health?.model_loaded ? "status ok" : "status"}>
          {health === null ? "Checking API…" : health === false ? "API offline"
            : health.model_loaded ? `Model ready: ${health.model}` : "No model yet. Run make train."}
        </p>
      </header>

      <section className="bench">
        <div
          className="plate"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); }}
        >
          {file ? (
            <canvas ref={canvasRef} aria-label="Image with detected objects" />
          ) : (
            <label className="drop">
              <input type="file" accept="image/*" onChange={(e) => pick(e.target.files[0])} />
              Drop an image here, or choose a file
            </label>
          )}
        </div>

        <aside>
          <div className="actions">
            <label className="btn ghost">
              Choose image
              <input type="file" accept="image/*" onChange={(e) => pick(e.target.files[0])} hidden />
            </label>
            <button className="btn" disabled={!file || busy} onClick={detect}>
              {busy ? "Detecting…" : "Detect objects"}
            </button>
          </div>
          {error && <p className="error" role="alert">{error}</p>}

          {result ? (
            <>
              <h2>{result.detections.length} found <small>in {result.latency_ms} ms</small></h2>
              <ul className="dets">
                {result.detections.map((d, i) => (
                  <li key={i}><span>{d.label}</span><b>{Math.round(d.confidence * 100)}%</b></li>
                ))}
              </ul>
            </>
          ) : (
            <p className="hint">Results appear here after you detect.</p>
          )}

          <h2>Recent</h2>
          <ul className="dets history">
            {history.length === 0 && <li className="hint">Nothing yet.</li>}
            {history.map((h) => (
              <li key={h.id}><span>{h.filename}</span><b>{h.num_detections}</b></li>
            ))}
          </ul>
        </aside>
      </section>
    </main>
  );
}

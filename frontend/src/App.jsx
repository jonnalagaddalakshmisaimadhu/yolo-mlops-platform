import { useEffect, useRef, useState } from "react";

const PALETTE = {
  hardhat: "#12a37f",      // Green for hardhat
  "no-hardhat": "#ff5a36", // Red/Orange for no-hardhat
};

export default function App() {
  const [mode, setMode] = useState("file"); // "file" | "webcam"
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null); // "image" | "video"
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const [isLiveActive, setIsLiveActive] = useState(false);

  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const offscreenCanvasRef = useRef(document.createElement("canvas"));
  const isTrackingRef = useRef(false);
  const latestDetectionsRef = useRef([]);
  const animFrameIdRef = useRef(null);

  const loadHistory = () =>
    fetch("/api/predictions?limit=8").then((r) => r.json()).then(setHistory).catch(() => {});

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => setHealth(false));
    loadHistory();
    return () => stopWebcam();
  }, []);

  const drawBoxesOnCanvas = (ctx, detections, w, h) => {
    if (!detections || detections.length === 0) return;
    const lw = Math.max(2, Math.round(w / 300));
    const fs = Math.max(14, Math.round(w / 45));
    ctx.lineWidth = lw;
    ctx.font = `700 ${fs}px "Schibsted Grotesk", sans-serif`;

    detections.forEach((d) => {
      const color = PALETTE[d.label] || "#1f6feb";
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const text = `${d.label} ${Math.round(d.confidence * 100)}%`;
      const tw = ctx.measureText(text).width + 10;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - fs - 6), tw, fs + 6);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x1 + 5, Math.max(fs, y1 - 6));
    });
  };

  // Render static image
  useEffect(() => {
    if (fileType !== "image" || !file || !canvasRef.current) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const c = canvasRef.current;
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      drawBoxesOnCanvas(ctx, result?.detections, c.width, c.height);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, [file, fileType, result]);

  const pickFile = (f) => {
    if (!f) return;
    stopWebcam();
    stopLiveTracking();
    setResult(null);
    setError("");
    setFile(f);
    latestDetectionsRef.current = [];

    if (f.type.startsWith("video/")) {
      setFileType("video");
      setMode("file");
      const url = URL.createObjectURL(f);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        videoRef.current.load();
      }
    } else {
      setFileType("image");
      setMode("file");
    }
  };

  // Video/Webcam Render loop
  const renderVideoLoop = () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (v && c && !v.paused && !v.ended && v.videoWidth) {
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(v, 0, 0, c.width, c.height);
      drawBoxesOnCanvas(ctx, latestDetectionsRef.current, c.width, c.height);
    }
    if (isTrackingRef.current) {
      animFrameIdRef.current = requestAnimationFrame(renderVideoLoop);
    }
  };

  // Detection loop for Video / Webcam frames
  const runDetectionLoop = () => {
    if (!isTrackingRef.current) return;
    const v = videoRef.current;
    if (!v || v.paused || v.ended || !v.videoWidth) {
      if (isTrackingRef.current) setTimeout(runDetectionLoop, 100);
      return;
    }

    const oc = offscreenCanvasRef.current;
    oc.width = v.videoWidth;
    oc.height = v.videoHeight;
    const oCtx = oc.getContext("2d");
    oCtx.drawImage(v, 0, 0);

    oc.toBlob(
      async (blob) => {
        if (!blob || !isTrackingRef.current) return;
        try {
          const body = new FormData();
          body.append("file", blob, "frame.jpg");
          const res = await fetch("/api/predict", { method: "POST", body });
          if (res.ok) {
            const data = await res.json();
            latestDetectionsRef.current = data.detections;
            setResult(data);
          }
        } catch (e) {
          // ignore transient frame network errors
        }
        if (isTrackingRef.current) {
          setTimeout(runDetectionLoop, 60); // ~15 FPS max inference
        }
      },
      "image/jpeg",
      0.8
    );
  };

  const startLiveTracking = () => {
    isTrackingRef.current = true;
    setIsLiveActive(true);
    animFrameIdRef.current = requestAnimationFrame(renderVideoLoop);
    runDetectionLoop();
  };

  const stopLiveTracking = () => {
    isTrackingRef.current = false;
    setIsLiveActive(false);
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
    }
  };

  const startWebcam = async () => {
    stopWebcam();
    setFile(null);
    setFileType("video");
    setMode("webcam");
    setResult(null);
    setError("");
    latestDetectionsRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (videoRef.current) {
        videoRef.current.src = "";
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          startLiveTracking();
        };
      }
    } catch (err) {
      setError("Cannot access webcam: " + err.message);
      setMode("file");
    }
  };

  const stopWebcam = () => {
    stopLiveTracking();
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  // Static image detection
  const detectStaticImage = async () => {
    if (!file) return;
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
        <div>
          <h1>PPE & Helmet Detector</h1>
          <p className="subtitle">Real-time Safety Detection for Images, Videos & Live Camera</p>
        </div>
        <p className={health?.model_loaded ? "status ok" : "status"}>
          {health === null
            ? "Checking API…"
            : health === false
            ? "API offline"
            : health.model_loaded
            ? `Model ready: ${health.model}`
            : "No model yet. Run make train."}
        </p>
      </header>

      {/* Mode Navigation */}
      <nav className="mode-nav">
        <button
          className={`tab-btn ${mode === "file" && (!file || fileType === "image") ? "active" : ""}`}
          onClick={() => {
            stopWebcam();
            setMode("file");
          }}
        >
          📷 Image Mode
        </button>
        <button
          className={`tab-btn ${mode === "file" && fileType === "video" ? "active" : ""}`}
          onClick={() => {
            stopWebcam();
            setMode("file");
          }}
        >
          🎥 Video Upload
        </button>
        <button
          className={`tab-btn ${mode === "webcam" ? "active" : ""}`}
          onClick={startWebcam}
        >
          🔴 Live Webcam
        </button>
      </nav>

      <section className="bench">
        <div
          className="plate"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            pickFile(e.dataTransfer.files[0]);
          }}
        >
          {mode === "webcam" || fileType === "video" ? (
            <div className="media-container">
              <video
                ref={videoRef}
                controls={mode !== "webcam"}
                playsInline
                muted
                onPlay={() => {
                  if (!isLiveActive) startLiveTracking();
                }}
                onPause={() => {
                  if (mode !== "webcam") stopLiveTracking();
                }}
                onEnded={stopLiveTracking}
                className="hidden-video"
              />
              <canvas ref={canvasRef} aria-label="Video with real-time detection boxes" />
            </div>
          ) : file ? (
            <canvas ref={canvasRef} aria-label="Image with detected objects" />
          ) : (
            <label className="drop">
              <input
                type="file"
                accept="image/*,video/*"
                onChange={(e) => pickFile(e.target.files[0])}
              />
              <div>
                <p style={{ fontSize: "1.2rem", fontWeight: "bold", margin: "0 0 8px" }}>
                  Drop an Image or Video here
                </p>
                <p style={{ margin: 0 }}>Supports JPG, PNG, MP4, WebM, MOV</p>
              </div>
            </label>
          )}
        </div>

        <aside>
          <div className="actions">
            <label className="btn ghost">
              Choose File
              <input
                type="file"
                accept="image/*,video/*"
                onChange={(e) => pickFile(e.target.files[0])}
                hidden
              />
            </label>

            {fileType === "image" && (
              <button className="btn" disabled={!file || busy} onClick={detectStaticImage}>
                {busy ? "Detecting…" : "Detect Objects"}
              </button>
            )}

            {fileType === "video" && mode !== "webcam" && (
              <button
                className={`btn ${isLiveActive ? "btn-stop" : ""}`}
                onClick={() => {
                  if (isLiveActive) {
                    stopLiveTracking();
                    videoRef.current?.pause();
                  } else {
                    videoRef.current?.play();
                    startLiveTracking();
                  }
                }}
              >
                {isLiveActive ? "Pause Tracking" : "Start Tracking"}
              </button>
            )}

            {mode === "webcam" && (
              <button className="btn btn-stop" onClick={stopWebcam}>
                Stop Camera
              </button>
            )}
          </div>

          {error && <p className="error" role="alert">{error}</p>}

          {/* Detections Info */}
          {result ? (
            <div className="result-card">
              <h2>
                {result.detections.length} Detected{" "}
                <small>({result.latency_ms} ms)</small>
              </h2>
              <ul className="dets">
                {result.detections.length === 0 && (
                  <li className="hint">No objects detected above threshold.</li>
                )}
                {result.detections.map((d, i) => (
                  <li key={i} className={`tag-item ${d.label}`}>
                    <span>
                      <span className="dot" style={{ backgroundColor: PALETTE[d.label] || "#999" }}></span>
                      {d.label}
                    </span>
                    <b>{Math.round(d.confidence * 100)}%</b>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="hint">
              {fileType === "video" || mode === "webcam"
                ? "Play video or turn on webcam to see real-time bounding boxes."
                : "Results appear here after detection."}
            </p>
          )}

          <h2>Recent Predictions</h2>
          <ul className="dets history">
            {history.length === 0 && <li className="hint">Nothing yet.</li>}
            {history.map((h) => (
              <li key={h.id}>
                <span>{h.filename}</span>
                <b>{h.num_detections}</b>
              </li>
            ))}
          </ul>
        </aside>
      </section>
    </main>
  );
}

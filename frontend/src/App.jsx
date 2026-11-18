import { useEffect, useRef, useState } from "react";

const PALETTE = {
  hardhat: "#12a37f",      // Green for hardhat
  "no-hardhat": "#ff5a36", // Red/Orange for no-hardhat
};

export default function App() {
  const [tab, setTab] = useState("video"); // "image" | "video" | "webcam"
  const [file, setFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const imgCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const videoRef = useRef(null);
  const offscreenCanvasRef = useRef(document.createElement("canvas"));
  const busyInferRef = useRef(false);
  const isTrackingRef = useRef(false);

  const loadHistory = () =>
    fetch("/api/predictions?limit=8")
      .then((r) => r.json())
      .then(setHistory)
      .catch(() => {});

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(false));
    loadHistory();
    return () => {
      stopWebcam();
      if (videoSrc) URL.revokeObjectURL(videoSrc);
    };
  }, []);

  const drawBoxes = (ctx, detections, w, h) => {
    if (!detections || detections.length === 0) return;
    const lw = Math.max(2, Math.round(w / 350));
    const fs = Math.max(13, Math.round(w / 50));
    ctx.lineWidth = lw;
    ctx.font = `bold ${fs}px "Schibsted Grotesk", system-ui, sans-serif`;

    detections.forEach((d) => {
      const color = PALETTE[d.label] || "#1f6feb";
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const text = `${d.label} ${Math.round(d.confidence * 100)}%`;
      const tw = ctx.measureText(text).width + 10;
      const tagH = fs + 6;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - tagH), tw, tagH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 + 5, Math.max(fs, y1 - 4));
    });
  };

  // Render static image on its canvas
  useEffect(() => {
    if (tab !== "image" || !file || !imgCanvasRef.current) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const c = imgCanvasRef.current;
      if (!c) return;
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      drawBoxes(ctx, result?.detections, c.width, c.height);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, [file, tab, result]);

  // Update overlay canvas for video / webcam
  const updateVideoOverlay = (detections) => {
    const v = videoRef.current;
    const c = overlayCanvasRef.current;
    if (!v || !c || !v.videoWidth) return;
    if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
      c.width = v.videoWidth;
      c.height = v.videoHeight;
    }
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    drawBoxes(ctx, detections, c.width, c.height);
  };

  // Capture current video frame and run model inference
  const captureAndPredictFrame = (v) => {
    if (!v || !v.videoWidth || busyInferRef.current) return;
    busyInferRef.current = true;

    const oc = offscreenCanvasRef.current;
    oc.width = v.videoWidth;
    oc.height = v.videoHeight;
    const oCtx = oc.getContext("2d");
    oCtx.drawImage(v, 0, 0);

    oc.toBlob(
      async (blob) => {
        if (!blob) {
          busyInferRef.current = false;
          return;
        }
        try {
          const body = new FormData();
          body.append("file", blob, "frame.jpg");
          const res = await fetch("/api/predict", { method: "POST", body });
          if (res.ok) {
            const data = await res.json();
            setResult(data);
            updateVideoOverlay(data.detections);
          }
        } catch (e) {
          // ignore transient frame network errors
        } finally {
          busyInferRef.current = false;
        }
      },
      "image/jpeg",
      0.8
    );
  };

  // Video playback detection loop (~12 FPS)
  const runDetectionLoop = () => {
    if (!isTrackingRef.current) return;
    const v = videoRef.current;
    if (v && !v.paused && !v.ended && v.videoWidth) {
      captureAndPredictFrame(v);
    }
    if (isTrackingRef.current) {
      setTimeout(runDetectionLoop, 80);
    }
  };

  const startLiveTracking = () => {
    isTrackingRef.current = true;
    setIsPlaying(true);
    runDetectionLoop();
  };

  const stopLiveTracking = () => {
    isTrackingRef.current = false;
    setIsPlaying(false);
  };

  // File selection
  const handleFile = (f) => {
    if (!f) return;
    setError("");
    setResult(null);

    if (f.type.startsWith("video/")) {
      setTab("video");
      if (videoSrc) URL.revokeObjectURL(videoSrc);
      const url = URL.createObjectURL(f);
      setVideoSrc(url);
      setFile(f);
      stopWebcam();
    } else {
      setTab("image");
      setFile(f);
      if (videoSrc) {
        URL.revokeObjectURL(videoSrc);
        setVideoSrc(null);
      }
      stopWebcam();
    }
  };

  // Webcam handling
  const startWebcam = async () => {
    stopWebcam();
    if (videoSrc) {
      URL.revokeObjectURL(videoSrc);
      setVideoSrc(null);
    }
    setFile(null);
    setResult(null);
    setError("");
    setTab("webcam");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          startLiveTracking();
        };
      }
    } catch (err) {
      setError("Cannot access webcam: " + err.message);
      setTab("image");
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
          className={`tab-btn ${tab === "image" ? "active" : ""}`}
          onClick={() => {
            stopWebcam();
            setTab("image");
          }}
        >
          📷 Image Mode
        </button>
        <button
          className={`tab-btn ${tab === "video" ? "active" : ""}`}
          onClick={() => {
            stopWebcam();
            setTab("video");
          }}
        >
          🎥 Video Upload
        </button>
        <button
          className={`tab-btn ${tab === "webcam" ? "active" : ""}`}
          onClick={startWebcam}
        >
          🔴 Live Webcam
        </button>
      </nav>

      <section className="bench">
        {/* Main Display Plate */}
        <div
          className="plate"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            handleFile(e.dataTransfer.files[0]);
          }}
        >
          {tab === "video" && videoSrc ? (
            <div className="video-viewport">
              <video
                ref={videoRef}
                src={videoSrc}
                controls
                playsInline
                muted
                className="active-video"
                onLoadedData={(e) => {
                  const v = e.target;
                  if (overlayCanvasRef.current) {
                    overlayCanvasRef.current.width = v.videoWidth;
                    overlayCanvasRef.current.height = v.videoHeight;
                  }
                  captureAndPredictFrame(v);
                }}
                onPlay={() => {
                  startLiveTracking();
                }}
                onPause={() => {
                  stopLiveTracking();
                }}
                onSeeked={(e) => {
                  captureAndPredictFrame(e.target);
                }}
                onEnded={stopLiveTracking}
              />
              <canvas ref={overlayCanvasRef} className="overlay-canvas" />
            </div>
          ) : tab === "webcam" ? (
            <div className="video-viewport">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="active-video"
                onLoadedMetadata={(e) => {
                  const v = e.target;
                  if (overlayCanvasRef.current) {
                    overlayCanvasRef.current.width = v.videoWidth;
                    overlayCanvasRef.current.height = v.videoHeight;
                  }
                  startLiveTracking();
                }}
              />
              <canvas ref={overlayCanvasRef} className="overlay-canvas" />
            </div>
          ) : tab === "image" && file ? (
            <canvas ref={imgCanvasRef} className="static-canvas" aria-label="Image with detected objects" />
          ) : (
            <label className="drop">
              <input
                type="file"
                accept={tab === "video" ? "video/*" : "image/*"}
                onChange={(e) => handleFile(e.target.files[0])}
              />
              <div className="drop-icon">
                {tab === "video" ? "🎥" : "📷"}
              </div>
              <p style={{ fontSize: "1.2rem", fontWeight: "bold", margin: "0 0 6px" }}>
                {tab === "video"
                  ? "Drop a Video here or Click to Browse"
                  : "Drop an Image here or Click to Browse"}
              </p>
              <p style={{ margin: 0 }}>
                {tab === "video"
                  ? "Supports MP4, WebM, MOV"
                  : "Supports JPG, PNG, WebP"}
              </p>
            </label>
          )}
        </div>

        {/* Sidebar Controls & History */}
        <aside>
          <div className="actions">
            <label className="btn ghost">
              Choose File
              <input
                type="file"
                accept={tab === "video" ? "video/*" : "image/*"}
                onChange={(e) => handleFile(e.target.files[0])}
                hidden
              />
            </label>

            {tab === "image" && (
              <button className="btn" disabled={!file || busy} onClick={detectStaticImage}>
                {busy ? "Detecting…" : "Detect Objects"}
              </button>
            )}

            {tab === "video" && videoSrc && (
              <button
                className={`btn ${isPlaying ? "btn-stop" : ""}`}
                onClick={() => {
                  const v = videoRef.current;
                  if (!v) return;
                  if (v.paused) {
                    v.play();
                    startLiveTracking();
                  } else {
                    v.pause();
                    stopLiveTracking();
                  }
                }}
              >
                {isPlaying ? "Pause Tracking" : "Start Tracking"}
              </button>
            )}

            {tab === "webcam" && (
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
                  <li className="hint">No PPE/helmets detected in frame.</li>
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
              {tab === "video" || tab === "webcam"
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

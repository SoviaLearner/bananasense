"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Camera,
  CheckCircle2,
  ChevronRight,
  Database,
  ImagePlus,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Target,
  Upload,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const defaultLabels = ["mentah", "setengah-matang", "matang", "terlalu-matang"];

type MetricsResponse = {
  labels: string[];
  metrics: Record<string, number>;
  history: Record<string, number[]>;
  train_counts: Record<string, number>;
  test_counts: Record<string, number>;
  model_ready: boolean;
};

type Prediction = {
  label: string;
  probability: number;
};

function labelName(value: string) {
  return value.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function MetricCard({ icon, label, value, accent }: { icon: React.ReactNode; label: string; value: string; accent: string }) {
  return (
    <div className="metric-card">
      <div className="metric-icon" style={{ color: accent, background: `${accent}16` }}>{icon}</div>
      <div><span className="metric-label">{label}</span><strong className="metric-value">{value}</strong></div>
    </div>
  );
}

function LineChart({ values, secondary, color }: { values: number[]; secondary?: number[]; color: string }) {
  const allValues = [...values, ...(secondary || [])];
  if (!allValues.length) return <div className="empty-chart">Train model untuk melihat grafik.</div>;
  const max = Math.max(...allValues, 1);
  const min = Math.min(...allValues, 0);
  const range = max - min || 1;
  const makePath = (series: number[]) => series.map((value, index) => {
    const x = series.length === 1 ? 300 : (index / (series.length - 1)) * 580 + 10;
    const y = 160 - ((value - min) / range) * 140;
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg className="line-chart" viewBox="0 0 600 180" role="img" aria-label="Training chart">
      <path d="M10 20H590 M10 90H590 M10 160H590" className="chart-grid" />
      <path d={makePath(values)} fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" />
      {secondary && <path d={makePath(secondary)} fill="none" stroke="#ffd21f" strokeWidth="3" strokeDasharray="7 7" strokeLinecap="round" />}
    </svg>
  );
}

export default function Home() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [probabilities, setProbabilities] = useState<Prediction[]>([]);
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [cameraOpen, setCameraOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/metrics`).then((response) => response.json()).then(setMetrics).catch(() => setError("Backend belum aktif. Jalankan FastAPI di port 8000."));
    return () => streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const selectFile = (selectedFile: File | null) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0] || null);

  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOpen(true);
      setError("");
    } catch {
      setError("Kamera tidak dapat dibuka. Izinkan akses kamera atau gunakan upload gambar.");
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) selectFile(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", .92);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setCameraOpen(false);
  };

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/api/predict`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediksi gagal.");
      setResult(payload.prediction);
      setProbabilities(payload.probabilities);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediksi gagal.");
    } finally {
      setLoading(false);
    }
  };

  const labels = metrics?.labels || defaultLabels;
  const accuracy = metrics?.metrics?.test_accuracy;
  const modelReady = metrics?.model_ready;
  const history = metrics?.history || {};

  return (
    <main className="page-shell">
      <nav className="topbar">
        <div className="brand"><div className="brand-mark">🍌</div><span>BananaSense</span></div>
        <div className="system-status"><span className="online-dot" /> AI SYSTEM ONLINE</div>
      </nav>

      <section className="hero-card">
        <div className="hero-copy"><span className="eyebrow">SMART RIPENESS SCANNER</span><h1>Know your banana.<br /><em>Make it delicious.</em></h1><p>Scan a banana image and let our CNN model identify its ripeness in seconds.</p></div>
        <div className="hero-banana">🍌</div>
        <div className="circuit circuit-one" /><div className="circuit circuit-two" />
      </section>

      <section className="section-block"><div className="section-heading"><div><span className="eyebrow">OVERVIEW</span><h2>Model snapshot</h2></div><span className="live-pill"><Activity size={14} /> Live data</span></div>
        <div className="metric-grid">
          <MetricCard icon={<Database size={19} />} label="Training images" value={metrics ? String(Object.values(metrics.train_counts).reduce((sum, value) => sum + value, 0)) : "—"} accent="#b6e33b" />
          <MetricCard icon={<Target size={19} />} label="Test accuracy" value={accuracy == null ? "—" : `${(accuracy * 100).toFixed(1)}%`} accent="#ffd21f" />
          <MetricCard icon={<BrainCircuit size={19} />} label="Best epoch" value={metrics?.metrics?.best_epoch ? String(metrics.metrics.best_epoch) : "—"} accent="#b6e33b" />
          <MetricCard icon={<ShieldCheck size={19} />} label="Model status" value={!modelReady ? "Offline" : accuracy && accuracy >= .85 ? "Good" : "Tune"} accent="#ffd21f" />
        </div>
      </section>

      <section className="section-block"><div className="section-heading"><div><span className="eyebrow">LIVE SCANNER</span><h2>Capture your banana</h2></div><span className="muted">Upload a photo or use your device camera.</span></div>
        <div className="scanner-grid">
          <div className="surface-card scan-card">
            <div className="input-tabs"><button className={mode === "upload" ? "active" : ""} onClick={() => { setMode("upload"); setCameraOpen(false); }}><Upload size={16} /> Upload image</button><button className={mode === "camera" ? "active" : ""} onClick={() => setMode("camera")}><Camera size={16} /> Use camera</button></div>
            {mode === "upload" ? <label className="dropzone"><ImagePlus size={32} /><strong>Drop your image here</strong><span>JPG, PNG or WEBP · max 10MB</span><input type="file" accept="image/*" onChange={onFileChange} /></label> : <div className="camera-box">{cameraOpen ? <><video ref={videoRef} autoPlay playsInline muted /><button className="capture-button" onClick={capturePhoto}><Camera size={18} /> Capture photo</button></> : <button className="camera-start" onClick={openCamera}><Camera size={25} /><strong>Open camera</strong><span>Allow camera access to scan</span></button>}<canvas ref={canvasRef} hidden /></div>}
            {preview && <div className="preview-wrap"><img src={preview} alt="Selected banana" /><span>Image ready to scan</span></div>}
            <button className="primary-button" disabled={!file || loading} onClick={predict}>{loading ? "Analyzing..." : <><ScanLine size={18} /> Scan ripeness <ChevronRight size={18} /></>}</button>
            {error && <p className="error-message">{error}</p>}
          </div>
          <div className="surface-card result-card">
            <div className="result-heading"><div><span className="eyebrow">SCAN RESULT</span><h3>Your AI-powered insight</h3></div><Sparkles color="#ffd21f" size={22} /></div>
            {result ? <><div className="result-highlight"><span>Predicted ripeness</span><strong>{labelName(result.label)}</strong><div className="confidence-row"><span>Confidence</span><b>{(result.probability * 100).toFixed(1)}%</b></div></div><div className="probability-list">{probabilities.map((item) => <div key={item.label} className="probability-item"><div><span>{labelName(item.label)}</span><b>{(item.probability * 100).toFixed(1)}%</b></div><div className="progress-track"><div style={{ width: `${item.probability * 100}%` }} /></div></div>)}</div></> : <div className="empty-result"><div className="empty-icon"><ScanLine size={30} /></div><strong>Ready when you are.</strong><span>Upload an image or open the camera to start.</span></div>}
          </div>
        </div>
      </section>

      <section className="section-block"><div className="section-heading"><div><span className="eyebrow">MODEL INSIGHTS</span><h2>Training intelligence</h2></div><span className="muted">Best epoch: {metrics?.metrics?.best_epoch || "—"}</span></div>
        <div className="insight-grid"><div className="surface-card chart-card"><div className="card-heading"><div><h3>Learning curve</h3><span>Loss across training epochs</span></div><span className="chart-legend"><i /> Train <i className="yellow" /> Validation</span></div><LineChart values={history.loss || []} secondary={history.val_loss} color="#b6e33b" /></div><div className="surface-card chart-card"><div className="card-heading"><div><h3>Gradient descent</h3><span>Gradient norm by epoch</span></div><Activity size={18} color="#b6e33b" /></div><div className="formula">θₜ₊₁ = θₜ − η × ∇J(θₜ)</div><LineChart values={history.gradient_norm || []} color="#ffd21f" /></div></div>
      </section>

      <footer><div className="brand"><div className="brand-mark small">🍌</div><span>BananaSense</span></div><span>Computer vision for better bananas.</span></footer>
    </main>
  );
}

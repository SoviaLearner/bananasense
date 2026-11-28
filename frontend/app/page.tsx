"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import { ArrowRight, Camera, Check, ImagePlus, Moon, ScanLine, Sun, Upload, X } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8017";

type Prediction = { label: string; probability: number };

function labelName(value: string) {
  return value.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [probabilities, setProbabilities] = useState<Prediction[]>([]);
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [cameraOpen, setCameraOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(true);
  const [splash, setSplash] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setSplash(false), 10000);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

  const selectFile = (selectedFile: File | null) => {
    if (!selectedFile) return;
    if (!selectedFile.type.match(/^image\/(png|jpe?g)$/)) {
      setError("Gunakan gambar PNG, JPG, atau JPEG.");
      return;
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Ukuran gambar maksimal 10 MB.");
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setProbabilities([]);
    setError("");
  };

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
    }, "image/jpeg", 0.92);
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

  const clearSelection = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setProbabilities([]);
    setError("");
  };

  if (splash) {
    return <main className={`reference-splash ${dark ? "" : "reference-light"}`}>
      <div className="reference-splash-orb" />
      <div className="splash-banana-frame"><img src="/banana-hero.png" alt="Ilustrasi pisang" /></div>
      <span className="splash-kicker">SELAMAT DATANG</span>
      <h1>BananaSense</h1>
      <p>Cek kematangan pisang dengan mudah</p>
      <button className="reference-yellow-button" onClick={() => setSplash(false)}>Mulai pemeriksaan <ArrowRight size={18} /></button>
      <small>Masuk otomatis dalam 10 detik</small>
      <div className="splash-progress" />
    </main>;
  }

  return <main className={`reference-user ${dark ? "" : "reference-light"}`}>
    <nav className="reference-nav">
      <a href="#beranda" className="reference-brand"><span className="reference-logo"><img src="/banana-hero.png" alt="" /></span><strong>BananaSense</strong></a>
      <div className="reference-links"><a href="#beranda">Beranda</a><a href="#pemeriksaan">Pemeriksaan</a><a href="#cara-pakai">Cara Pakai</a><a href="/admin">Admin</a><button className="reference-theme" onClick={() => setDark(!dark)}>{dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? "Terang" : "Gelap"}</button></div>
    </nav>

    <section id="beranda" className="reference-hero">
      <div className="reference-hero-copy"><span className="reference-kicker">BANANASENSE</span><h1>Periksa <em>kematangan</em><br />pisang dengan<br />mudah.</h1><p>Pilih foto atau gunakan kamera. BananaSense membantu mengenali apakah pisang masih mentah, setengah matang, matang, atau terlalu matang.</p><div className="reference-hero-actions"><a href="#pemeriksaan" className="reference-yellow-button">Mulai cek sekarang <ArrowRight size={17} /></a><a href="#cara-pakai" className="reference-outline-button">Cara penggunaan</a></div><div className="reference-trust"><Check size={16} /> <span><strong>Mudah digunakan</strong><small>Tiga langkah: pilih foto, periksa, lalu lihat hasil.</small></span></div></div>
      <div className="reference-hero-image"><img src="/banana-hero.png" alt="Pisang untuk pemeriksaan kematangan" /></div>
    </section>

    <section id="cara-pakai" className="reference-steps"><div><span className="step-number">01</span><strong>Empat tingkat kematangan</strong><small>Dari mentah sampai terlalu matang</small></div><div><span className="step-number">02</span><strong>Foto atau kamera</strong><small>Gunakan cara yang paling mudah</small></div><div><span className="step-number">03</span><strong>Hasil mudah dibaca</strong><small>Hasil ditampilkan dengan jelas</small></div></section>

    <section id="pemeriksaan" className="reference-scan-section"><div className="reference-section-title"><span className="reference-kicker">PEMERIKSAAN</span><h2>Periksa kematangan pisang</h2><p>Pilih foto pisang dari perangkat atau ambil foto dengan kamera.</p></div><div className="reference-scan-grid">
      <div className="reference-panel reference-input-panel"><div className="reference-tabs"><button className={mode === "upload" ? "active" : ""} onClick={() => { setMode("upload"); setCameraOpen(false); }}><Upload size={16} /> Unggah gambar</button><button className={mode === "camera" ? "active" : ""} onClick={() => setMode("camera")}><Camera size={16} /> Gunakan kamera</button></div>{mode === "upload" ? <label className="reference-dropzone"><ImagePlus size={30} /><strong>Pilih gambar pisang</strong><span>Format PNG, JPG, atau JPEG</span><small>Ukuran maksimal 10 MB</small><input type="file" accept="image/png,image/jpg,image/jpeg" onChange={(event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0] || null)} /></label> : <div className="reference-camera">{cameraOpen ? <><video ref={videoRef} autoPlay playsInline muted /><button className="reference-yellow-button" onClick={capturePhoto}><Camera size={17} /> Ambil foto</button></> : <button className="reference-camera-start" onClick={openCamera}><Camera size={28} /><strong>Buka kamera</strong><span>Izinkan akses kamera</span></button>}<canvas ref={canvasRef} hidden /></div>}{preview && <div className="reference-preview"><img src={preview} alt="Pratinjau pisang" /><span>Gambar siap diperiksa</span><button onClick={clearSelection} aria-label="Hapus gambar"><X size={16} /></button></div>}<button className="reference-submit" disabled={!file || loading} onClick={predict}>{loading ? "Sedang memeriksa..." : <><ScanLine size={18} /> Periksa kematangan</>}</button>{error && <p className="reference-error">{error}</p>}</div>
      <div className="reference-panel reference-result-panel"><div className="reference-result-title"><div><span className="reference-kicker">HASIL</span><h3>Hasil pemeriksaan</h3></div><ScanLine size={21} /></div>{result ? <><div className="reference-result-main"><span>Perkiraan tingkat kematangan</span><strong>{labelName(result.label)}</strong><div><span>Tingkat keyakinan</span><b>{(result.probability * 100).toFixed(1)}%</b></div></div><div className="reference-probabilities">{probabilities.map((item) => <div key={item.label}><div><span>{labelName(item.label)}</span><b>{(item.probability * 100).toFixed(1)}%</b></div><i><span style={{ width: `${item.probability * 100}%` }} /></i></div>)}</div></> : <div className="reference-empty-result"><div><ScanLine size={28} /></div><strong>Hasil akan tampil di sini</strong><span>Pilih foto pisang terlebih dahulu untuk memulai pemeriksaan.</span><small><b>1</b> Pilih foto <b>2</b> Periksa <b>3</b> Lihat hasil</small></div>}</div>
    </div></section>
    <footer className="reference-footer"><div><a href="#beranda" className="reference-brand"><span className="reference-logo"><img src="/banana-hero.png" alt="" /></span><strong>BananaSense</strong></a><p>Alat sederhana untuk membantu mengenali kematangan pisang.</p></div><div><a href="#beranda">Beranda</a><a href="#pemeriksaan">Pemeriksaan</a><a href="#cara-pakai">Cara Pakai</a></div><small>© 2025 BananaSense</small></footer>
  </main>;
}

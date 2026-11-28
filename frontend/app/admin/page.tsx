"use client";

import { useEffect, useState } from "react";
import { BarChart3, Database, Gauge, Layers3, Moon, ShieldCheck, Sun } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8017";
type Metrics = { labels: string[]; metrics: any; history: Record<string, number[]>; confusion_matrix: number[][]; train_counts: Record<string, number>; test_counts: Record<string, number>; model_ready: boolean };
const name = (value: string) => value.replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const pct = (value?: number) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;

function MiniChart({ values, color = "#b7e33c" }: { values: number[]; color?: string }) {
  if (!values?.length) return <div className="empty-chart">Belum ada data training.</div>;
  const max = Math.max(...values, 1); const min = Math.min(...values, 0); const range = max - min || 1;
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${100 - ((value - min) / range) * 88 - 6}`).join(" ");
  return <svg className="admin-chart" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M0 94H100 M0 50H100 M0 6H100" className="chart-grid" /><polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" /></svg>;
}

export default function AdminPage() {
  const [data, setData] = useState<Metrics | null>(null); const [dark, setDark] = useState(true);
  useEffect(() => { fetch(`${API_URL}/api/metrics`).then((response) => response.json()).then(setData); }, []);
  const metrics = data?.metrics || {}; const history = data?.history || {}; const mcc = metrics.mcc || {};
  return <main className={`admin-shell ${dark ? "theme-dark" : "theme-light"}`}>
    <header className="admin-topbar"><div className="brand"><div className="brand-mark">🍌</div><div><strong>BananaSense Admin</strong><small>Model evaluation center</small></div></div><div className="admin-actions"><a href="/">User app</a><button className="theme-button" onClick={() => setDark(!dark)}>{dark ? <Sun size={17} /> : <Moon size={17} />} {dark ? "Terang" : "Gelap"}</button></div></header>
    <section className="admin-hero"><div><span className="eyebrow">ADMINISTRATOR</span><h1>Evaluasi performa model</h1><p>Ringkasan dataset, hasil klasifikasi, Matthews Correlation Coefficient, dan proses pembelajaran CNN.</p></div><div className="admin-status"><span className="online-dot" /> {data?.model_ready ? "Model siap digunakan" : "Model belum tersedia"}</div></section>
    <section className="admin-grid four"><div className="admin-card"><Database size={20} /><small>Data training</small><strong>{data ? Object.values(data.train_counts).reduce((a, b) => a + b, 0) : "—"}</strong></div><div className="admin-card"><Gauge size={20} /><small>Test accuracy</small><strong>{pct(metrics.test_accuracy)}</strong></div><div className="admin-card"><ShieldCheck size={20} /><small>MCC macro</small><strong>{mcc.macro == null ? "—" : mcc.macro.toFixed(4)}</strong></div><div className="admin-card"><Layers3 size={20} /><small>Best epoch</small><strong>{metrics.best_epoch || "—"}</strong></div></section>
    <section className="admin-grid two"><div className="admin-panel"><div className="panel-title"><div><span className="eyebrow">METRIK EVALUASI</span><h2>Classification report</h2></div><BarChart3 size={20} /></div><div className="table-wrap"><table><thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>MCC</th></tr></thead><tbody>{(data?.labels || []).map((label) => { const row = metrics.classification_report?.[label] || {}; return <tr key={label}><td>{name(label)}</td><td>{pct(row.precision)}</td><td>{pct(row.recall)}</td><td>{pct(row["f1-score"])}</td><td>{mcc.per_class?.[label]?.toFixed(4) || "—"}</td></tr>; })}</tbody></table></div><div className="mcc-note"><strong>MCC macro: {mcc.macro == null ? "—" : mcc.macro.toFixed(4)}</strong><span>MCC bernilai -1 sampai 1. Nilai mendekati 1 menunjukkan prediksi dan label aktual semakin konsisten.</span></div></div><div className="admin-panel"><div className="panel-title"><div><span className="eyebrow">CONFUSION MATRIX</span><h2>Kesalahan prediksi</h2></div></div><div className="matrix">{(data?.confusion_matrix || []).map((row, rowIndex) => row.map((value, columnIndex) => <div key={`${rowIndex}-${columnIndex}`} className="matrix-cell" style={{ opacity: Math.max(.28, Math.min(1, value / Math.max(...(data?.confusion_matrix.flat() || [1])) + .2)) }}>{value}<small>{data?.labels[rowIndex]} → {data?.labels[columnIndex]}</small></div>))}</div><p className="panel-caption">Baris menunjukkan label aktual, kolom menunjukkan hasil prediksi.</p></div></section>
    <section className="admin-grid two"><div className="admin-panel"><div className="panel-title"><div><span className="eyebrow">TRAINING</span><h2>Loss per epoch</h2></div><span className="legend">Train · Validasi</span></div><MiniChart values={history.loss || []} /><MiniChart values={history.val_loss || []} color="#ffd21f" /><div className="axis-note">X: epoch · Y: loss</div></div><div className="admin-panel"><div className="panel-title"><div><span className="eyebrow">OPTIMIZATION</span><h2>Gradient descent</h2></div><span className="legend">Gradient norm</span></div><div className="formula">θ<sub>t+1</sub> = θ<sub>t</sub> − η × ∇J(θ<sub>t</sub>)</div><MiniChart values={history.gradient_norm || []} color="#ffbd35" /><div className="axis-note">X: epoch · Y: gradient norm · naik-turun per batch masih normal</div></div></section>
    <footer className="admin-footer">BananaSense · Dashboard administrasi model CNN</footer>
  </main>;
}

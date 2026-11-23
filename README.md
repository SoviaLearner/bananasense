# BananaSense

Klasifikasi tingkat kematangan pisang menggunakan CNN berbasis citra digital.

## Dataset

Dataset sudah tersedia dalam dua bagian:

```text
data/
├── train/
│   ├── mentah/
│   ├── setengah-matang/
│   ├── matang/
│   └── terlalu-matang/
└── test/
    ├── mentah/
    ├── setengah-matang/
    ├── matang/
    └── terlalu-matang/
```

## Instalasi

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training CNN

Jalankan dari root `C:\Semester5-SDT\Banana Sense`:

```powershell
python -m src.bananasense.train --epochs 20 --batch-size 32
```

Training akan menghasilkan:

- `models/bananasense.keras`: model CNN
- `models/training_history.json`: loss, accuracy, dan gradient norm setiap epoch
- `models/metrics.json`: akurasi, loss, dan classification report
- `models/confusion_matrix.json`: confusion matrix data test
- `models/class_names.json`: urutan label model

Sebanyak 20% citra dari `data/train/` otomatis digunakan sebagai validation selama training. Citra di `data/test/` hanya digunakan untuk evaluasi akhir agar hasil pengukuran tidak bocor.

## Menjalankan dashboard Next.js

Terminal 1 — backend CNN:

```powershell
uvicorn backend.api:app --reload --port 8000
```

Terminal 2 — frontend:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Buka `http://localhost:3000`. Dashboard Next.js menerima gambar dari file atau kamera langsung. Output menampilkan prediksi, confidence setiap kelas, epoch terbaik, validation accuracy, test accuracy, grafik loss/accuracy, dan visualisasi gradient descent.

## Arsitektur aplikasi

```text
Next.js frontend (port 3000)
        │ upload / kamera
        ▼
FastAPI backend (port 8000)
        │ load model lokal atau Hugging Face
        ▼
CNN BananaSense + metrics training
```

Untuk deployment, simpan model dan metadata di Hugging Face Model Hub. Dataset train/test dapat disimpan di Hugging Face Dataset Hub. Backend FastAPI otomatis mengambil artifact dari Hugging Face jika folder `models/` tidak tersedia.

Deploy backend menggunakan `render.yaml`, lalu isi `HF_REPO_ID`, `HF_TOKEN`, dan `FRONTEND_ORIGIN` di environment Render. Di Vercel, pilih folder `frontend` sebagai Root Directory dan isi `NEXT_PUBLIC_API_URL` dengan URL backend Render.

## Arti gradient descent

Model memperbarui bobot menggunakan rumus:

```text
θ baru = θ lama − learning_rate × gradient(loss)
```

`gradient_norm` dicatat setiap epoch supaya proses perubahan gradien dapat diamati di dashboard.

Flowchart lengkap tersedia di `docs/flowchart-cnn.md`.

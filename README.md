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

## Menjalankan dashboard

```powershell
streamlit run app/app.py
```

Dashboard menerima gambar dari file atau kamera langsung. Output menampilkan prediksi, confidence setiap kelas, epoch terbaik, validation accuracy, test accuracy, grafik loss/accuracy, distribusi data, confusion matrix, dan visualisasi gradient descent.

## Arti gradient descent

Model memperbarui bobot menggunakan rumus:

```text
θ baru = θ lama − learning_rate × gradient(loss)
```

`gradient_norm` dicatat setiap epoch supaya proses perubahan gradien dapat diamati di dashboard.

Flowchart lengkap tersedia di `docs/flowchart-cnn.md`.

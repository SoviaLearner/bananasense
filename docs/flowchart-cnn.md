# Flowchart Proses CNN BananaSense

## Alur utama

```mermaid
flowchart TD
    A[Dataset gambar pisang] --> B{Pembagian dataset}
    B --> C[Data train]
    B --> D[Data test]

    C --> E[Resize 224 x 224]
    E --> F[Normalisasi pixel 0 sampai 1]
    F --> G[Augmentasi opsional]
    G --> H[Input ke CNN]

    H --> I[Conv2D + ReLU]
    I --> J[MaxPooling]
    J --> K[Conv2D + ReLU]
    K --> L[MaxPooling]
    L --> M[Conv2D + ReLU]
    M --> N[Global Average Pooling]
    N --> O[Dropout]
    O --> P[Dense + Softmax]
    P --> Q[Prediksi 4 kelas]

    Q --> R[Hitung loss dan accuracy]
    R --> S[Update bobot dengan Adam]
    S --> T{Epoch selesai?}
    T -- Tidak --> H
    T -- Ya --> U[Evaluasi menggunakan data test]
    U --> V[Simpan model .keras]

    V --> W[Dashboard Streamlit]
    W --> X[User unggah foto pisang]
    X --> Y[Preprocessing foto]
    Y --> Z[Model CNN melakukan prediksi]
    Z --> AA[Tampilkan jenis kematangan dan confidence]
```

## Kelas keluaran

| Jenis | Tingkat kematangan |
|---|---|
| Semua citra pisang | Mentah, setengah matang, matang, terlalu matang |

## Ringkasan proses

1. Gambar pada `data/train/` digunakan untuk mempelajari pola warna, tekstur, dan bentuk.
2. CNN mengekstrak fitur melalui lapisan convolution dan pooling.
3. Lapisan `Dense + Softmax` memilih satu dari empat kelas.
4. Bobot model diperbarui berulang kali selama beberapa epoch.
5. `data/test/` mengukur kemampuan model pada gambar yang belum dipelajari.
6. Model tersimpan di `models/bananasense.keras` dan digunakan oleh dashboard.

> Catatan: struktur dataset saat ini mengelompokkan gambar berdasarkan tingkat kematangan. Jika jenis Cavendish dan Ambon juga ingin menjadi label terpisah, gunakan subfolder `jenis/tingkat_kematangan` di dalam `train/` dan `test/`.

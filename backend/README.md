# BananaSense API

Jalankan dari root project:

```powershell
uvicorn backend.api:app --reload --port 8017
```

## Hugging Face

Backend akan memakai file lokal di `models/` jika tersedia. Jika tidak, backend otomatis mengambil model dan metadata dari Hugging Face berdasarkan environment variable:

```text
HF_REPO_ID=username/bananasense-cnn
HF_MODEL_FILENAME=bananasense.keras
HF_TOKEN=hf_...
FRONTEND_ORIGIN=https://project-kamu.vercel.app
```

Upload artifact setelah training:

```powershell
python scripts/upload_huggingface.py `
  --repo-id username/bananasense-cnn `
  --repo-type model `
  --path models `
  --token hf_TOKEN_KAMU
```

Untuk dataset, gunakan repo terpisah bertipe `dataset` dan upload folder `data`:

```powershell
python scripts/upload_huggingface.py `
  --repo-id username/bananasense-dataset `
  --repo-type dataset `
  --path data `
  --token hf_TOKEN_KAMU
```

Endpoint yang tersedia:

- `GET /api/health`
- `GET /api/metrics`
- `POST /api/predict` dengan field multipart `file`

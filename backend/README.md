# BananaSense API

Jalankan dari root project:

```powershell
uvicorn backend.api:app --reload --port 8000
```

Endpoint yang tersedia:

- `GET /api/health`
- `GET /api/metrics`
- `POST /api/predict` dengan field multipart `file`

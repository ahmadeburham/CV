# Real Estate CV Verification System

This project provides a FastAPI endpoint that performs a basic verification flow for a user:
1. Document validation from uploaded ID image
2. Liveness check (camera-based, optional)
3. Face matching between ID image and selfie
4. OCR text extraction from the ID image

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn MAIN:app --reload
```

## Open Swagger docs

Once running, open:

- http://127.0.0.1:8000/docs

## Main endpoint

`POST /verify-user/`

Multipart form-data inputs:
- `id_image`: uploaded ID image file
- `selfie_image`: uploaded selfie image file

### Notes

- Camera liveness is only enabled when `ENABLE_CAMERA=1` is set in the environment.
- If `ENABLE_CAMERA` is not set to `1`, liveness check is bypassed so local import/smoke tests remain non-interactive.

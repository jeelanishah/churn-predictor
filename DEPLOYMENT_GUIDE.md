# Deployment Guide

Step-by-step instructions for deploying Churn Predictor Pro to Render and other cloud platforms.

---

## Prerequisites

- A [GitHub](https://github.com) account with this repository pushed to it
- A free [Render](https://render.com) account

---

## Deploy to Render (Recommended)

### Option A — Automatic deploy via `render.yaml`

1. **Fork or push** this repository to your GitHub account.

2. Log in to [render.com](https://render.com) and click **New → Blueprint**.

3. Connect your GitHub account and select the **churn-predictor** repository.

4. Render detects `render.yaml` automatically and shows a preview of the service.

5. Click **Apply** — Render will install dependencies and start the app.

6. Once the build finishes (≈ 3–5 minutes) you will receive a public URL, e.g.  
   `https://churn-predictor.onrender.com`

### Option B — Manual Web Service setup

1. Log in to [render.com](https://render.com) and click **New → Web Service**.

2. Connect your GitHub repository.

3. Fill in the service settings:

   | Field | Value |
   |-------|-------|
   | Name | `churn-predictor` |
   | Runtime | `Python` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
   | Plan | `Free` |

4. Add the following environment variables under **Environment**:

   | Key | Value |
   |-----|-------|
   | `PYTHON_VERSION` | `3.10.12` |
   | `PYTHONUNBUFFERED` | `1` |

5. Click **Create Web Service**.

---

## Enabling Automatic CI/CD Deploys

The repository includes `.github/workflows/deploy.yml`, which runs tests on every
push to `main` and optionally triggers a Render deploy hook.

### Set up the deploy hook

1. In the Render dashboard open your service, go to **Settings → Deploy Hook** and
   copy the URL.

2. In your GitHub repository go to **Settings → Secrets and variables → Actions**
   and create a new secret:

   | Name | Value |
   |------|-------|
   | `RENDER_DEPLOY_HOOK_URL` | *paste the Render hook URL* |

From this point, every push to `main` will:
1. Run the full test suite (tests must pass before deploy).
2. Trigger a new Render deploy automatically.

---

## Local Docker Setup

### Build and run with Docker Compose

```bash
docker compose up --build
```

The app is available at **http://localhost:8501**.

### Build and run manually

```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

---

## Local Python Setup (no Docker)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Running Tests

```bash
pip install -r tests/requirements.txt
pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `PYTHONDONTWRITEBYTECODE` | `1` | Do not create `.pyc` files |
| `PORT` | `8501` | Port used by Render (injected automatically) |

---

## Health Check

Once deployed, verify the service is healthy:

```
GET https://<your-app>.onrender.com/_stcore/health
```

Expected response: `200 OK`

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

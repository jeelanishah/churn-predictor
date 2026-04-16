# Deployment Guide — Churn Predictor Pro

Complete step-by-step guide to deploying the Churn Predictor Pro on **Render** (free tier).

---

## Prerequisites

- GitHub account with the repository pushed
- Render account (free at [render.com](https://render.com))

---

## Option 1: Deploy to Render (Recommended — Free)

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Production deployment setup"
git push origin main
```

### Step 2: Create Render Account

1. Go to [https://render.com](https://render.com)
2. Click **"Get Started for Free"**
3. Sign up with GitHub (easiest option)

### Step 3: Create a New Web Service

1. Click **"New +"** → **"Web Service"**
2. Select **"Build and deploy from a Git repository"**
3. Click **"Connect"** next to your `churn-predictor` repository

### Step 4: Configure the Service

Render auto-detects `render.yaml` — the following values are pre-filled:

| Setting | Value |
|---------|-------|
| **Name** | churn-predictor |
| **Environment** | Python |
| **Region** | Oregon (US West) |
| **Branch** | main |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false` |

### Step 5: Deploy

1. Scroll down and click **"Create Web Service"**
2. Watch the build logs — takes ~3–5 minutes
3. Once complete, you'll see:
   ```
   ✅ Your service is live at: https://churn-predictor-xxxx.onrender.com
   ```

### Step 6: Verify Deployment

1. Open your Render URL in a browser
2. Test all 5 pages:
   - ✅ 🏠 Home loads with metrics
   - ✅ 👤 Single Prediction form works
   - ✅ 📁 Batch Prediction CSV upload works
   - ✅ 📈 Analytics charts display
   - ✅ ⚙️ API Info shows green status

---

## Option 2: Local Docker Deployment

### Prerequisites

- Docker Desktop installed ([get it here](https://www.docker.com/products/docker-desktop))

### Steps

```bash
# Clone the repo
git clone https://github.com/jeelanishah/churn-predictor.git
cd churn-predictor

# Build and start with Docker Compose
docker compose up --build

# Or manually:
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Open **http://localhost:8501** in your browser.

### Verify Docker Container

```bash
# Check container is running
docker ps

# View logs
docker logs churn-predictor-app

# Check health
curl http://localhost:8501/_stcore/health
```

---

## Option 3: Local Python (No Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## CI/CD Pipeline

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that:

1. **On every push to `main`:**
   - Runs all tests (`pytest tests/ -v`)
   - Verifies Docker build succeeds

2. **Render auto-deploys** on successful push to `main` (when connected via `render.yaml`)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | 8501 | Port the app listens on |
| `STREAMLIT_SERVER_HEADLESS` | true | Disable browser auto-open |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | false | Disable telemetry |
| `PYTHON_VERSION` | 3.10.12 | Python version for Render |

---

## Health Check Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/_stcore/health` | GET | Streamlit built-in health check |

---

## Troubleshooting

### App won't start on Render

**Problem:** Build fails with `ModuleNotFoundError`  
**Fix:** Check `requirements.txt` has all dependencies, then click "Manual Deploy"

### Model not loading

**Problem:** `Model artifacts not found` warning  
**Fix:** Ensure `model/` directory is committed to GitHub (not in `.gitignore`)

### Docker build fails

**Problem:** `COPY model/ ./model/` fails  
**Fix:** Make sure `model/*.pkl` files are present before building

### App loads but predictions fail

**Problem:** `Missing required features` error  
**Fix:** Ensure CSV has all 19 required columns with correct names

### Render free tier sleep

**Note:** Free tier services sleep after 15 minutes of inactivity.  
First load after sleep takes ~30 seconds to wake up — this is normal.

---

## Performance Tips

- **Render free tier:** Adequate for demo & internship submission
- **Render paid tier:** For production with consistent uptime
- **Docker:** Best for local testing before cloud deployment
- **Model loading:** ~2-3 seconds on cold start, instant thereafter

---

## Support

For issues:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Review Render build logs for specific errors

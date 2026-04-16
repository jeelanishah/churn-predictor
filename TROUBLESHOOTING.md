# Troubleshooting Guide

Common issues and solutions for Churn Predictor Pro.

---

## Local Setup

### `ModuleNotFoundError: No module named 'streamlit'`

Run the dependency install step:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment make sure it is activated first.

---

### App opens but shows `Error loading model artifacts`

The `model/` directory is missing or incomplete.  Regenerate the artifacts:

```bash
python train_and_save.py
```

The script creates all required `.pkl` files in `model/`.

---

### `streamlit: command not found`

Streamlit was not installed or is not on your `PATH`.

```bash
pip install streamlit
# then retry
streamlit run app.py
```

---

### Port 8501 is already in use

Either stop the process using the port or run on a different port:

```bash
streamlit run app.py --server.port=8502
```

---

## Docker

### `docker compose up` fails with `permission denied`

On Linux, add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Then retry the command.

---

### Container exits immediately after starting

Check the logs:

```bash
docker compose logs churn-predictor
```

Common causes:
- Missing `model/` artifacts — run `python train_and_save.py` first, then rebuild.
- Port conflict — change the host port in `docker-compose.yml`.

---

### Health check fails inside container

The health check uses Python's built-in `urllib` (no `curl` required).  If you see
health check failures, increase the `start_period` in `docker-compose.yml`:

```yaml
healthcheck:
  start_period: 60s
```

---

## Render Deployment

### Build fails with `ModuleNotFoundError`

Ensure `requirements.txt` lists every dependency and is committed to the repository.

---

### App is slow or times out on the free plan

Render free instances spin down after 15 minutes of inactivity and take ~30 seconds
to restart on the next request.  This is expected behaviour on the free tier.

To avoid it, upgrade to a paid plan or set up an external uptime monitor (e.g.
[UptimeRobot](https://uptimerobot.com)) to ping the health-check URL every 10 minutes.

---

### Deploy hook not triggering

1. Confirm the secret `RENDER_DEPLOY_HOOK_URL` is set in  
   **GitHub → Settings → Secrets and variables → Actions**.
2. Verify the hook URL is correct in the Render dashboard under  
   **Service → Settings → Deploy Hook**.

---

### `RENDER_DEPLOY_HOOK_URL` secret missing warning in CI

If the secret is not set, the deploy job logs:

```
RENDER_DEPLOY_HOOK_URL secret not set — skipping automatic deploy trigger.
```

This is a soft warning; tests still run and the message explains what to do.

---

## Tests

### Tests fail with `ImportError`

Make sure the test dependencies are installed:

```bash
pip install -r tests/requirements.txt
```

---

### Coverage below 80 %

Run tests with coverage details to find uncovered lines:

```bash
pytest tests/ --cov=api --cov-report=term-missing -v
```

Add tests for the highlighted lines.

---

## Getting Help

If your issue is not listed here, open a GitHub issue with:
- The full error message
- Steps to reproduce
- Your operating system and Python version (`python --version`)

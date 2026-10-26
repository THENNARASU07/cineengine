---
title: CineEngine ML
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# CineEngine ML

Script intelligence and box-office forecasting for Indian cinema: calibrated revenue range, hit/flop risk, genres, comparable films, budget what-if and a language market lens.

Open `/app` for the website. Estimates, not guarantees. This product uses the TMDB API but is not endorsed or certified by TMDB.

## Deploy on Render

This project includes a Render Blueprint in `render.yaml` and uses the Dockerfile in this directory. The required models and movie data must be present in `artifacts/`; they are included in this project.

1. Push this project directory, including `artifacts/`, to a GitHub repository.
2. In Render, choose **New + → Blueprint**, connect the repository, and deploy the `render.yaml` blueprint.
3. Open `/app` on the Render service URL. The `/api/health` endpoint is configured as the health check.

No API keys are required to start. `TMDB_API_KEY`, `LLM_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL` can be added as Render environment variables to enable the optional poster lookup and synopsis enhancement.

The blueprint uses Render's free web-service plan. Free instances have limited memory and may sleep when idle; this ML app loads PyTorch and a sentence-transformer model, so it may exceed the available memory or restart. If that happens, the service needs a higher-memory plan or a smaller model/runtime.

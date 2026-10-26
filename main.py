import os, json, joblib, requests, numpy as np, pandas as pd
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

try:                                   # optional: read keys from a .env file next to main.py
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

A = "artifacts/"
sbert = SentenceTransformer("all-MiniLM-L6-v2")
m1 = joblib.load(A + "model1_india_final.joblib")     # {model, cols}      revenue
rb = joblib.load(A + "riskband.joblib")              # {models, cols, margin} calibrated 80% revenue range
m2 = joblib.load(A + "model2_hit_flop_gbc.pkl")       # {model, threshold} hit / flop
m3 = joblib.load(A + "model3_genre.joblib")           # {clf, classes}     genres
split = json.load(open(A + "model4_split.json"))      # India share ranges from real films
films = pd.read_csv(A + "combined.csv")               # similar-film search

if os.path.exists(A + "film_emb.npy"):
    film_emb = np.load(A + "film_emb.npy")
else:                                                  # first start only (~1 min), then cached
    film_emb = sbert.encode(films.overview.tolist(), batch_size=64)
    try:
        np.save(A + "film_emb.npy", film_emb)
    except Exception:                                  # read-only disk on some hosts: just skip the cache
        pass
film_unit = film_emb / np.linalg.norm(film_emb, axis=1, keepdims=True)

TMDB_KEY = os.getenv("TMDB_API_KEY")
LLM_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")   # OpenAI or Groq (OpenAI-compatible)
LLM_KEY, LLM_MODEL = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
FALLBACK = "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=500"

app = FastAPI(title="CineEngine ML")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ScriptInput(BaseModel):
    script_text: str = ""
    logline: str = ""
    budget_crores: float = 10.0
    language: str = "ta"          # hi, ta, te, ml, kn, other
    release_year: int = 2024
    enhance: bool = True          # AI synopsis step is on by default; falls back to the original text if the AI is offline

HAS_PU = "poster_url" in films.columns                 # posters saved ahead of time (best: no key needed at runtime)
_poster_cache = {}

def poster(title, year=None):
    k = (title, year)
    if k in _poster_cache:
        return _poster_cache[k]
    try:
        prm = {"api_key": TMDB_KEY, "query": title}
        if year: prm["primary_release_year"] = int(year)
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=prm, timeout=3).json().get("results", [])
        for x in r:
            if x.get("poster_path"):
                _poster_cache[k] = "https://image.tmdb.org/t/p/w500" + x["poster_path"]   # only successes are cached
                return _poster_cache[k]
    except Exception:
        pass
    return FALLBACK

def poster_for(i):
    if HAS_PU and pd.notna(films.poster_url[i]):
        return films.poster_url[i]
    return poster(films.title[i], films.year[i] if pd.notna(films.year[i]) else None)

LLM_HINTS = {401: "invalid API key", 403: "key not allowed for this model", 404: "model name or base URL not found", 429: "rate limit or no credit", 400: "bad request (check the model name)"}

def enhance_text(text):
    """Returns (synopsis or None, status, short safe detail). Status: ok | not_configured | failed."""
    if not (LLM_KEY and LLM_MODEL):
        return None, "not_configured", None
    try:
        r = requests.post(LLM_URL + "/chat/completions", headers={"Authorization": f"Bearer {LLM_KEY}"}, timeout=20,
                          json={"model": LLM_MODEL, "messages": [
                              {"role": "system", "content": "Rewrite the film idea as a 3-4 sentence plot synopsis in the style of a movie database summary. Keep the genre, characters and setting. Do not add real people or existing film names."},
                              {"role": "user", "content": text}]})
        if r.status_code != 200:
            return None, "failed", f"HTTP {r.status_code}: " + LLM_HINTS.get(r.status_code, "provider error")
        out = r.json()["choices"][0]["message"]["content"].strip()
        return (out or None), ("ok" if out else "failed"), (None if out else "empty answer")
    except requests.Timeout:
        return None, "failed", "timeout"
    except Exception as e:
        return None, "failed", type(e).__name__

rnd = lambda v: round(float(v), 2)

def features(cols, lang, budget, year):
    g = lang if lang in cols else "other"
    return np.hstack([[np.log1p(budget)], [float(g == c) for c in cols], [year]]).reshape(1, -1)

@app.get("/")
def home():
    return {"status": "CineEngine ML API running"}

@app.get("/api/health")
def health():                               # website status pill + a quick check that the keys reached the server
    tmdb_ok = False
    if TMDB_KEY:
        try:
            tmdb_ok = requests.get("https://api.themoviedb.org/3/configuration", params={"api_key": TMDB_KEY}, timeout=3).status_code == 200
        except Exception:
            pass
    return {"status": "ok", "films": int(len(films)), "tmdb_key_valid": tmdb_ok, "llm_configured": bool(LLM_KEY and LLM_MODEL)}

@app.get("/app")
def ui():                                   # the website, served by the same server (no CORS problems)
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(f):
        raise HTTPException(404, "index.html not found next to main.py")
    return FileResponse(f)

@app.post("/api/analyze-script")
def analyze(d: ScriptInput):
    text = (d.script_text or d.logline).strip()
    if not text:
        raise HTTPException(400, "Send script_text or logline")
    budget = max(0.5, d.budget_crores)
    enhanced, ai_status, ai_detail = enhance_text(text) if d.enhance else (None, "skipped", None)   # LLM step: text only, never numbers
    q = sbert.encode([enhanced or text])[0]

    sims = film_unit @ (q / np.linalg.norm(q))                         # similar films
    archetypes = [{"rank": r, "title": films.title[i], "match_percentage": rnd(sims[i] * 100),
                   "overview": films.overview[i][:140] + "...", "image_url": poster_for(i),
                   "budget_cr": rnd(films.budget_cr[i]), "revenue_cr": rnd(films.rev_cr[i]),
                   "year": int(films.year[i]) if pd.notna(films.year[i]) else None, "language": str(films.grp[i])}
                  for r, i in enumerate(np.argsort(sims)[::-1][:3], 1)]

    lang_used = d.language if d.language in m1["cols"] else "other"
    x1 = features(m1["cols"], d.language, budget, d.release_year)      # Model 1: revenue
    revenue = float(np.expm1(m1["model"].predict(x1)[0]))
    xr = features(rb["cols"], d.language, budget, d.release_year)      # RiskBand: calibrated 80% range
    qs = np.sort([rb["models"][q].predict(xr)[0] for q in (0.1, 0.5, 0.9)])
    band = [rnd(np.expm1(qs[0] - rb["margin"])), rnd(np.expm1(qs[2] + rb["margin"]))]

    x2 = np.hstack([[np.log1p(budget), d.release_year], q]).reshape(1, -1)   # Model 2: hit / flop
    hit_p = float(m2["model"].predict_proba(x2)[0][1])
    is_hit = hit_p >= m2["threshold"]

    gp = dict(zip(m3["classes"], m3["clf"].predict_proba(q.reshape(1, -1))[0]))   # Model 3: genres
    top = lambda names: rnd(100 * max(gp.get(n, 0) for n in names))

    lo, mid, hi = split["india_share"]; nlo, nmid, nhi = split["net_gross"]     # Model 4: India split
    india_gross = revenue * mid
    profit = revenue - budget
    status = ("HIGH COMMERCIAL VIABILITY" if is_hit and profit > 0
              else "HIGH RISK" if (not is_hit) and profit <= 0 else "MODERATE / BREAK-EVEN")

    return {
        "director_module": {
            "archetypes": archetypes, "enhanced_logline": enhanced, "ai_status": ai_status, "ai_detail": ai_detail,
            "pacing_scores": {"act1_setup": None, "act2_interval": None, "act3_climax": None,
                              "emotional_depth": rnd(100 * gp.get("Drama", 0)),
                              "commercial_appeal": rnd(hit_p * 100)},
            "emotional_valence": {"action_thrill": top(["Action", "Adventure", "War"]),
                                  "emotional_drama": top(["Drama", "Romance", "Family"]),
                                  "grit_mystery": top(["Crime", "Mystery", "Thriller", "Horror"])},
            "top_genres": {g: rnd(100 * p) for g, p in sorted(gp.items(), key=lambda x: -x[1])[:3]},
        },
        "producer_module": {
            "proposed_budget": rnd(budget), "projected_gross": rnd(revenue), "worldwide_gross": rnd(revenue), "revenue_range_80": band,
            "india_gross": rnd(india_gross), "india_net": rnd(india_gross * nmid),
            "overseas_gross": rnd(revenue - india_gross),
            "india_gross_range": [rnd(revenue * lo), rnd(revenue * hi)],
            "net_profit": rnd(profit), "roi_factor": rnd(revenue / budget),
            "hit_probability": rnd(hit_p * 100), "commercial_status": status,
            "language_used": lang_used, "language_training_films": int((films.grp == lang_used).sum()),
            "note": "Treat revenue as a range: the 80% range covered 80.2% of held-out films; about 67% of point forecasts fell within 2x of actual (repeated 5-fold CV).",
        },
    }

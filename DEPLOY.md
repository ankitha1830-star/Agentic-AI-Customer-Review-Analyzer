# Deploying "Agentic AI Customer Review Analyzer"

## Files you need (already prepared)
- `app.py` — your Streamlit app (renamed from app_correct.py)
- `requirements.txt` — all the Python packages the app imports

## Option A: Streamlit Community Cloud (free, easiest — recommended)

1. **Create a GitHub repo**
   - Go to github.com → New repository → name it e.g. `review-analyzer`.
   - Upload `app.py` and `requirements.txt` to the repo root (or `git push` them).

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io and sign in with GitHub.
   - Click **"New app"**.
   - Select your repo, branch (`main`), and main file path: `app.py`.
   - Click **Deploy**.
   - First build takes several minutes (it installs torch, transformers, sentence-transformers — these are large).

3. **You'll get a public URL** like `https://your-app.streamlit.app` — this is what you share with your professor.

4. **If the app crashes on first load** — it's almost always one of:
   - Missing package in `requirements.txt` → check the "Manage app" logs, add the missing package, redeploy.
   - Memory limit exceeded (free tier is ~1GB RAM). The Hugging Face models (`distilbert`, `emotion-english-distilroberta`, `all-MiniLM-L6-v2`) plus torch can be heavy. If it fails on memory, see the note below.

## Note on the heavy ML models
Your app loads 3 models at startup:
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment)
- `j-hartmann/emotion-english-distilroberta-base` (emotion)
- `all-MiniLM-L6-v2` (semantic search embeddings)

These download from Hugging Face on first run and are cached with `@st.cache_resource`, so it's slow only on the very first load per session — but combined they need real memory. If Streamlit Cloud's free tier (1 GB RAM) fails to load them:
- Use **Hugging Face Spaces** instead (Option B) — it gives more generous free CPU resources for exactly this kind of app.
- Or disable emotion detection / semantic search by default in the sidebar to lighten the load.

## Option B: Hugging Face Spaces (better if models cause memory issues)

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK, pick a name, set visibility.
3. Upload `app.py` and `requirements.txt` (Space expects `app.py` at root — that already matches).
4. The Space builds automatically and gives you a URL like `https://huggingface.co/spaces/yourname/review-analyzer`.

## Option C: Run locally to test before deploying
```bash
pip install -r requirements.txt
streamlit run app.py
```
Test this locally first — if it runs fine on your machine, 95% of deployment issues are just missing packages or memory limits, not code bugs.

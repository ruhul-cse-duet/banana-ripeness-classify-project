# Banana Ripeness Classification

Streamlit application that classifies banana photos into four ripeness stages (Unripe, Ripe, Overripe, Rotten) using a custom CNN checkpoint.

## Live project link below
https://banana-ripeness-classify.streamlit.app/


## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run App.py
```

Ensure that `models/resnet_Model.pth` exists before launching the app.

## Container Build & Run

1. Build the image:
   ```bash
   docker build -t banana-ripeness-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 --name banana-ripeness banana-ripeness-app
   ```
3. Open http://localhost:8501 in your browser.

If you need live code reloads during development, mount the repo as a volume:

```bash
docker run --rm -p 8501:8501 -v ${PWD}:/app banana-ripeness-app
```

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Initial banana ripeness app"
git branch -M main
git remote add origin https://github.com/ruhul-cse-duet/banana-ripeness-classify-project.git
git push -u origin main
```

After pushing, enable GitHub Actions or the repository’s container registry if you need automated builds. Update `README.md` with the image name/tag once it exists on Docker Hub or GHCR.



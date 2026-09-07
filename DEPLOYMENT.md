# Cloud Deployment Guide: Vehicle Number Plate Enhancement App

This guide outlines step-by-step instructions to deploy your Streamlit application to **free cloud hosting platforms**, specifically **Streamlit Community Cloud** (primary recommended platform) and **Hugging Face Spaces**.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended • 100% Free)

Streamlit Community Cloud is the official, zero-configuration hosting platform for Streamlit applications. It connects directly to your GitHub repository and automatically re-deploys every time you push code updates.

### Prerequisites
- Your code pushed to GitHub: [babitaincoding/Vehicle-Number-Plate-Enhancement](https://github.com/babitaincoding/Vehicle-Number-Plate-Enhancement)
- A free account on [share.streamlit.io](https://share.streamlit.io) (sign in with your GitHub account).

### Step-by-Step Deployment:
1. **Push your latest changes to GitHub:**
   Open PowerShell or Terminal in your project directory and run:
   ```bash
   git add .
   git commit -m "feat: modernize clean Streamlit UI and prepare cloud deployment"
   git push origin main
   ```

2. **Open Streamlit Community Cloud:**
   - Navigate to [https://share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account (`babitaincoding`).

3. **Create New App:**
   - Click the **"New app"** (or **"Create app"**) button in the top right.
   - Choose **"I already have an app"**.

4. **Fill in the Repository Details:**
   - **Repository:** `babitaincoding/Vehicle-Number-Plate-Enhancement`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (Custom Subdomain):** (Optional, e.g., `plate-enhancement.streamlit.app`)

5. **Advanced Settings (Optional):**
   - Python Version: Choose `3.10` or `3.11`.
   - No secret keys are required.

6. **Click "Deploy!":**
   - Streamlit Cloud will read `packages.txt` (installing `libgl1` and `libglib2.0-0` for OpenCV) and `requirements.txt`.
   - Within 2-3 minutes, your live public URL will be active!

---

## 🤗 Option 2: Hugging Face Spaces (100% Free • 16 GB RAM Tier)

Hugging Face Spaces offers a generous free tier (2 vCPUs, 16GB RAM) which provides ample memory for deep learning inference (PyTorch & EasyOCR).

### Step-by-Step Deployment:
1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces) and log in or create a free account.
2. Click **"Create new Space"**.
3. Configure the Space:
   - **Space Name:** `vehicle-plate-enhancement`
   - **License:** `mit` or `openrail`
   - **Select Space SDK:** Choose **Streamlit**.
   - **Space Hardware:** Select **CPU basic • 2 vCPU • 16GB • Free**.
4. Clone or push your repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/vehicle-plate-enhancement
   git push space main
   ```
5. Hugging Face Spaces will automatically build the environment and host your Streamlit app with an embedded public link.

---

## 🐳 Option 3: Local or Cloud Container (Docker)

If you wish to deploy to platforms like **Render**, **Railway**, **Google Cloud Run**, or test locally using Docker:

### Build and Run Locally:
```bash
# Build the Docker image
docker build -t vehicle-plate-enhancer .

# Run the container mapping port 8501
docker run -d -p 8501:8501 --name plate-app vehicle-plate-enhancer
```
Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🛠️ Included Deployment Assets in this Repository

| File | Purpose |
| :--- | :--- |
| `app.py` | Modernized, clean Streamlit application with OCR caching and visual diagnostics |
| `requirements.txt` | Clean, UTF-8 encoded production dependencies with `opencv-python-headless` |
| `packages.txt` | System Debian packages (`libgl1`, `libglib2.0-0`) for Linux cloud runtimes |
| `.streamlit/config.toml` | Production server configuration (headless mode, disabled CORS, custom slate theme) |
| `Dockerfile` | Multi-stage production container setup based on `python:3.10-slim` |
| `.dockerignore` | Excludes unnecessary local artifacts to keep image builds fast and lean |
| `data/sample_plates/` | 21 curated sample plates indexed for immediate testing out of the box |

---

## 💡 Troubleshooting & FAQs

- **Q: Why does the first OCR run take 2-3 seconds on cloud free tiers?**
  - **A:** On the initial run, EasyOCR downloads the pre-trained English CRNN weights (~8MB) into memory. Subsequent runs on the same plate are cached via `@st.cache_data` and execute instantaneously!
- **Q: Why `opencv-python-headless`?**
  - **A:** Standard `opencv-python` looks for X11/GUI libraries which are absent on headless Linux cloud containers, causing `ImportError: libGL.so.1`. Headless OpenCV coupled with `packages.txt` guarantees zero deployment crashes.

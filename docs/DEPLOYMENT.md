# 🌐 Streamlit Community Cloud Deployment Guide

This guide explains how to host the **DeepResearch Agent** on [Streamlit Community Cloud](https://share.streamlit.io/) directly from your GitHub repository, complete with username/password authentication for portfolio demonstrations.

---

## 📋 Prerequisites

1. A GitHub account with access to `deepakj111/deep-research-agent`.
2. An OpenAI API key (`OPENAI_API_KEY`).
3. (Optional) Tavily API key (`TAVILY_API_KEY`) and GitHub Personal Access Token (`GITHUB_TOKEN`).

---

## 🚀 Step-by-Step Deployment

### Step 1: Sign in to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Sign in with your GitHub account.

### Step 2: Create a New App
1. Click **Create app** (or **New app**).
2. Choose **Use existing repo**.
3. Select your repository: `deepakj111/deep-research-agent`.
4. Branch: `main`.
5. Main file path: `app/streamlit_app.py`.

---

## 🔒 Step 3: Configure Authentication & API Secrets

1. Before clicking Deploy, expand **Advanced settings**.
2. In the **Secrets** text area, paste the following configuration (replace with your desired credentials and keys):

```toml
# ── Demo Login Credentials ──
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "your_custom_password_here"

# ── API Keys ──
OPENAI_API_KEY = "sk-proj-..."
TAVILY_API_KEY = "tvly-..."
GITHUB_TOKEN = "ghp_..."
```

3. Click **Save** and then click **Deploy!**

---

## 👤 How the Demo Login Works

- Upon opening your app URL (e.g. `https://deep-research-agent.streamlit.app`), users will see a **Portfolio Demo Access** login screen.
- Only users entering the correct `DEMO_USERNAME` and `DEMO_PASSWORD` configured in Streamlit Cloud Secrets can access the app.
- Authenticated users will see a **Log Out** button in the sidebar to return to the login gateway at any time.

---

## 🛠️ Local Development & Testing

To test the login locally:
1. Set `DEMO_USERNAME` and `DEMO_PASSWORD` in your `.env` file:
   ```env
   DEMO_USERNAME=admin
   DEMO_PASSWORD=demo123
   ```
2. Or create `.streamlit/secrets.toml` based on `.streamlit/secrets.toml.example`.
3. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

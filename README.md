<div align="center">

# 📡 NeuralRAG – Production‑Level AI Workspace

A premium, self‑hosted AI assistant that combines Retrieval‑Augmented Generation (RAG) with powerful remote system control.

<br />

[![View Live Hub](https://img.shields.io/badge/🔴_VIEW_LIVE_HUB-Click_Here-red?style=for-the-badge)](https://nyc-benefit-ensure-precipitation.trycloudflare.com/)

<br /><br />

</div>

---

## ✨ What This Project Does

NeuralRAG is a **premium, self‑hosted AI assistant** that combines Retrieval‑Augmented Generation (RAG) with powerful remote system control. It runs entirely on GitHub Actions and is exposed to the world via a Cloudflare tunnel (or Ngrok), giving you a live web UI that works from any device.

### 🎯 Core Features

- **🔄 Multi‑API‑Key Auto‑Rotation** – Connects to Gemini, Groq, OpenRouter, etc. If one key hits its quota or fails, the system instantly switches to the next key without interrupting the conversation.
- **🛡️ Admin Password Protection** – The main UI is gated by an admin password. Visitors can see the UI but cannot use the AI until the password is entered.
- **🖥️ Remote System Control** – After authentication you can run commands on the host machine via a secure Cloudflare/Ngrok tunnel. Perfect for managing a server from anywhere.
- **🔎 Live Web Search** – When you ask about current events or docs, the assistant performs a real‑time web search and synthesizes the freshest information.
- **📂 Knowledge‑Base (RAG) Upload** – Drag‑and‑drop PDFs, TXT, CSV, Excel, or images directly in the sidebar. The files are indexed on‑the‑fly and used to answer your queries with zero‑shot accuracy.
- **⚙️ Seamless Deployments** – Each GitHub Actions run automatically updates the Cloudflare tunnel URL and publishes a tiny redirect page on GitHub Pages, so the link in the badge is always current.

> **Note:** The internal details (GitHub Actions configuration, secret handling, tunnel commands) are intentionally omitted from the public README to keep credentials safe.

## 📦 Quick Start (For Maintainers)

- Set the secrets `GOOGLE_API_KEYS` and `GROQ_API_KEYS` in the repository settings.
- Add an `ADMIN_PASSWORD` secret that the UI will require on first load.
- Merge to `main` → the workflow provisions a fresh Cloudflare tunnel and updates the live badge.

## 🧩 Technologies Used

- **Python 3.11** – FastAPI server handling chat, RAG, and system commands.
- **Cloudflare Tunnel** – Secure, zero‑config public URL.
- **GitHub Actions** – CI/CD, automatic URL rotation, GitHub Pages redirect.
- **Vanilla HTML/CSS/JS** – Clean, responsive UI with a persistent right‑hand sidebar.
- **🔐 Secure Secrets Management** – All API keys and passwords stored as GitHub Actions secrets.

---

### 🙏 Contributing

Feel free to open issues or PRs if you want to add new models, improve the UI, or extend system‑control capabilities. Please **do not expose any secret keys** in the repository.

---

**Enjoy exploring NeuralRAG!** 🎉

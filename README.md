# Production-Level RAG Agent 🧠

A sophisticated Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Gemini 2.5 Flash**, and **Qdrant Cloud**. This system employs a **Hybrid Search architecture** (Vector + Keyword) with **Reranking** and **Agentic decision-making**.

## 🚀 Key Features

* **🧠 Hybrid Search:** Combines Dense Vectors (Semantic meaning) with Sparse Vectors (BM25/Keyword matching).
* **⚖️ Reranking:** Retrieves 50 candidates and uses **FlashRank** to select the top 5 most accurate results.
* **🤖 Agentic Logic:** Intelligently switches between Document Search and Web Search (DuckDuckGo).
* **👁️ Multimodal Vision:** Processes images and charts using Gemini Vision.
* **⚡ Streaming & Memory:** Real-time typewriter output with session history.

## 🔑 Setup & API Keys

To run this project, you need to get free API keys from the following providers:

1.  **Google Gemini API Key** (The LLM Brain):
    * [Get API Key from Google AI Studio](https://aistudio.google.com/app/apikey)

2.  **Qdrant Cloud API Key & URL** (The Vector Database):
    * [Sign up for Qdrant Cloud](https://cloud.qdrant.io/)
    * *Create a Cluster (Free Tier) -> Data Access Control -> Create API Key.*

## 📦 How to Run

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Keys:** Open the `.env` file and paste your API keys inside the quotes.
4.  Initialize the database (Run once):
    ```bash
    python init_db.py
    ```
5.  Run the app:
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
* **LLM:** Google Gemini 2.5 Flash
* **Vector DB:** Qdrant Cloud (Hybrid Collection)
* **Orchestration:** LangChain
* **Reranker:** FlashRank
* **Frontend:** Streamlit
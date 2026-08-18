# Tech News Agent 🤖📰

![Tech News Agent Architecture](ai-agent-news-reader.jpg)
*Figure 1: Workflow pipeline for the Tech News AI Agent (Image generated via Gemini).*

An autonomous local AI researcher built with **CrewAI** and **Ollama**. This script searches the web for recent developments across specified technology topics, scrapes detailed article contents, and automatically saves a structured Markdown report to your system.

---

## 📌 Features

* **100% Local Inference:** Runs entirely locally via Ollama (`llama3.1`) to ensure privacy and avoid external API usage costs.
* **Custom Search Capabilities:** Wrapped with a CrewAI-compatible DuckDuckGo Search Tool to discover recent web links and articles.
* **Web Scraping:** Utilizes `ScrapeWebsiteTool` to inspect high-value web pages directly.
* **Auto-Saving Reports:** Summarizes key findings and exports them to a local file (`tech_news_summary.md`).
* **Stability Controls:** Configured with low temperature (`0.1`), restricted delegation, and iteration caps (`max_iter=5`) to keep local LLMs focused and prevent hallucinated URLs.

---

## ⚙️ Prerequisites

Before running the agent, make sure you have the following installed:

1. **Python 3.10+**
2. **Ollama** installed and running on your local machine (`http://localhost:11434`).
3. **Llama 3.1 Model** pulled into Ollama:
   ```bash
   ollama pull llama3.1

## 🏃 How to Run

### Step 1: Start Ollama

Make sure your local Ollama service is running in the background and serving requests at `http://localhost:11434`:

```bash
ollama serve
```

### Step 2: Sync Dependencies with uv
Install and sync all project dependencies into an isolated virtual environment using uv:

```bash
uv sync
```

### Step 3: Execute the Agent (uv run)
Run the script directly inside your uv environment using module execution mode:

```bash
uv run python -m ai_agents.tech_news_agent
```

### Step 4: Monitor Execution & Output
Watch real-time logs as the agent generates search queries, scrapes pages, and forms insights. Upon completion, inspect the generated report in your project root:

```bash
cat tech_news_summary.md
```

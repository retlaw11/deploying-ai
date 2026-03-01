# Assignment 2 — Walter's AI Security Assistant

Version: 1.0.0

Owner: Walter P.  <retlaw1@gmail.com> — GitHub: retlaw11

This folder contains `assignment-2` application: an AI application that combines three services under a Gradio-based conversational UI.

Project objectives
- Provide an integrated conversational interface for security workflows
- Demonstrate hybrid search (lexical + semantic) over an AI Risk dataset provided (AI_risk_database_v4.csv) and sourced from https://airisk.mit.edu
- Fetch and summarize real-time security news using Google News RSS + LLM
- Scan user-provided files with VirusTotal and interpret results for non-technical users

The above objectives meets the objectives of Assignment 2. 

Core files
- `app.py` — Main application 
- `service_1.py` — VirusTotal file scanner with LLM interpretation
- `service_2.py` — AI Risk Database hybrid semantic+lexical search. Uses file AI_risk_database_v4.csv which is an MIT AI Risk databse "https://airisk.mit.edu". 
- `service_3.py` — Security news web search with LLM synthesis
- `AI_risk_database_v4.csv` — Local dataset used by Service 2
- `.secrets` — Local environment/config file (gitignored)
- `.cache/` — Embeddings & caches

Overview of functionality
- File scanning (VirusTotal): Upload a file in the File Scanner tab; the app queries VirusTotal, fetches scan results, and then calls `interpret_scan_results_with_llm()` to produce an easy-to-read summary subject to guardrail checks. You can also request on the home page to run a file scan, the llm will send you to the separate tab.
- AI Risk Search (Service 2): A hybrid retrieval pipeline: lexical pre-filtering followed by OpenAI embeddings re-ranking. The `Top results` slider controls how many final results are returned.
- Security News (Service 3): Fetches Google News RSS results for a query, cleans article metadata, and sends the articles to OpenAI (GPT-4o-mini) for a synthesized executive summary, findings and recommendations.

Design highlights
- Hybrid search weighting: lexical × 0.35 + semantic × 0.65 (α = 0.35) to balance precision and semantic recall.
- Enterprise guardrails implemented to prevent forbidden topics, prompt injection, and rate-limit abusive inputs. Forbidden terms also baked into code base and system prompts locked out of suer access.
- Gradio chosen for rapid prototyping and simple multi-tab UX and as well being a requirement for the project.

File structure
```
assignment-2/
├── app.py                        # Main application (Gradio UI + orchestration)
├── service_1.py                  # VirusTotal scanning + helpers
├── service_2.py                  # Hybrid search implementation
├── service_3.py                  # Web search + LLM synthesis
├── AI_risk_database_v4.csv       # Local dataset used by Service 2
├── .secrets                      # Local environment (API keys) - gitignored
├── .cache/                       # Embeddings & caches
└── README.md                     # This file
```

Dependencies — local vs. install-on-build
This project uses a small set of external packages. The list below states packages used by `assignment-2` specifically and whether they are expected to be already present in your dev environment or need to be installed during setup.

Required (install during build / virtualenv):
- `gradio` (UI) — used for multi-tab interface and widgets
- `openai` (LLM client) — used for chat, embeddings, and synthesis (via API Gateway)
- `pydantic` (v2) — configuration and data validation
- `python-dotenv` — load `.secrets` environment variables
- `requests` — fetch RSS feeds and call VirusTotal
- `typing-extensions` — type helpers for backwards compatibility
 - `chromadb` — persistent embedding store used by `service_2.py` 

Optional / examples (only for extended functionality / tests):
- `langchain`, `langgraph` — used in `test.py` and agent experiments (not required to run the main app)

Standard library (no install needed):
- `os`, `sys`, `json`, `csv`, `re`, `time`, `datetime`, `hashlib`, `xml.etree.ElementTree`, `urllib.parse`, `dataclasses`, `enum`

## Getting Started: Complete Setup Guide

### Prerequisites
- **Python 3.8+** installed on your machine
- **Git** for version control
- **API accounts** (see API Keys section below)

### Step 1: Clone or navigate to the project

```bash
cd ../05_src/assignment-2
```

### Step 2: Create and activate a Python virtual environment

```bash
# Create virtual environment
python3 -m venv deploying-ai-env

# Activate it (macOS/Linux)
source deploying-ai-env/bin/activate

# Activate it (Windows)
# deploying-ai-env\Scripts\activate
```

You should see `(deploying-ai-env)` in your terminal prompt.

### Step 3: Upgrade pip and install dependencies

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required packages
pip install gradio openai pydantic python-dotenv requests typing-extensions chromadb

# Optional: Install additional packages for testing/extended features
pip install langchain langgraph fastmcp langchain-mcp-adapters
```

### Step 4: Obtain and configure API keys

#### 4a. VirusTotal API Key

1. Go to **https://www.virustotal.com/gui/home/upload**
2. Click **Sign In** (or create a free account)
3. Navigate to **Settings** → **API Key** (or go directly to https://www.virustotal.com/gui/user/[username]/apikey)
4. Copy your API key (you'll see a string starting with `ea827edd...`)

#### 4b. OpenAI API Key (via API Gateway)

The app uses an **API Gateway** to call OpenAI services. You need:
- **API_GATEWAY_KEY**: Your API gateway authentication key (provided by your organization or API provider)

If you don't have an API gateway set up:
1. Create an **OpenAI account** at https://platform.openai.com
2. Go to **API keys** and create a new secret key
3. Use this as your `API_GATEWAY_KEY` (or set up your own API gateway that forwards to OpenAI)

**Note**: This application uses:
- `gpt-4o-mini` for chat and synthesis
- `text-embedding-3-small` for semantic search

### Step 5: Create the `.secrets` file

Create a `.secrets` file in the `assignment-2/` directory with your API keys:

```bash
# Create the .secrets file
cat > .secrets << 'EOF'
VIRUSTOTAL_API_KEY=your_virustotal_api_key_here
API_GATEWAY_KEY=your_api_gateway_key_here
OPENAI_API_KEY=any_value
EOF
```

### Step 6: Run the application

```bash
# Make sure you're in the project directory
cd ../deploying-ai/05_src/assignment-2

# Make sure virtualenv is activated
source deploying-ai-env/bin/activate

# Run the app
python app.py
```

You should see output like:
```
✅ VirusTotal API configured successfully
✅ Application initialized
Running on http://127.0.0.1:7860
```

### Step 7: Access the web interface

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

You should see the Gradio interface with four tabs:
- **Chat** — General conversation with smart routing
- **File Scanner** — Upload and scan files with VirusTotal. To test scan functionality and file upload I used a dummy test file "eicar.com.txt" it is a fake virus that is used to test antivirus files. Available at: https://www.eicar.org
- **AI Risk Search** — Hybrid search over the AI Risk database
- **Security News** — Fetch and synthesize security news

### Troubleshooting

**Error: `ModuleNotFoundError: No module named 'gradio'`**
- Solution: Make sure your virtual environment is activated (`source deploying-ai-env/bin/activate`)
- Run: `pip install gradio openai pydantic python-dotenv requests typing-extensions chromadb`

**Error: `VIRUSTOTAL_API_KEY not found in .secrets file`**
- Solution: Create `.secrets` file with valid API keys (see Step 5)
- Verify the file is in the correct directory: `/Users/retlawair/Desktop/deploying-ai/05_src/assignment-2/.secrets`

**Error: `API Gateway connection failed`**
- Solution: Check that `API_GATEWAY_KEY` in `.secrets` is correct and has active API credits
- Verify your internet connection

**Error: `chromadb` issues on M1/M2 Mac**
- Solution: Install using: `pip install chromadb --upgrade --force-reinstall`

**Port 7860 already in use**
- Solution: Kill the existing process or specify a different port:
```bash
python app.py --server_name=127.0.0.1 --server_port=7861
```

### Deactivating the virtual environment

When you're done working:
```bash
deactivate
```


Service descriptions 

Service 1 — VirusTotal File Scanner (`service_1.py`)
- Purpose: Accept a file path, compute SHA256, upload or query VirusTotal, return scan metadata and aggregated engine detections.
- Main functions: `get_file_hash()`, `scan_file_with_details()`, `scan_url()`, `format_scan_results()`
- Integration: Invoked from the File Scanner tab in `app.py`. Interpretation step calls `interpret_scan_results_with_llm()` which uses `openai` to produce a user-friendly summary.

Service 2 — AI Risk Hybrid Search (`service_2.py`)
- Purpose: Provide relevant records from `AI_risk_database_v4.csv` using a hybrid approach.
- Algorithm: Lexical pre-filter (BM25-style) to select candidates, then semantic re-ranking with `text-embedding-3-small` embeddings from OpenAI. Final ranking uses α=0.35 weight for lexical.
- Main functions: `hybrid_search(query, top_k, lexical_k, alpha)`, `format_search_results_markdown()`
- Integration: Exposed via the AI Risk Search tab and callable directly from chat (smart routing).

Service 3 — Security News Web Search (`service_3.py`)
- Purpose: Fetch Google News RSS results for user query, clean and aggregate top articles, and synthesize a security-focused summary with an LLM.
- Main functions: `web_search(query, max_results)` and `search_security_news_with_llm(prompt, max_results)`
- Integration: Exposed via the Security News tab and callable directly from chat.


Security & guardrails
- Input validation, forbidden topic filters, prompt-injection regex checks, and rate limiting are implemented in `app.py` (class `ContentGuardrails`). These are active for chat and web search inputs and should remain enabled for production deployments.

Maintenance & extension notes
- To add a new service, create `service_N.py`, add import and safe call in `app.py`, then add a tab in `build_interface()` and update intent keywords in `IntentDetector`.
- Consider adding Redis for caching embeddings and Prometheus for metrics in production.

Contact & support
- Owner:  — retlaw1@gmail.com

---
Last updated: 2026-02-28
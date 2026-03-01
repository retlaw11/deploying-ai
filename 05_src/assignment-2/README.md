# Assignment 2 — Walter's AI Security Assistant

Version: 1.0.0

Owner: Walter Pareja <retlaw1@gmail.com> — GitHub: retlaw11

This folder contains `assignment-2` application: an AI application that combines three services under a Gradio-based conversational UI.

Project objectives
- Provide an integrated conversational interface for security workflows
- Demonstrate hybrid search (lexical + semantic) over an AI Risk dataset
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
- File scanning (VirusTotal): Upload a file in the File Scanner tab; the app queries VirusTotal, fetches scan results, and then calls `interpret_scan_results_with_llm()` to produce an easy-to-read summary subject to guardrail checks.
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

Install command (recommended in virtualenv)
```bash
python -m venv deploying-ai-env
source deploying-ai-env/bin/activate
pip install --upgrade pip
pip install gradio openai pydantic python-dotenv requests typing-extensions
# Optional (for tests/examples)
pip install langchain langgraph fastmcp langchain-mcp-adapters
# Install ChromaDB for persistent embeddings (recommended)
pip install chromadb
```

Configuration and secrets
- Create a `.secrets` file in this folder (gitignored). Required keys:
  - `API_GATEWAY_KEY` — API gateway key used for OpenAI requests
  - `VIRUSTOTAL_API_KEY` — VirusTotal API key for file scanning

Example `.secrets` content:
```
API_GATEWAY_KEY=sk-xxxxx
VIRUSTOTAL_API_KEY=vt-xxxxx
```

How to run
1. Activate your virtualenv (see install command above)
2. From this repository root run:
```bash
cd .. /05_src/assignment-2
python app.py
```
3. Open `http://127.0.0.1:7860` in your browser. The UI includes:
   - Chat tab (general LLM conversation + smart routing)
   - File Scanner (VirusTotal scanning + AI interpretation)
   - AI Risk Search (hybrid retrieval)
   - Security News Search (web + LLM synthesis)

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
- Owner: Walter Pareja — retlaw1@gmail.com

---
Last updated: 2026-02-28

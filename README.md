# ai-agents
This github repo will help understanding how to create AI Agent

## Using LangChain with Google Gemini (Gen AI) 🔧

Prerequisites:
- Python 3.10+ and an activated virtualenv (we recommend `.venv`)
- `langchain` and `google-genai` installed in the venv
- A Gemini/Generative AI key (or Vertex service account) available as env vars

Quick start:
1. Create a `.env` in the project root containing your key (don't commit this file):

```bash
echo 'GEMINI_API_KEY="your_key_here"' > .env
``` 

2. Run the sample LangChain examples:

```bash
source .venv/bin/activate
python3 sample_langchain.py
```

Files added:
- `langchain_google_genai.py` — a small LangChain-compatible LLM wrapper around `google-genai`.
- `sample_langchain.py` — demonstrates an `LLMChain` and a simple agent that uses a `calculator` tool.

Security:
- Rotate API keys if exposed, and store secrets in a secret manager for production. Prefer service accounts + ADC for production Vertex usage.

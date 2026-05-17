# ArXiv AI/ML Research Assistant

A LangChain-powered research assistant that performs deep ArXiv paper discovery,
summarizes findings, suggests experiments, and generates action items — all through
a clean Streamlit interface.

## Features

- Deep ArXiv search with relevance ranking (not just keyword matching)
- Interactive paper cards with PDF links and key findings
- Experiment design suggestions based on found literature
- ML metrics calculator
- Actionable next steps after every research session
- Full conversation memory within a session

## Stack

- **LangChain** — ReAct agent + tools + memory
- **Google Gemini 2.5 Flash** — language model
- **ArXiv API** — paper retrieval
- **Streamlit** — UI

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
streamlit run app/main.py
```

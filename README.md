# LangChain ArXiv Research Agent for AI/ML

An AI-powered research assistant that performs deep ArXiv paper discovery, analyzes findings, suggests experiments, and generates action items — all through a clean conversational interface built with LangChain, LangGraph, and Streamlit.

---

## ✨ What it does

You type a research topic. The agent autonomously:

1. **Searches ArXiv** for the most relevant papers using the official ArXiv API
2. **Analyzes** the top results — extracts problem statements, methodology, key contributions, and limitations
3. **Displays paper cards** on the right panel with abstracts, PDF links, and ArXiv links
4. **Designs experiments** based on the literature found
5. **Generates action items** — prioritized next steps, implementation roadmap, resources to follow
6. **Remembers context** across the entire session so follow-up questions just work

---

## 🧰 Tech Stack

| Layer           | Technology                                           |
| --------------- | ---------------------------------------------------- |
| LLM             | Google Gemini 2.5 Flash via `langchain-google-genai` |
| Agent framework | LangGraph `create_react_agent` + `MemorySaver`       |
| Tools           | Custom LangChain `@tool` functions                   |
| Paper retrieval | `arxiv` Python client (official API)                 |
| UI              | Streamlit                                            |
| Config          | `python-dotenv`                                      |

---

## 🤖 Agent & Tools

The agent follows a **ReAct** (Reasoning + Acting) loop — it reasons about what to do, picks a tool, observes the result, and continues until it has a complete answer.

### Tools

**`search_arxiv_papers`**
Searches ArXiv for papers matching a topic. Supports sorting by relevance or date, configurable result count (3–15). Returns structured metadata: title, authors, date, ArXiv ID, PDF URL, abstract, categories.

**`analyze_paper`**
Deeply analyzes a specific paper by title or ArXiv ID. Extracts: problem statement, methodology, key contributions, quantitative results, limitations, and practical relevance for ML engineers.

**`design_experiment`**
Designs a concrete, reproducible experiment based on a research topic or set of papers. Takes compute resources into account (Colab / cloud GPU / cluster). Returns hypothesis, baseline, dataset choice, training config, evaluation protocol, timeline, and success criteria.

**`calculate_ml_metrics`**
Calculates a full suite of evaluation metrics from a confusion matrix (TP, TN, FP, FN). Returns Accuracy, Precision, Recall, F1, Specificity, MCC, and approximate AUC-ROC with class imbalance detection.

**`generate_action_items`**
Produces a prioritized action plan after a research session. Includes key takeaways, immediate tasks, short-term goals, deep-dive recommendations, implementation roadmap, and open research questions.

### Memory

Two memory modes are available and switchable at runtime from the sidebar:

- **Buffer** — stores the full verbatim conversation history, best for focused sessions
- **Summary** — compresses history into a running summary via LangGraph's `MemorySaver`, scales to long sessions without hitting token limits

---

## 🚀 Setup

### Prerequisites

- Python 3.10 or higher
- A Google AI Studio API key → [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Serhii2009/langchain-arxiv-research-agent.git
cd langchain-arxiv-research-agent
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
copy .env.example .env
```

Open `.env` and replace the placeholder with your key:

```env
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

**5. Run the app**

```bash
python -m streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Use `python -m streamlit run app/main.py` rather than `streamlit run app/main.py` directly. This ensures Python resolves the `app` module path correctly from the project root.

---

## 💡 Usage

| What you want            | What to type                                            |
| ------------------------ | ------------------------------------------------------- |
| Find papers on a topic   | `Recent advances in vision transformers`                |
| Analyze a specific paper | `Analyze the paper Attention Is All You Need`           |
| Compare approaches       | `Compare LoRA vs full fine-tuning for LLMs`             |
| Design an experiment     | `Design an experiment for diffusion model distillation` |
| Calculate metrics        | `My model has TP=450, TN=8200, FP=300, FN=50`           |
| Get next steps           | `What should I do next based on what we found?`         |

Use the **Quick searches** buttons in the sidebar to get started instantly.

After any research session, use the **🧪 Design experiment** and **✅ Get action items** buttons that appear below the paper cards panel.

---

## 📁 Project Structure

```
langchain-arxiv-research-agent/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── app/
    ├── main.py              — Streamlit entry point and UI layout
    ├── agent.py             — LangGraph ReAct agent with MemorySaver
    └── tools/
    │   ├── arxiv_search.py  — ArXiv paper search tool
    │   ├── paper_analyze.py — Deep paper analysis tool
    │   ├── experiment.py    — Experiment design tool
    │   ├── metrics.py       — ML metrics calculator tool
    │   └── learning.py      — Action items generator tool
    └── ui/
        ├── paper_cards.py   — Paper card components
        └── styles.py        — Custom CSS (light theme)
```

---

## 📄 License

MIT

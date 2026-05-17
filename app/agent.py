import os
import uuid
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.tools.arxiv_search import search_arxiv_papers
from app.tools.paper_analyze import analyze_paper
from app.tools.experiment import design_experiment
from app.tools.metrics import calculate_ml_metrics
from app.tools.learning import generate_action_items

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arxiv_assistant")

SYSTEM_PROMPT = """You are an elite AI/ML Research Analyst. Your job is NOT to summarize papers — 
the UI already shows paper cards with abstracts. Your job is to produce a deep, insightful 
research briefing that a senior ML engineer would find genuinely valuable.

When a user gives you a research topic, follow this exact workflow:

STEP 1 — Search
Call search_arxiv_papers to find the most relevant recent papers (prioritize 2025-2026).

STEP 2 — Analyze
Call analyze_paper on the 2-3 most important results to extract deep technical details.

STEP 3 — Generate action items
Call generate_action_items with the topic and key insights found.

STEP 4 — Write your response using this exact structure:

---

## 🧠 What is [TOPIC]?
Explain the concept from first principles. Use intuitive analogies. Assume the reader knows 
ML basics but has not studied this specific topic. Be concrete, not vague.

## 🔍 Why does it matter?
Explain the core tension or open problem this research area is trying to solve. 
What breaks without understanding this? What becomes possible if we do understand it?

## 📡 What the 2025–2026 research frontier looks like
Based on the papers you found, describe the active research directions RIGHT NOW.
Do NOT list papers — describe the intellectual landscape:
- What are the main competing hypotheses or approaches?
- Where do researchers disagree?
- What has been recently proven or disproven?
- What surprising findings emerged?

## 🧪 Experiment you could run tomorrow
Design ONE concrete, novel experiment that:
- Has not been described in the papers you found
- Could be done on a single GPU or Google Colab
- Would produce a meaningful, publishable insight
- Includes: hypothesis, dataset, model setup, what to measure, expected outcome

## 🗺️ Research directions worth pursuing
List 3-4 specific open questions or directions that are:
- Not yet answered by existing literature
- Tractable for a small research team
- Likely to produce impactful results

## ✅ Action items
From generate_action_items output — concrete next steps, resources, repos to explore.

---

CRITICAL RULES:
- Never say 'Paper X discusses...' or 'According to paper Y...' — synthesize, don't cite
- Never repeat information already visible in paper cards (titles, abstracts, authors)
- Write as if explaining to a brilliant colleague over coffee, not as an academic report
- Be specific: name architectures, datasets, metrics, failure modes
- The experiment section must be genuinely novel — not a replication of found papers
- Minimum 600 words in your final response
- Always use the section headers exactly as shown above"""

TOOLS = [
    search_arxiv_papers,
    analyze_paper,
    design_experiment,
    calculate_ml_metrics,
    generate_action_items,
]


def build_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=4096,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _to_str(content) -> str:
    """Converts any Gemini content format to plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _parse_papers_from_tool_output(text: str) -> list:
    """Parses structured output of search_arxiv_papers into paper dicts."""
    import re
    papers = []
    blocks = re.split(r"\[\d+\]", text)

    for block in blocks[1:]:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        paper = {
            "title": lines[0],
            "authors": "",
            "published": "",
            "arxiv_id": "",
            "pdf_url": "",
            "abstract": "",
            "categories": [],
        }

        for line in lines[1:]:
            if line.startswith("Authors:"):
                paper["authors"] = line.replace("Authors:", "").strip()
            elif line.startswith("Published:"):
                parts = line.replace("Published:", "").strip().split("|")
                paper["published"] = parts[0].strip()
                if len(parts) > 1:
                    paper["arxiv_id"] = parts[1].replace("ArXiv ID:", "").strip()
            elif line.startswith("PDF:"):
                paper["pdf_url"] = line.replace("PDF:", "").strip()
            elif line.startswith("Categories:"):
                paper["categories"] = [
                    c.strip() for c in line.replace("Categories:", "").split(",")
                ]
            elif line.startswith("Abstract:"):
                paper["abstract"] = line.replace("Abstract:", "").strip()

        if paper["title"] and len(paper["title"]) > 5:
            papers.append(paper)

    return papers


def _extract_final_answer(messages: list) -> str:
    """
    Finds the last AIMessage that contains actual text content
    and does not contain tool_calls — that is the final answer.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue

        # Skip messages that are just tool invocations with no text
        tool_calls = getattr(msg, "tool_calls", None)
        content_str = _to_str(msg.content).strip()

        if tool_calls and not content_str:
            continue

        if content_str:
            return content_str

    return "The agent completed the search but did not produce a final summary. Try asking: 'Summarize what you found.'"


class AgentWrapper:
    def __init__(self, memory_type: str = "buffer"):
        self.memory_type = memory_type
        self.llm = build_llm()
        self.checkpointer = MemorySaver()
        self.graph = create_react_agent(
            model=self.llm,
            tools=TOOLS,
            checkpointer=self.checkpointer,
            prompt=SYSTEM_PROMPT,
        )
        self.thread_id = "session_001"
        logger.info(f"[AGENT] Built with memory_type={memory_type}")

    def invoke(self, payload: dict) -> dict:
        user_input = payload.get("input", "")
        config = {"configurable": {"thread_id": self.thread_id}}

        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            messages = result.get("messages", [])

            if not messages:
                return {"output": "No response generated.", "papers": []}

            # Log all message types for debugging
            for i, msg in enumerate(messages):
                logger.info(f"[AGENT] msg[{i}] type={type(msg).__name__} content_len={len(_to_str(msg.content))}")

            output = _extract_final_answer(messages)

            # Extract papers from ToolMessage outputs
            papers = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_text = _to_str(msg.content)
                    if "Found" in tool_text and "papers for" in tool_text:
                        parsed = _parse_papers_from_tool_output(tool_text)
                        papers.extend(parsed)

            logger.info(f"[AGENT] Final output: {len(output)} chars | Papers: {len(papers)}")
            return {"output": output, "papers": papers}

        except Exception as e:
            logger.error(f"[AGENT] invoke error: {e}")
            return {"output": f"⚠️ Error: {str(e)}", "papers": []}

    def clear_memory(self):
        self.thread_id = str(uuid.uuid4())
        logger.info(f"[AGENT] Memory cleared, new thread: {self.thread_id}")


def build_agent(memory_type: str = "buffer") -> AgentWrapper:
    return AgentWrapper(memory_type=memory_type)
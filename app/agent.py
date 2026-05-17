import os
import uuid
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
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

SYSTEM_PROMPT = """You are an expert AI/ML Research Assistant specialized in discovering,
analyzing, and synthesizing academic literature from ArXiv.

Your capabilities:
- search_arxiv_papers: Deep semantic search across ArXiv, returns structured paper metadata
- analyze_paper: Extract key contributions, methodology, and findings from a paper
- design_experiment: Suggest concrete experiments based on the literature found
- calculate_ml_metrics: Calculate Accuracy, Precision, Recall, F1, MCC from confusion matrix values
- generate_action_items: Produce prioritized next steps after a research session

Behavior rules:
1. Always start with search_arxiv_papers when the user asks about a research topic
2. After search, automatically call analyze_paper on the top 2-3 most relevant results
3. Always end a research session with generate_action_items
4. When the user asks a follow-up referencing previous papers, use the titles from memory
5. For metric questions always use calculate_ml_metrics, never approximate in prose
6. Be precise about paper titles and arxiv IDs — never fabricate them"""

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
        max_output_tokens=2048,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


def _parse_papers_from_tool_output(text: str) -> list:
    """
    Parses the structured output of search_arxiv_papers tool into paper dicts.
    """
    import re
    papers = []
    blocks = re.split(r"\[\d+\]", text)

    for block in blocks[1:]:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
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

            # Extract final assistant text
            output = _to_str(messages[-1].content)

            # Extract papers from all ToolMessage outputs in this turn
            papers = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_text = _to_str(msg.content)
                    if "Found" in tool_text and "papers for" in tool_text:
                        parsed = _parse_papers_from_tool_output(tool_text)
                        papers.extend(parsed)

            logger.info(f"[AGENT] Response: {len(output)} chars, {len(papers)} papers extracted")
            return {"output": output, "papers": papers}

        except Exception as e:
            logger.error(f"[AGENT] invoke error: {e}")
            return {"output": f"Error: {str(e)}", "papers": []}

    def clear_memory(self):
        self.thread_id = str(uuid.uuid4())
        logger.info(f"[AGENT] Memory cleared, new thread: {self.thread_id}")


def build_agent(memory_type: str = "buffer") -> AgentWrapper:
    return AgentWrapper(memory_type=memory_type)
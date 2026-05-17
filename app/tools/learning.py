import logging
import os
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("arxiv_assistant")


def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=2048,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


@tool
def generate_action_items(query: str) -> str:
    """
    Generates prioritized action items and next steps after a research session.
    Input format: '<research_topic> | <context>'
    Context is optional — pass a brief summary of papers found or insights gained.
    Examples:
      'efficient fine-tuning of LLMs'
      'medical image segmentation | found 8 papers, top ones use nnUNet and SAM adaptation'
    Returns: prioritized action plan, implementation roadmap, resources to read next.
    """
    logger.info(f"[TOOL] generate_action_items: {query}")

    try:
        parts = [p.strip() for p in query.split("|")]
        topic = parts[0]
        context = parts[1] if len(parts) > 1 else ""

        if not topic.strip():
            return "Please provide the research topic."

        llm = _get_llm()
        prompt = f"""Based on a research session on "{topic}"{f' with context: {context}' if context else ''}, generate a comprehensive action plan.

Structure as:
1. **Key Takeaways** — 3-5 most important insights from this research area
2. **Immediate Actions** (this week) — concrete tasks you can start right now
3. **Short-term Goals** (1 month) — what to build or implement
4. **Deep Dives** — 3 specific papers or techniques worth studying in depth
5. **Implementation Roadmap** — step-by-step path from reading to working code
6. **Community & Resources** — relevant GitHub repos, communities, benchmarks to follow
7. **Open Questions** — important unanswered questions in this area worth investigating

Be specific. Include GitHub repo names, conference names (NeurIPS, ICML, CVPR etc.), 
and concrete implementation suggestions.
"""
        response = llm.invoke(prompt)
        logger.info("[TOOL] generate_action_items completed")
        return response.content

    except Exception as e:
        logger.error(f"[TOOL] generate_action_items error: {e}")
        return f"Error generating action items: {str(e)}"
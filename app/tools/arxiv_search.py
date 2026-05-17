import logging
import arxiv
from langchain.tools import tool

logger = logging.getLogger("arxiv_assistant")

CATEGORY_MAP = {
    "machine learning": "cs.LG",
    "computer vision": "cs.CV",
    "nlp": "cs.CL",
    "natural language processing": "cs.CL",
    "reinforcement learning": "cs.AI",
    "robotics": "cs.RO",
    "neural networks": "cs.NE",
}


@tool
def search_arxiv_papers(query: str) -> str:
    """
    Searches ArXiv for papers relevant to a research topic.
    Input format: '<topic> | <max_results> | <sort_by>'
    - max_results: integer 3-15, default 8
    - sort_by: 'relevance' or 'date', default 'relevance'
    Examples:
      'vision transformers for medical imaging'
      'diffusion models | 10 | date'
      'RLHF alignment | 5 | relevance'
    Returns structured paper data including title, authors, abstract, ArXiv ID, PDF link.
    """
    logger.info(f"[TOOL] search_arxiv_papers: {query}")

    try:
        parts = [p.strip() for p in query.split("|")]
        topic = parts[0]
        max_results = int(parts[1]) if len(parts) > 1 else 8
        sort_raw = parts[2].lower() if len(parts) > 2 else "relevance"

        max_results = max(3, min(15, max_results))
        sort_by = (
            arxiv.SortCriterion.Relevance
            if sort_raw == "relevance"
            else arxiv.SortCriterion.SubmittedDate
        )

        if not topic.strip():
            return "Please provide a search topic."

        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=sort_by,
        )

        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            results.append({
                "title": paper.title,
                "authors": authors,
                "published": paper.published.strftime("%Y-%m-%d"),
                "arxiv_id": paper.entry_id.split("/")[-1],
                "pdf_url": paper.pdf_url,
                "abstract": paper.summary[:400] + "..." if len(paper.summary) > 400 else paper.summary,
                "categories": paper.categories[:3],
            })

        if not results:
            return f"No papers found for '{topic}'. Try broader search terms."

        output_lines = [f"Found {len(results)} papers for '{topic}':\n"]
        for i, p in enumerate(results, 1):
            output_lines.append(
                f"[{i}] {p['title']}\n"
                f"    Authors: {p['authors']}\n"
                f"    Published: {p['published']} | ArXiv ID: {p['arxiv_id']}\n"
                f"    PDF: {p['pdf_url']}\n"
                f"    Categories: {', '.join(p['categories'])}\n"
                f"    Abstract: {p['abstract']}\n"
            )

        logger.info(f"[TOOL] search_arxiv_papers returned {len(results)} results")
        return "\n".join(output_lines)

    except ValueError as e:
        return f"Invalid input format: {str(e)}"
    except Exception as e:
        logger.error(f"[TOOL] search_arxiv_papers error: {e}")
        return f"Search error: {str(e)}. Please try again with a different query."
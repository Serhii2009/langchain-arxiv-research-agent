import logging
import os
import arxiv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("arxiv_assistant")


def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=2048,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


@tool
def analyze_paper(query: str) -> str:
    """
    Deeply analyzes a specific ArXiv paper given its title or ArXiv ID.
    Input format: '<paper_title_or_arxiv_id>'
    Examples:
      'Attention Is All You Need'
      '2303.08774'
      'LoRA: Low-Rank Adaptation of Large Language Models'
    Extracts: problem statement, methodology, key contributions, results, limitations, and relevance.
    """
    logger.info(f"[TOOL] analyze_paper: {query}")

    try:
        query_clean = query.strip()
        if not query_clean:
            return "Please provide a paper title or ArXiv ID."

        client = arxiv.Client()

        # Try by ArXiv ID first, then by title search
        is_id = all(c.isdigit() or c == "." or c == "v" for c in query_clean.replace("-", ""))
        if is_id or len(query_clean) < 15:
            search = arxiv.Search(id_list=[query_clean])
        else:
            search = arxiv.Search(query=f'ti:"{query_clean}"', max_results=1)

        paper = next(client.results(search), None)

        if paper is None:
            # Fallback: general search
            search = arxiv.Search(query=query_clean, max_results=1)
            paper = next(client.results(search), None)

        if paper is None:
            return f"Paper not found: '{query_clean}'. Try using the exact title or ArXiv ID."

        authors = ", ".join(a.name for a in paper.authors[:5])
        if len(paper.authors) > 5:
            authors += " et al."

        llm = _get_llm()
        analysis_prompt = f"""Analyze this research paper deeply:

Title: {paper.title}
Authors: {authors}
Published: {paper.published.strftime('%Y-%m-%d')}
Abstract: {paper.summary}

Provide a structured analysis:
1. **Problem** — What specific problem does this paper solve?
2. **Methodology** — What approach/architecture/algorithm do they use?
3. **Key Contributions** — List 3-5 concrete contributions
4. **Results** — Main quantitative or qualitative results
5. **Limitations** — What are the known weaknesses or future work mentioned?
6. **Practical Relevance** — When should an ML engineer apply this work?
"""
        response = llm.invoke(analysis_prompt)

        result = (
            f"📄 Paper Analysis\n"
            f"{'='*50}\n"
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"ArXiv ID: {paper.entry_id.split('/')[-1]}\n"
            f"PDF: {paper.pdf_url}\n"
            f"{'='*50}\n\n"
            f"{response.content}"
        )

        logger.info(f"[TOOL] analyze_paper completed for: {paper.title[:50]}")
        return result

    except StopIteration:
        return f"Paper not found: '{query}'"
    except Exception as e:
        logger.error(f"[TOOL] analyze_paper error: {e}")
        return f"Analysis error: {str(e)}"
import logging
import os
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("arxiv_assistant")


def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        max_output_tokens=2048,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


@tool
def design_experiment(query: str) -> str:
    """
    Designs a concrete experiment based on a research topic or set of papers.
    Input format: '<research_topic_or_paper_titles> | <resources>'
    Resources: 'limited' (laptop/Colab), 'moderate' (cloud GPU), 'full' (cluster).
    Examples:
      'LoRA fine-tuning for domain adaptation | limited'
      'vision transformers vs CNNs on medical images | moderate'
      'RLHF for code generation | full'
    Returns: hypothesis, dataset choice, model setup, evaluation plan, expected timeline.
    """
    logger.info(f"[TOOL] design_experiment: {query}")

    try:
        parts = [p.strip() for p in query.split("|")]
        topic = parts[0]
        resources = parts[1].lower() if len(parts) > 1 else "moderate"

        valid_resources = {"limited", "moderate", "full"}
        if resources not in valid_resources:
            resources = "moderate"

        if not topic.strip():
            return "Please provide a research topic or paper titles."

        resource_context = {
            "limited": "Google Colab free tier or personal laptop (≤16GB RAM, no dedicated GPU or T4)",
            "moderate": "Cloud GPU instance (A100 40GB or equivalent, ~$2-5/hr budget)",
            "full": "Multi-GPU cluster or TPU pod, no compute constraints",
        }

        llm = _get_llm()
        prompt = f"""Design a rigorous ML experiment for: "{topic}"
Available resources: {resource_context[resources]}

Structure your experiment plan as:
1. **Hypothesis** — Clear, falsifiable hypothesis
2. **Baseline** — What to compare against and why
3. **Dataset** — Which dataset(s) to use, split strategy, preprocessing steps
4. **Model Setup** — Architecture, hyperparameters, training config (specific values)
5. **Evaluation Protocol** — Metrics, statistical significance tests, ablation plan
6. **Expected Timeline** — Phase breakdown with realistic time estimates
7. **Risk Factors** — What could go wrong and mitigation strategies
8. **Success Criteria** — How to know if the experiment succeeded

Be concrete and actionable. Include specific library names, model names, and parameter values.
"""
        response = llm.invoke(prompt)
        logger.info("[TOOL] design_experiment completed")
        return response.content

    except Exception as e:
        logger.error(f"[TOOL] design_experiment error: {e}")
        return f"Experiment design error: {str(e)}"
import logging
import json
import math
from langchain.tools import tool

logger = logging.getLogger("arxiv_assistant")


@tool
def calculate_ml_metrics(query: str) -> str:
    """
    Calculates ML evaluation metrics from confusion matrix values.
    Input: JSON string with keys tp, tn, fp, fn (all non-negative integers).
    Example: '{"tp": 450, "tn": 8200, "fp": 300, "fn": 50}'
    Returns: Accuracy, Precision, Recall, F1, Specificity, MCC, AUC-ROC (approximated).
    """
    logger.info(f"[TOOL] calculate_ml_metrics: {query}")

    try:
        data = json.loads(query)
        required = {"tp", "tn", "fp", "fn"}
        missing = required - set(data.keys())
        if missing:
            return f"Missing fields: {missing}. Required: tp, tn, fp, fn"

        tp, tn, fp, fn = int(data["tp"]), int(data["tn"]), int(data["fp"]), int(data["fn"])

        if any(v < 0 for v in [tp, tn, fp, fn]):
            return "All values must be non-negative integers."

        total = tp + tn + fp + fn
        if total == 0:
            return "Total cannot be zero."

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0.0
        auc = (recall + specificity) / 2

        imbalance_ratio = abs((tp + fn) - (tn + fp)) / total
        imbalance_note = (
            "⚠️  Class imbalance detected — prioritize F1 and MCC over Accuracy."
            if imbalance_ratio > 0.3
            else "✓  Dataset appears reasonably balanced."
        )

        return (
            f"📊 Metrics Report\n"
            f"{'='*40}\n"
            f"Confusion Matrix: TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}\n"
            f"Total samples: {total:,}\n\n"
            f"Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)\n"
            f"Precision   : {precision:.4f}\n"
            f"Recall      : {recall:.4f}\n"
            f"Specificity : {specificity:.4f}\n"
            f"F1 Score    : {f1:.4f}\n"
            f"MCC         : {mcc:.4f}\n"
            f"AUC (approx): {auc:.4f}\n\n"
            f"{imbalance_note}"
        )

    except json.JSONDecodeError:
        return 'Invalid JSON. Example: {"tp": 90, "tn": 850, "fp": 10, "fn": 50}'
    except Exception as e:
        logger.error(f"[TOOL] calculate_ml_metrics error: {e}")
        return f"Calculation error: {str(e)}"
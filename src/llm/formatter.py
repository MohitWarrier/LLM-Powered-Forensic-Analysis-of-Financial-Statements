def prompt_builder(results: dict) -> str:
    """
    Takes the output of analyze_company and returns a natural prompt for LLM.
    Only includes flagged items.
    """
    flagged = {k: v for k, v in results.items() if v.get("flagged")}

    if not flagged:
        return "No major financial red flags were detected."

    parts = []
    for metric, data in flagged.items():
        val = data["value"]
        reason = data["reason"]
        parts.append(f"{metric.replace('_', ' ').title()}: {val} — {reason}")

    prompt = (
        "The following financial red flags were detected in a company's financials:\n\n"
        + "\n".join(f"- {line}" for line in parts)
        + "\n\nSummarize the concerns in clear, simple terms."
    )
    
    return prompt

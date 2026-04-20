from agent.state import ReportOutput


def to_markdown(report: ReportOutput) -> str:
    lines = [
        f"# {report.title}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Key Findings",
        "",
    ]
    for i, finding in enumerate(report.key_findings, 1):
        lines.append(f"### {i}. {finding.claim}")
        lines.append(f"*Confidence: {finding.confidence}*")
        lines.append("")
        for c in finding.citations:
            lines.append(
                f"- [{c.title}]({c.source_url}) `{c.source_type}` (trust: {c.trust_score:.2f})"
            )
        lines.append("")

    if report.emerging_trends:
        lines.extend(["## Emerging Trends", ""])
        for trend in report.emerging_trends:
            lines.append(f"- {trend}")
        lines.append("")

    if report.recommended_next_steps:
        lines.extend(["## Recommended Next Steps", ""])
        for step in report.recommended_next_steps:
            lines.append(f"- {step}")
        lines.append("")

    if report.model_disagreements:
        lines.extend(["## Model Disagreements (Flagged)", ""])
        for d in report.model_disagreements:
            lines.append(f"> {d}")
        lines.append("")

    lines.extend(["## Sources", ""])
    for c in report.sources:
        lines.append(f"- [{c.title}]({c.source_url}) — {c.source_type}")

    return "\n".join(lines)

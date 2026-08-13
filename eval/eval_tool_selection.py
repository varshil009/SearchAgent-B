"""
Agent Tool Selection Evaluation Script

Evaluates the LLM's ability to select the correct tool (websearch, wikisearch,
node_python) for a given user prompt by running each eval case through NodeLLM
and comparing the predicted tool against the ground truth.

Captures the raw LLM response for debugging without modifying any graph code.

Usage:
    python -m eval.eval_tool_selection

Output:
    eval/eval_report.json   — Structured report (JSON)
    eval/eval_report.txt    — Human-readable report (TXT)
"""

import ast
import csv
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

# ── Project imports (must be runnable from the project root) ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.node_llm import NodeLLM
from graph.agent_state import AgentState

# ── Constants ─────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).resolve().parent / "agent_tool_selection_eval.csv"
REPORT_JSON = Path(__file__).resolve().parent / "eval_report.json"
REPORT_TXT = Path(__file__).resolve().parent / "eval_report.txt"

GROUND_TRUTH_MAP = {
    "web_search": "websearch",
    "wikipedia_search": "wikisearch",
    "python_repl": "node_python",
}

# Cost weights for Cost Efficiency Index
COST_WEIGHTS = {
    "node_python": 1,
    "wikisearch": 2,
    "websearch": 5,
    "final": 0,
    "backup": 10,  # Penalty for failures
}


class EvalNodeLLM(NodeLLM):
    """
    Thin eval-only subclass of NodeLLM that captures the raw LLM response
    without modifying the original graph code.
    """

    def generate(self, state: AgentState) -> dict[str, Any]:
        # Call the original generate method
        result = super().generate(state)
        # The raw response was captured by the monkey-patched client
        # and stored on the instance
        raw_response = getattr(self, "_captured_raw_response", "")
        result["raw_llm_response"] = raw_response
        return result


def load_eval_cases() -> list[dict]:
    """Load and parse the CSV into a list of eval case dicts."""
    cases = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["ground_truth_tool"] = GROUND_TRUTH_MAP.get(
                row["ground_truth_tool"], row["ground_truth_tool"]
            )
            cases.append(row)
    return cases


def run_single_eval(node_llm: EvalNodeLLM, case: dict) -> dict:
    """
    Run a single eval case through NodeLLM and return the result.
    Captures the raw LLM response by monkey-patching the client.
    """
    prompt = case["prompt"]
    ground_truth = case["ground_truth_tool"]

    state: AgentState = {
        "messages": [HumanMessage(content=prompt)],
        "convo_memory": "",
        "tool_run_id": case["id"],
    }

    # Monkey-patch the LLM client's generate_response to capture raw output
    original_generate = node_llm.llm.generate_response

    def capturing_generate(convo, system_prompt):
        response = original_generate(convo, system_prompt)
        node_llm._captured_raw_response = response
        return response

    node_llm.llm.generate_response = capturing_generate
    node_llm._captured_raw_response = ""

    start_time = time.time()
    try:
        result = node_llm.generate(state)
    except Exception as exc:
        result = {"next_action": "backup", "error": str(exc), "raw_llm_response": f"EXCEPTION: {exc}"}
    elapsed = time.time() - start_time

    # Restore original method
    node_llm.llm.generate_response = original_generate

    predicted_tool = result.get("next_action", "backup")
    tool_query = None
    if "tool_request" in result and result["tool_request"]:
        tool_query = result["tool_request"].get("tool_query")

    raw_llm_response = result.get("raw_llm_response", "")

    is_correct = predicted_tool == ground_truth

    return {
        "id": case["id"],
        "prompt": prompt,
        "ground_truth": ground_truth,
        "prediction": predicted_tool,
        "tool_query": tool_query,
        "raw_llm_response": raw_llm_response[:200],  # First 200 chars for compact report
        "raw_llm_response_full": raw_llm_response,
        "match": is_correct,
        "domain": case.get("domain_category", ""),
        "rationale": case.get("selection_rationale", ""),
        "latency_sec": round(elapsed, 3),
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute all evaluation metrics from the per-row results."""
    total = len(results)
    correct = sum(1 for r in results if r["match"])
    wrong = total - correct

    # ── Per-tool metrics ──────────────────────────────────────────────
    tools = ["websearch", "wikisearch", "node_python"]
    tool_metrics = {}

    for tool in tools:
        true_positives = sum(
            1 for r in results if r["ground_truth"] == tool and r["prediction"] == tool
        )
        false_positives = sum(
            1 for r in results if r["ground_truth"] != tool and r["prediction"] == tool
        )
        false_negatives = sum(
            1 for r in results if r["ground_truth"] == tool and r["prediction"] != tool
        )
        support = sum(1 for r in results if r["ground_truth"] == tool)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        tool_metrics[tool] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    macro_f1 = (
        sum(tool_metrics[t]["f1_score"] for t in tools) / len(tools)
        if tools
        else 0.0
    )

    # ── Confusion Matrix ──────────────────────────────────────────────
    confusion = {}
    for gt in tools:
        confusion[gt] = {}
        for pred in tools + ["final", "backup"]:
            confusion[gt][pred] = sum(
                1
                for r in results
                if r["ground_truth"] == gt and r["prediction"] == pred
            )

    # ── Tool Argument Quality ─────────────────────────────────────────
    python_executable_count = 0
    python_total = 0
    concise_query_count = 0
    query_total = 0

    for r in results:
        if r["prediction"] == "node_python" and r["tool_query"]:
            python_total += 1
            try:
                ast.parse(r["tool_query"])
                python_executable_count += 1
            except SyntaxError:
                pass

        if r["prediction"] in ("websearch", "wikisearch") and r["tool_query"]:
            query_total += 1
            # Heuristic: concise queries are short, no filler phrases
            query = r["tool_query"].strip()
            if (
                len(query.split()) <= 20
                and not re.search(
                    r"(?i)\b(can you|please|could you|i need|find|search for|look up)\b",
                    query,
                )
            ):
                concise_query_count += 1

    python_exec_rate = (
        round(python_executable_count / python_total, 4) if python_total > 0 else 0.0
    )
    query_conciseness = (
        round(concise_query_count / query_total, 4) if query_total > 0 else 0.0
    )

    # ── Cost Efficiency Index ─────────────────────────────────────────
    total_cost_score = 0.0
    max_cost_score = 0.0
    for r in results:
        gt_weight = COST_WEIGHTS.get(r["ground_truth"], 5)
        pred_weight = COST_WEIGHTS.get(r["prediction"], 5)
        max_cost_score += gt_weight
        if r["match"]:
            total_cost_score += gt_weight
        else:
            # Penalty: if heavier tool was used, add penalty
            penalty = max(0, pred_weight - gt_weight)
            total_cost_score += max(0, gt_weight - penalty)

    cost_efficiency = (
        round(total_cost_score / max_cost_score, 4) if max_cost_score > 0 else 0.0
    )

    # ── Latency Stats ─────────────────────────────────────────────────
    latencies = [r["latency_sec"] for r in results]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    max_latency = round(max(latencies), 3) if latencies else 0.0
    min_latency = round(min(latencies), 3) if latencies else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "tool_metrics": tool_metrics,
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": confusion,
        "tool_argument_quality": {
            "python_executability_rate": python_exec_rate,
            "python_executable_count": python_executable_count,
            "python_total": python_total,
            "query_conciseness_score": query_conciseness,
            "concise_query_count": concise_query_count,
            "query_total": query_total,
        },
        "cost_efficiency_index": cost_efficiency,
        "latency": {
            "avg_sec": avg_latency,
            "max_sec": max_latency,
            "min_sec": min_latency,
        },
    }


def format_report(metrics: dict, results: list[dict]) -> str:
    """Format the evaluation report as a human-readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("Agent Tool Selection Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Total Evaluations: {metrics['total']}")
    lines.append("")

    # ── Overall Accuracy ──
    lines.append("--- Overall Accuracy ---")
    lines.append(f"  Correct:  {metrics['correct']} / {metrics['total']} ({metrics['accuracy'] * 100:.2f}%)")
    lines.append(f"  Wrong:    {metrics['wrong']} / {metrics['total']} ({metrics['wrong'] / metrics['total'] * 100:.2f}%)")
    lines.append("")

    # ── Per-Tool Metrics ──
    lines.append("--- Per-Tool Metrics ---")
    lines.append(f"{'Tool':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    lines.append("-" * 66)
    for tool in ["websearch", "wikisearch", "node_python"]:
        m = metrics["tool_metrics"][tool]
        lines.append(
            f"{tool:<20} {m['precision'] * 100:<11.2f}% {m['recall'] * 100:<11.2f}% "
            f"{m['f1_score']:<12.4f} {m['support']:<10}"
        )
    lines.append("")
    lines.append(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    lines.append("")

    # ── Confusion Matrix ──
    lines.append("--- Confusion Matrix (rows=ground_truth, cols=prediction) ---")
    headers = ["websearch", "wikisearch", "node_python", "final", "backup"]
    lines.append(f"{'':<20} {'websearch':<12} {'wikisearch':<12} {'node_python':<12} {'final':<8} {'backup':<8}")
    lines.append("-" * 72)
    for gt in ["websearch", "wikisearch", "node_python"]:
        row = [str(metrics["confusion_matrix"][gt][h]) for h in headers]
        lines.append(f"{gt:<20} {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<8} {row[4]:<8}")
    lines.append("")

    # ── Tool Argument Quality ──
    q = metrics["tool_argument_quality"]
    lines.append("--- Tool Argument Quality ---")
    lines.append(f"  Python Executability Rate: {q['python_executable_count']} / {q['python_total']} ({q['python_executability_rate'] * 100:.2f}%)")
    lines.append(f"  Query Conciseness Score:   {q['concise_query_count']} / {q['query_total']} ({q['query_conciseness_score'] * 100:.2f}%)")
    lines.append("")

    # ── Cost Efficiency ──
    lines.append("--- Cost Efficiency Index ---")
    lines.append(f"  Score: {metrics['cost_efficiency_index'] * 100:.2f}%")
    lines.append("")

    # ── Latency ──
    lines.append("--- Latency ---")
    lines.append(f"  Average: {metrics['latency']['avg_sec']:.3f}s")
    lines.append(f"  Min:     {metrics['latency']['min_sec']:.3f}s")
    lines.append(f"  Max:     {metrics['latency']['max_sec']:.3f}s")
    lines.append("")

    # ── Detailed Results ──
    lines.append("--- Detailed Results ---")
    lines.append(f"{'ID':<12} {'Ground Truth':<16} {'Prediction':<16} {'Match':<8} {'Domain':<30}")
    lines.append("-" * 82)
    for r in results:
        match_symbol = "✓" if r["match"] else "✗"
        lines.append(
            f"{r['id']:<12} {r['ground_truth']:<16} {r['prediction']:<16} "
            f"{match_symbol:<8} {r['domain']:<30}"
        )
    lines.append("")

    # ── Failure Analysis: Raw LLM Responses for wikisearch ──
    wiki_failures = [r for r in results if r["ground_truth"] == "wikisearch" and not r["match"]]
    if wiki_failures:
        lines.append("=" * 70)
        lines.append("Failure Analysis — wikisearch Ground Truth (Raw LLM Responses)")
        lines.append("=" * 70)
        lines.append(f"Total wikisearch failures: {len(wiki_failures)}")
        lines.append("")
        for r in wiki_failures[:10]:  # Show first 10 to keep report readable
            lines.append(f"--- {r['id']} | Predicted: {r['prediction']} | Domain: {r['domain']} ---")
            lines.append(f"  Prompt: {r['prompt'][:120]}")
            lines.append(f"  Raw LLM Response (first 300 chars):")
            raw = r.get("raw_llm_response_full", r.get("raw_llm_response", ""))
            lines.append(f"  {raw[:300]}")
            lines.append("")

    # ── Failure Analysis: Raw LLM Responses for node_python → final ──
    py_failures = [r for r in results if r["ground_truth"] == "node_python" and not r["match"]]
    if py_failures:
        lines.append("=" * 70)
        lines.append("Failure Analysis — node_python Ground Truth (Raw LLM Responses)")
        lines.append("=" * 70)
        lines.append(f"Total node_python failures: {len(py_failures)}")
        lines.append("")
        for r in py_failures:
            lines.append(f"--- {r['id']} | Predicted: {r['prediction']} | Domain: {r['domain']} ---")
            lines.append(f"  Prompt: {r['prompt'][:120]}")
            lines.append(f"  Raw LLM Response (first 300 chars):")
            raw = r.get("raw_llm_response_full", r.get("raw_llm_response", ""))
            lines.append(f"  {raw[:300]}")
            lines.append("")

    return "\n".join(lines)


def main():
    print("Loading eval cases from CSV...")
    cases = load_eval_cases()
    print(f"  Loaded {len(cases)} cases.")

    print("Initializing EvalNodeLLM (no graph code modified)...")
    node_llm = EvalNodeLLM()

    print("Running evaluations...")
    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}: {case['prompt'][:60]}...")
        result = run_single_eval(node_llm, case)
        results.append(result)
        print(f"    → Ground truth: {result['ground_truth']}, Prediction: {result['prediction']}, Match: {result['match']}")

    print("\nComputing metrics...")
    metrics = compute_metrics(results)

    print("Generating reports...")
    # JSON report
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "results": results,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved to: {REPORT_JSON}")

    # Text report
    text_report = format_report(metrics, results)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"  Text report saved to:  {REPORT_TXT}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  Cost Eff:  {metrics['cost_efficiency_index'] * 100:.2f}%")
    print(f"  Avg Lat:   {metrics['latency']['avg_sec']:.3f}s")
    print(f"  Total:      {metrics['correct']}/{metrics['total']} correct")
    print(f"  Report:    {REPORT_TXT}")


if __name__ == "__main__":
    main()
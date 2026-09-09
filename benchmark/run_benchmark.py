import os
import json
import argparse
import time
from pathlib import Path

# Ensure root directory on path
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend import benchmark_travel_agent
from benchmark.adapter import evaluate_itinerary_result
from verification.analysis_formatter import print_analysis_report


def load_dataset(dataset_path: Path) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(eval_results: list[dict]) -> dict:
    total = max(len(eval_results), 1)
    delivered_count = sum(1 for r in eval_results if r["delivered"])
    cs_count = sum(1 for r in eval_results if r["commonsense_pass"])
    hc_count = sum(1 for r in eval_results if r["hard_constraint_pass"])
    final_count = sum(1 for r in eval_results if r["final_pass"])
    avg_score = sum(r["score"] for r in eval_results) / total

    return {
        "total": total,
        "delivery_rate": (delivered_count / total) * 100.0,
        "commonsense_pass_rate": (cs_count / total) * 100.0,
        "hard_constraint_pass_rate": (hc_count / total) * 100.0,
        "final_pass_rate": (final_count / total) * 100.0,
        "avg_score": avg_score,
    }


def main():
    parser = argparse.ArgumentParser(description="NaviGo TravelPlanner Benchmark Suite")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of benchmark queries to run (default: 5)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed LLM & constraint analysis per query")
    args = parser.parse_args()

    dataset_path = BASE_DIR / "benchmark" / "dataset.json"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    queries = load_dataset(dataset_path)[: args.sample_size]
    print(f"\n=========================================================")
    print(f" Starting NaviGo Benchmark Suite (Sample Size: {len(queries)})")
    print(f"=========================================================\n")

    baseline_evals = []
    verifier_evals = []

    for idx, sample in enumerate(queries, 1):
        q_text = sample["query"]
        print(f"[{idx}/{len(queries)}] Processing: '{q_text}'")

        # 1. Baseline Run (Verifier Disabled)
        t0 = time.time()
        try:
            res_base = benchmark_travel_agent(q_text, verifier_enabled=False)
            eval_base = evaluate_itinerary_result(sample, res_base)
        except Exception as exc:
            print(f"  - Baseline error: {exc}")
            eval_base = {"delivered": False, "commonsense_pass": False, "hard_constraint_pass": False, "final_pass": False, "violations_count": 99, "score": 0.0}

        time.sleep(3)  # Pacing pause to respect Groq free-tier rate limits

        # 2. Verifier Enabled Run
        t1 = time.time()
        try:
            res_ver = benchmark_travel_agent(q_text, verifier_enabled=True)
            eval_ver = evaluate_itinerary_result(sample, res_ver)
            if args.verbose:
                print_analysis_report(res_ver, query=q_text)
        except Exception as exc:
            print(f"  - Verifier error: {exc}")
            eval_ver = {"delivered": False, "commonsense_pass": False, "hard_constraint_pass": False, "final_pass": False, "violations_count": 99, "score": 0.0}

        t2 = time.time()
        print(f"  -> Baseline Score: {eval_base['score']:.2f} (Final Pass: {eval_base['final_pass']})")
        print(f"  -> Verifier Score: {eval_ver['score']:.2f} (Final Pass: {eval_ver['final_pass']})\n")

        baseline_evals.append(eval_base)
        verifier_evals.append(eval_ver)
        time.sleep(3)

    m_base = calculate_metrics(baseline_evals)
    m_ver = calculate_metrics(verifier_evals)

    print("\n" + "=" * 65)
    print(" BENCHMARK RESULTS SUMMARY (NaviGo vs. TravelPlanner Framework)")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Baseline (No Verifier)':<20} | {'NaviGo (With Verifier)'}")
    print("-" * 65)
    print(f"{'Delivery Rate':<30} | {m_base['delivery_rate']:>18.1f}% | {m_ver['delivery_rate']:>20.1f}%")
    print(f"{'Commonsense Pass Rate':<30} | {m_base['commonsense_pass_rate']:>18.1f}% | {m_ver['commonsense_pass_rate']:>20.1f}%")
    print(f"{'Hard Constraint Pass Rate':<30} | {m_base['hard_constraint_pass_rate']:>18.1f}% | {m_ver['hard_constraint_pass_rate']:>20.1f}%")
    print(f"{'Final Pass Rate':<30} | {m_base['final_pass_rate']:>18.1f}% | {m_ver['final_pass_rate']:>20.1f}%")
    print(f"{'Average Quality Score':<30} | {m_base['avg_score']:>18.2f}  | {m_ver['avg_score']:>20.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


def load_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        raise ValueError("No metrics provided")

    task_suite_name = metrics_list[0]["task_suite_name"]
    num_trials_per_task = metrics_list[0]["num_trials_per_task"]
    num_tasks_in_suite = metrics_list[0]["num_tasks_in_suite"]

    total_episodes = 0
    total_successes = 0
    evaluated_task_ids = []
    evaluated_task_names = []
    per_task_success_rate = {}
    per_category_evaluated_count = {}
    per_category_success_count = {}

    for metrics in metrics_list:
        if metrics["task_suite_name"] != task_suite_name:
            raise ValueError("Mismatched task_suite_name across metrics")
        if metrics["num_trials_per_task"] != num_trials_per_task:
            raise ValueError("Mismatched num_trials_per_task across metrics")
        if metrics["num_tasks_in_suite"] != num_tasks_in_suite:
            raise ValueError("Mismatched num_tasks_in_suite across metrics")

        total_episodes += metrics["total_episodes"]
        total_successes += metrics["total_successes"]
        evaluated_task_ids.extend(metrics.get("evaluated_task_ids", []))
        evaluated_task_names.extend(metrics.get("evaluated_task_names", []))
        per_task_success_rate.update(metrics.get("per_task_success_rate", {}))

        for category, value in metrics.get("per_category_evaluated_count", {}).items():
            per_category_evaluated_count[category] = per_category_evaluated_count.get(category, 0) + value
        for category, value in metrics.get("per_category_success_count", {}).items():
            per_category_success_count[category] = per_category_success_count.get(category, 0) + value

    duplicates = sorted({task_id for task_id in evaluated_task_ids if evaluated_task_ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"Found duplicated evaluated task ids: {duplicates[:20]}")

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    per_category_evaluated_success_rate = {
        category: (
            float(per_category_success_count.get(category, 0)) / float(count) if count > 0 else 0.0
        )
        for category, count in per_category_evaluated_count.items()
    }

    return {
        "task_suite_name": task_suite_name,
        "num_trials_per_task": num_trials_per_task,
        "num_tasks_in_suite": num_tasks_in_suite,
        "evaluated_tasks": len(evaluated_task_ids),
        "evaluated_task_ids": sorted(evaluated_task_ids),
        "evaluated_task_names": evaluated_task_names,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "per_task_success_rate": per_task_success_rate,
        "per_category_evaluated_count": per_category_evaluated_count,
        "per_category_success_count": per_category_success_count,
        "per_category_evaluated_success_rate": per_category_evaluated_success_rate,
    }


def compare_metrics(reference: dict, candidate: dict) -> dict:
    return {
        "same_task_ids": sorted(reference.get("evaluated_task_ids", [])) == sorted(candidate.get("evaluated_task_ids", [])),
        "same_total_episodes": reference.get("total_episodes") == candidate.get("total_episodes"),
        "same_total_successes": reference.get("total_successes") == candidate.get("total_successes"),
        "same_per_task_success_rate": reference.get("per_task_success_rate", {}) == candidate.get("per_task_success_rate", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input worker metrics.json files")
    parser.add_argument("--output", required=True, help="Aggregated output json path")
    parser.add_argument("--reference", help="Optional reference metrics.json to compare against")
    args = parser.parse_args()

    metrics_list = [load_metrics(path) for path in args.inputs]
    aggregated = aggregate_metrics(metrics_list)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    if args.reference:
        reference = load_metrics(args.reference)
        comparison = compare_metrics(reference, aggregated)
        comparison_path = output_path.with_name(output_path.stem + "_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

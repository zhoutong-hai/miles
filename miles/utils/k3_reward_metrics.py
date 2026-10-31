"""Numeric evaluation summaries from the existing shared reward results."""

import json
import math
import os


def summarize_evaluation(data):
    expected = json.loads(os.environ["MILES_EVAL_THEME_COUNTS_JSON"])
    if set(data) != set(expected):
        raise ValueError("K3 evaluation theme coverage differs from the sealed input")
    output = {}
    scenarios = []
    means = []
    for theme, count in expected.items():
        point = data[theme]
        rewards, samples = point["rewards"], point["samples"]
        if len(rewards) != count or len(samples) != count:
            raise ValueError("K3 evaluation sample coverage differs")
        if any(value is None or not math.isfinite(float(value)) for value in rewards):
            raise ValueError("K3 evaluation lacks a finite scenario score")
        scenarios.extend(float(value) for value in rewards)
        means.append(sum(rewards) / count)
        for field in ("score", "scenario_score", "style_score", "repetition_score", "veto_min_score"):
            values = [sample.reward.get(field) for sample in samples]
            observed = [float(value) for value in values if value is not None]
            if any(not math.isfinite(value) for value in observed):
                raise ValueError("K3 evaluation contains nonfinite guardrail reward")
            output[f"eval/{theme}/{field}_observed"] = len(observed)
            if observed:
                output[f"eval/{theme}/{field}"] = sum(observed) / len(observed)
    output.update({"eval/scenario_aggregate": sum(scenarios) / len(scenarios),
                   "eval/scenario_macro": sum(means) / len(means),
                   "eval/samples": len(scenarios), "eval/missing_scenario": 0})
    return output

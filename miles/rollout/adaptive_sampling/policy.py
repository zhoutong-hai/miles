"""CPU-only curriculum policy; no generation, reward execution, or loss changes."""

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass

BUCKETS = ("informative", "unseen", "solved", "hard")


@dataclass(frozen=True)
class SamplingConfig:
    # Relative selection weights, not minimum passing quotas.
    weights: tuple[float, ...] = (4, 2, 1, 1)
    strata_keys: tuple[str, ...] = ()
    reward_key: str | None = None
    error_key: str | None = None
    success_reward: float = 1.0
    variation_tolerance: float = 1e-6
    warmup_rollouts: int = 1
    unseen_schedule: str = "fixed"
    unseen_start_fraction: float = 0.75
    unseen_end_fraction: float = 0.25
    # Caller-owned scoring revision; changing it invalidates saved curriculum history.
    reward_version: str = "default"

    @classmethod
    def from_mapping(cls, values: dict):
        values = dict(values)
        for key in ("weights", "strata_keys"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def __post_init__(self):
        if len(self.weights) != len(BUCKETS) or any(not math.isfinite(x) or x <= 0 for x in self.weights):
            raise ValueError("adaptive sampling requires four finite positive bucket weights")
        if not math.isfinite(self.success_reward) or not math.isfinite(self.variation_tolerance):
            raise ValueError("reward thresholds must be finite")
        if self.variation_tolerance < 0 or type(self.warmup_rollouts) is not int or self.warmup_rollouts < 0:
            raise ValueError("invalid variation tolerance or warmup_rollouts")
        if not self.reward_version or any(not isinstance(key, str) or not key for key in self.strata_keys):
            raise ValueError("reward_version and strata_keys must be nonempty strings")
        if self.unseen_schedule not in ("fixed", "coverage_linear"):
            raise ValueError("unseen_schedule must be fixed or coverage_linear")
        if not (
            math.isfinite(self.unseen_start_fraction)
            and math.isfinite(self.unseen_end_fraction)
            and 0 < self.unseen_end_fraction <= self.unseen_start_fraction <= 1
        ):
            raise ValueError("coverage fractions require 0 < end <= start <= 1")


def coverage_unseen_fraction(*, unseen: int, total: int, start: float, end: float) -> float:
    """Discovery probability, linear in remaining coverage, not training steps."""
    if total <= 0 or not 0 <= unseen <= total:
        raise ValueError("invalid coverage counts")
    return end + (start - end) * unseen / total if unseen else 0.0


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class AdaptivePromptPolicy:
    """Own history and RNG. A row is never retired, and each batch uses distinct rows.

    Stratum credit preserves dataset proportions across batches. Within each stratum,
    soft bucket targets prefer informative history while retaining exploration and
    solved/hard revisits. Missing pools fall back; they never trigger extra generation.
    """

    def __init__(self, *, row_fingerprints: list[str], strata: list[str], config: SamplingConfig, seed: int):
        if not strata or len(strata) != len(row_fingerprints):
            raise ValueError("a nonempty dataset with one stratum per row is required")
        self.config = config
        self.strata = tuple(strata)
        identity_config = asdict(config)
        if config.unseen_schedule == "fixed":
            # Preserve existing fixed-policy checkpoint identity and RNG behavior.
            for key in ("unseen_schedule", "unseen_start_fraction", "unseen_end_fraction"):
                identity_config.pop(key)
        self.identity = fingerprint(
            {"rows": row_fingerprints, "strata": strata, "config": identity_config, "seed": seed}
        )
        self._rng = random.Random(seed)
        counts = Counter(strata)
        self._stratum_sizes = counts
        self._proportions = {key: count / len(strata) for key, count in counts.items()}
        self._credit = dict.fromkeys(counts, 0.0)
        self._history = [{"bucket": "unseen", "visits": 0, "last_seen": -1, "observations": 0} for _ in strata]
        self.batches = 0
        self._last_selection = {}

    def select(self, count: int) -> tuple[list[int], dict[str, int]]:
        if not 0 < count <= len(self.strata):
            raise ValueError("rollout_batch_size must be positive and no larger than the dataset")
        candidates = {}
        unseen_counts = Counter()
        for row, stratum in enumerate(self.strata):
            candidates.setdefault(stratum, []).append(row)
            if self._history[row]["visits"] == 0:
                unseen_counts[stratum] += 1
        for stratum, proportion in self._proportions.items():
            self._credit[stratum] += count * proportion
        targets = {
            key: count * weight / sum(self.config.weights)
            for key, weight in zip(BUCKETS, self.config.weights, strict=True)
        }
        selected, actual = [], Counter()
        scheduled_sum, new_rows = 0.0, 0
        for _ in range(count):
            available_strata = [key for key, rows in candidates.items() if rows]
            stratum = max(available_strata, key=lambda key: (self._credit[key], self._rng.random()))
            if self.config.unseen_schedule == "coverage_linear":
                row, share = self._select_coverage_row(
                    candidates[stratum], unseen_counts[stratum], self._stratum_sizes[stratum]
                )
                scheduled_sum += share
            else:
                row, lane = self._select_row(candidates[stratum], targets)
                targets[lane] -= 1
            bucket = self._history[row]["bucket"]
            self._credit[stratum] -= 1
            candidates[stratum].remove(row)
            selected.append(row)
            actual[bucket] += 1
            if self._history[row]["visits"] == 0:
                unseen_counts[stratum] -= 1
                new_rows += 1
            self._history[row]["visits"] += 1
            self._history[row]["last_seen"] = self.batches
        self.batches += 1
        if self.config.unseen_schedule == "coverage_linear":
            self._last_selection = {
                "scheduled_unseen_fraction": scheduled_sum / count,
                "selected_new_rows": new_rows,
                "selected_revisit_rows": count - new_rows,
                "actual_new_fraction": new_rows / count,
            }
        return selected, {key: actual[key] for key in BUCKETS}

    def _select_coverage_row(self, rows: list[int], unseen: int, total: int) -> tuple[int, float]:
        share = coverage_unseen_fraction(
            unseen=unseen,
            total=total,
            start=self.config.unseen_start_fraction,
            end=self.config.unseen_end_fraction,
        )
        if self.batches < self.config.warmup_rollouts:
            return self._rng.choice(rows), share
        new = [row for row in rows if self._history[row]["visits"] == 0]
        revisits = [row for row in rows if self._history[row]["visits"] > 0]
        if new and (not revisits or self._rng.random() < share):
            return self._rng.choice(new), share
        # Invalid judgments do not make a presented row new or difficult. Such
        # rows get a neutral refresh lane, without changing trusted bucket labels.
        pools = {key: [row for row in revisits if self._history[row]["bucket"] == key] for key in BUCKETS}
        weights = dict(zip(BUCKETS, self.config.weights, strict=True))
        weights["unseen"] = 1.0  # Unrated revisits, NOT the scheduled discovery weight.
        available = [key for key in BUCKETS if pools[key]]
        lane = self._rng.choices(available, weights=[weights[key] for key in available], k=1)[0]
        row = min(pools[lane], key=lambda row: (self._history[row]["last_seen"], self._rng.random()))
        return row, share

    def _select_row(self, rows: list[int], targets: dict[str, float]) -> tuple[int, str]:
        if self.batches < self.config.warmup_rollouts:
            row = self._rng.choice(rows)
            return row, self._history[row]["bucket"]
        pools = {bucket: [row for row in rows if self._history[row]["bucket"] == bucket] for bucket in BUCKETS}
        # Once unseen rows are exhausted, its allocation becomes a least-recently
        # sampled exploration lane across all buckets, not additional exploitation.
        pools["unseen"] = pools["unseen"] or rows
        available = [bucket for bucket in BUCKETS if pools[bucket]]
        bucket = max(available, key=lambda key: (targets[key], self._rng.random()))
        row = min(pools[bucket], key=lambda row: (self._history[row]["last_seen"], self._rng.random()))
        return row, bucket

    def observe(self, row: int, rewards: list[float] | None) -> bool:
        """Ignore unreliable judgments; never turn provider failures into difficulty."""
        if not rewards or any(not math.isfinite(value) for value in rewards):
            return False
        low, high = min(rewards), max(rewards)
        if low >= self.config.success_reward:
            bucket = "solved"
        elif high - low > self.config.variation_tolerance:
            bucket = "informative"
        else:
            bucket = "hard"
        self._history[row]["bucket"] = bucket
        self._history[row]["observations"] += 1
        return True

    def metrics(self) -> dict[str, float]:
        counts = Counter(item["bucket"] for item in self._history)
        return {
            **{f"pool_{key}": counts[key] for key in BUCKETS},
            "unique_rows_seen": sum(item["visits"] > 0 for item in self._history),
            "prompt_presentations": sum(item["visits"] for item in self._history),
            **self._last_selection,
            **(
                {"unseen_rows_remaining": sum(item["visits"] == 0 for item in self._history)}
                if self.config.unseen_schedule == "coverage_linear"
                else {}
            ),
        }

    def state_dict(self) -> dict:
        return {
            "version": 2 if self.config.unseen_schedule == "coverage_linear" else 1,
            "identity": self.identity,
            "rng": self._rng.getstate(),
            "credit": dict(self._credit),
            "history": [dict(item) for item in self._history],
            "batches": self.batches,
            **(
                {"last_selection": dict(self._last_selection)}
                if self.config.unseen_schedule == "coverage_linear"
                else {}
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        version = 2 if self.config.unseen_schedule == "coverage_linear" else 1
        if state.get("version") != version or state.get("identity") != self.identity:
            raise ValueError("adaptive sampling checkpoint dataset/config/reward revision mismatch")
        if len(state["history"]) != len(self.strata) or set(state["credit"]) != set(self._credit):
            raise ValueError("invalid adaptive sampling checkpoint shape")
        if (
            type(state["batches"]) is not int
            or state["batches"] < 0
            or any(not math.isfinite(value) for value in state["credit"].values())
        ):
            raise ValueError("invalid adaptive sampling checkpoint counters")
        for item in state["history"]:
            if item["bucket"] not in BUCKETS or any(
                type(item[key]) is not int or item[key] < minimum
                for key, minimum in (("visits", 0), ("observations", 0), ("last_seen", -1))
            ):
                raise ValueError("invalid adaptive sampling checkpoint history")
        self._rng.setstate(state["rng"])
        self._credit = dict(state["credit"])
        self._history = [dict(item) for item in state["history"]]
        self.batches = state["batches"]
        self._last_selection = dict(state.get("last_selection", {}))

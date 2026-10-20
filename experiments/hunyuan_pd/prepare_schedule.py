"""Freeze three rotating N=C batches without loading a tokenizer or a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def describe(rows):
    result = {}
    for group in ("short", "long"):
        for language in ("zh", "en"):
            selected = [r for r in rows if r["group"] == group and r["language"] == language]
            if selected:
                lengths = [r["char_count"] for r in selected]
                result[f"{group}_{language}"] = {
                    "n": len(selected),
                    "mean_chars": statistics.mean(lengths),
                    "median_chars": statistics.median(lengths),
                    "min_chars": min(lengths),
                    "max_chars": max(lengths),
                }
    return result


def quotas(group, concurrency, round_number):
    groups = ("short", "long") if group == "mixed" else (group,)
    result = {}
    english_total = concurrency // 4
    for index, name in enumerate(groups):
        count = concurrency // len(groups)
        english = english_total if len(groups) == 1 else english_total // 2
        if len(groups) == 2 and english_total % 2 and index == (round_number - 1) % 2:
            english += 1
        result[(name, "en")] = english
        result[(name, "zh")] = count - english
    return result


def choose_pd(catalog, group, round_number):
    selected = []
    for (name, language), count in quotas(group, 24, round_number).items():
        pool = sorted(
            [r for r in catalog if r["group"] == name and r["language"] == language],
            key=lambda r: (r["char_count"], r["id"]),
        )
        # Two within-language character-length strata; rotate within each.
        halves = (pool[: len(pool) // 2], pool[len(pool) // 2 :])
        low_count = count // 2 + (count % 2 if round_number % 2 else 0)
        for half, needed in zip(halves, (low_count, count - low_count)):
            offset = (round_number - 1) * max(1, len(half) - needed)
            selected.extend(half[(offset + i) % len(half)] for i in range(needed))
    random.Random(4200 + round_number).shuffle(selected)
    return selected


def choose_subset(pd_rows, group, concurrency, round_number):
    rng = random.Random(f"42:{group}:{concurrency}:{round_number}")
    selected_ids = set()
    for (name, language), count in quotas(group, concurrency, round_number).items():
        pool = sorted(
            [r for r in pd_rows if r["group"] == name and r["language"] == language],
            key=lambda r: (r["char_count"], r["id"]),
        )
        target_mean = statistics.mean(r["char_count"] for r in pool)
        target_median = statistics.median(r["char_count"] for r in pool)
        low_ids = {r["id"] for r in pool[: len(pool) // 2]}

        def score(rows):
            lengths = [r["char_count"] for r in rows]
            return (
                abs(statistics.mean(lengths) / target_mean - 1)
                + abs(statistics.median(lengths) / target_median - 1)
                + 0.2 * abs(sum(r["id"] in low_ids for r in rows) / count - len(low_ids) / len(pool))
            )

        # Fixed seed, no performance data used in selection. Tiny strata can
        # make exact distribution matching impossible; retain actual stats.
        candidates = (rng.sample(pool, count) for _ in range(2000))
        selected_ids.update(r["id"] for r in min(candidates, key=score))
    return [r for r in pd_rows if r["id"] in selected_ids]


def build_schedule(output_dir: Path):
    catalog_path = ROOT / "inputs/catalog.jsonl"
    catalog = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()]
    manifest = {
        "version": 1,
        "selection": "language + existing short/long groups + within-language character length; NOT token calibrated",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "rounds": 3,
        "selected_round": 3,
        "points": [],
    }
    if output_dir.exists() and any(p.name != ".gitignore" for p in output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen schedule: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for group in ("short", "long", "mixed"):
        pd_rounds = [choose_pd(catalog, group, r) for r in (1, 2, 3)]
        for deployment, concurrency in (("pd", 24), ("npd", 12), ("npd", 16)):
            point = {"deployment": deployment, "group": group, "concurrency": concurrency, "batches": []}
            seen = set()
            previous = set()
            for r, pd_rows in enumerate(pd_rounds, 1):
                rows = pd_rows if deployment == "pd" else choose_subset(pd_rows, group, concurrency, r)
                assert len(rows) == concurrency and len({row["prompt_sha256"] for row in rows}) == concurrency
                ids = {row["id"] for row in rows}
                path = f"{deployment}-{group}-c{concurrency}/round{r}.jsonl"
                target = output_dir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
                point["batches"].append(
                    {
                        "round": r,
                        "path": path,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "ids": [row["id"] for row in rows],
                        "reused_from_earlier_rounds_in_this_point": len(ids & seen),
                        "changed_from_previous_round": len(ids - previous) if r > 1 else None,
                        "strata": describe(rows),
                    }
                )
                seen.update(ids)
                previous = ids
            manifest["points"].append(point)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "schedule")
    args = parser.parse_args()
    build_schedule(args.output_dir)
    print(f"Frozen 27 batches: {args.output_dir.resolve()}")

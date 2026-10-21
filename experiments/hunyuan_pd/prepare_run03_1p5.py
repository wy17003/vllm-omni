"""Freeze PD48/NPD32 inputs from run03; retain PD bytes and all prior NPD inputs."""

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path

from prepare_run03 import write_json
from probe_t2i_batch import load_frozen_batch

ROOT = Path(__file__).resolve().parent


def extend_subset(parent, previous):
    """Expand each language to 2/3 of PD, minimizing the input-token mean difference."""
    selected = {r["id"] for r in previous}
    parent_by_id = {r["id"]: r for r in parent}
    if len(selected) != len(previous) or any(parent_by_id.get(r["id"]) != r for r in previous):
        raise ValueError("Previous NPD inputs must be an unchanged unique subset of PD")
    for language in ("zh", "en"):
        pool = [r for r in parent if r["language"] == language]
        if len(pool) % 3:
            raise ValueError("Cannot select exactly 2/3 of this language stratum")
        count = len(pool) * 2 // 3
        old = [r for r in previous if r["language"] == language]
        remaining = sorted((r for r in pool if r["id"] not in selected), key=lambda r: r["id"])
        old_sum = sum(r["full_input_tokens"] for r in old)
        parent_sum = sum(r["full_input_tokens"] for r in pool)
        # Integer arithmetic makes selection deterministic across platforms.
        extra = min(
            itertools.combinations(remaining, count - len(old)),
            key=lambda rows: (
                abs((old_sum + sum(r["full_input_tokens"] for r in rows)) * len(pool) - parent_sum * count),
                tuple(r["id"] for r in rows),
            ),
        )
        selected.update(r["id"] for r in extra)
    return [r for r in parent if r["id"] in selected]


def prepare(source, output):
    raw_manifest = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if (
        manifest["version"] != 3
        or manifest.get("comparison_profile", "ar_cards_2x") != "ar_cards_2x"
        or manifest["rounds"] != 1
        or manifest["wave_delay_s"] != 5
        or manifest["max_num_batched_tokens"] != 8192
        or manifest["ar_max_num_seqs"] != 64
    ):
        raise ValueError("Expected original run03 source with fixed 5s waves and 8192/64 limits")
    sources = {}
    for point in manifest["points"]:
        path = (source / point["path"]).resolve()
        if source.resolve() not in path.parents:
            raise ValueError("Source path escapes schedule directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != point["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
        count = 48 if point["deployment"] == "pd" else 24
        load_frozen_batch(path, "long", count, "unused-for-generation")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        for index, row in enumerate(rows):
            wave = 1 if point["workload"] == "single" or index < count // 2 else 2
            if row["wave_id"] != wave or row["scheduled_offset_s"] != (0 if wave == 1 else 5):
                raise ValueError("Unexpected source wave assignment")
        sources[(point["deployment"], point["workload"])] = (point, rows, raw)
    if set(sources) != {(d, w) for d in ("pd", "npd") for w in ("single", "waves")}:
        raise ValueError("Expected the four original run03 points")
    prepared, points = {}, []
    for workload in ("single", "waves"):
        pd_point, parent, pd_raw = sources[("pd", workload)]
        _, previous, _ = sources[("npd", workload)]
        selected = []
        for wave in sorted({r["wave_id"] for r in parent}):
            selected.extend(
                extend_subset(
                    [r for r in parent if r["wave_id"] == wave], [r for r in previous if r["wave_id"] == wave]
                )
            )
        for deployment, rows in (("pd", parent), ("npd", selected)):
            name = f"{deployment}-{workload}.jsonl"
            raw = (
                pd_raw
                if deployment == "pd"
                else "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            )
            prepared[name] = raw
            points.append(
                dict(
                    deployment=deployment,
                    workload=workload,
                    path=name,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    requests=len(rows),
                    wave_sizes=[
                        sum(r["wave_id"] == wave for r in rows) for wave in sorted({r["wave_id"] for r in rows})
                    ],
                    full_input_tokens_mean=statistics.mean(r["full_input_tokens"] for r in rows),
                    action="reuse_run03" if deployment == "pd" else "run_once",
                    parent_pd_sha256=pd_point["sha256"],
                )
            )
    manifest.update(
        comparison_profile="total_cards_1p5x",
        points=points,
        source_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        selection=(
            "Retain original NPD subset; add by language/input-token mean within each wave; no output-based selection"
        ),
        ar_cards={"pd": 8, "npd": 4},
        total_cards={"pd": 12, "npd": 8},
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in prepared.items():
        (output / name).write_bytes(raw)
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "schedule_run03")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))

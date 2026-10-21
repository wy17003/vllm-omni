"""Run all frozen experiment points for one already-started PD or non-PD server."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from audit_tokens import load_tokenizer, tokenizer_metadata

ROOT = Path(__file__).resolve().parent


def run(args):
    manifest_path = args.schedule_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points = [p for p in manifest["points"] if p["deployment"] == args.deployment]
    points.sort(key=lambda p: (p["concurrency"], ("short", "long", "mixed").index(p["group"])))
    expected = {
        (group, c) for group in ("short", "long", "mixed") for c in ((24,) if args.deployment == "pd" else (12, 16))
    }
    if len(points) != len(expected) or {(p["group"], p["concurrency"]) for p in points} != expected:
        raise ValueError("Schedule does not contain exactly the expected experiment points")
    points = [p for p in points if p.get("action", "run") == "run" and p["group"] in args.groups]
    if not points:
        raise ValueError("No runnable points selected; run02 PD short/mixed reuse run01")
    if manifest.get("require_token_audit") and not args.tokenizer_json:
        raise ValueError("This schedule requires --tokenizer-json for post-batch token auditing")
    if args.tokenizer_json:
        metadata = tokenizer_metadata(args.tokenizer_json)
        if manifest.get("tokenizer_sha256", metadata["sha256"]) != metadata["sha256"]:
            raise ValueError("Tokenizer differs from the one used to freeze this schedule")
        load_tokenizer(args.tokenizer_json)  # Fail before any HTTP request if dependency/file is invalid.
    # Verify all inputs before submitting any HTTP requests.
    for point in points:
        if [b["round"] for b in point["batches"]] != [1, 2, 3]:
            raise ValueError("Each point must have exactly rounds 1, 2, 3")
        for batch in point["batches"]:
            path = args.schedule_dir / batch["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != batch["sha256"]:
                raise ValueError(f"Frozen input changed: {path}")
    if not args.prepare_only and (not args.server_log or not args.server_log.is_file()):
        raise ValueError("Provide an existing local --server-log for TTFT/TPOT collection")
    destination = args.output_root.resolve() / args.deployment
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, destination / "schedule_manifest.json")
    shutil.copyfile(ROOT / f"configs/{args.deployment}.yaml", destination / "declared_config.yaml")
    # Record the working tree too: core fixes may not have been committed yet.
    repo = ROOT.parent.parent
    for name, command in (
        ("git_head.txt", ["git", "rev-parse", "HEAD"]),
        ("git_status.txt", ["git", "status", "--short"]),
        ("working_tree.patch", ["git", "diff", "HEAD"]),
    ):
        result = subprocess.run(command, cwd=repo, capture_output=True)
        (destination / name).write_bytes(result.stdout + result.stderr)
    limits = [32, 32, 1] if args.deployment == "pd" else [32, 1]
    tag = "_".join(f"s{i}-{n}" for i, n in enumerate(limits))
    results = []
    for point in points:
        point_dir = destination / f"{args.deployment}-{point['group']}-c{point['concurrency']}_pc-on_step-off_{tag}"
        command = [
            sys.executable,
            str(ROOT / "probe_t2i_batch.py"),
            "--deployment",
            args.deployment,
            "--group",
            point["group"],
            "--concurrency",
            str(point["concurrency"]),
            "--rounds",
            "3",
            "--stage-max-num-seqs",
            *map(str, limits),
            "--output-dir",
            str(point_dir),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--timeout",
            str(args.timeout),
            "--batch-files",
            *[str(args.schedule_dir / b["path"]) for b in point["batches"]],
        ]
        if args.server_log:
            command.extend(["--server-log", str(args.server_log.resolve())])
        if args.tokenizer_json:
            command.extend(["--tokenizer-json", str(args.tokenizer_json.resolve())])
        if args.prepare_only:
            command.append("--prepare-only")
        print(f"Running {args.deployment} {point['group']} C={point['concurrency']}", flush=True)
        completed = subprocess.run(command)
        row = dict(
            group=point["group"],
            concurrency=point["concurrency"],
            output_dir=str(point_dir),
            exit_code=completed.returncode,
        )
        if (point_dir / "selected_result.json").exists():
            row["selected_result"] = json.loads((point_dir / "selected_result.json").read_text(encoding="utf-8"))
        results.append(row)
        (destination / "summary.json").write_text(
            json.dumps(
                dict(deployment=args.deployment, prepare_only=args.prepare_only, points=results),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if completed.returncode:
            raise SystemExit("Experiment stopped at failed point. No retries or concurrency changes.")
    print(f"Results: {destination / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, choices=("pd", "npd"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "schedule")
    parser.add_argument("--groups", nargs="+", choices=("short", "long", "mixed"), default=["short", "long", "mixed"])
    parser.add_argument("--tokenizer-json", type=Path, help="Required by run02; checks logged AR token round trips")
    parser.add_argument("--model", default="/new_data/HunyuanImage-3.0-Instruct/")
    parser.add_argument("--base-url", default="http://localhost:9000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    run(args)

"""Recompute stage metrics and token audits into a NEW analysis directory."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_tokens import audit_tokens, load_tokenizer, tokenizer_metadata
from summarize_metrics import parse_stage_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def reparse(source, output, deployment, tokenizer_path):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Analysis output must be outside the source result directory")
    tokenizer = load_tokenizer(tokenizer_path)
    output.mkdir(parents=True, exist_ok=False)
    points = []
    for folder in sorted(source.glob(f"{deployment}-*_pc-*")):
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        target = output / folder.name
        target.mkdir()
        rounds = []
        for number in (1, 2, 3):
            result_path = folder / f"round{number}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            requests_path = folder / f"round{number}_inputs/requests.json"
            requests = json.loads(requests_path.read_text(encoding="utf-8"))
            log_path = folder / f"round{number}_server.log"
            log = log_path.read_text(encoding="utf-8")
            old_metrics = result.get("ar_metrics") or {}
            metrics = parse_stage_metrics(log, deployment, result["request_count"])
            if "log_byte_interval" in old_metrics:
                metrics["log_byte_interval"] = old_metrics["log_byte_interval"]
            audit = audit_tokens(log, requests, tokenizer)
            result["ar_metrics"] = metrics
            result["token_audit"] = audit
            result["analysis_eligible"] = (
                result["valid"]
                and audit["valid"]
                and all(metrics[key]["status"] == "complete" for key in ("ttft_ms", "tpot_ms"))
            )
            result["analysis_sources"] = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (result_path, requests_path, log_path)
            }
            write_json(target / result_path.name, result)
            rounds.append(result)
        third = next((r for r in rounds if r["round"] == 3), None)
        eligible = len(rounds) == 3 and third is not None and third["analysis_eligible"]
        point = dict(
            group=plan["group"],
            concurrency=plan["concurrency"],
            source_dir=str(folder),
            analysis_dir=str(target),
            third_round_eligible=eligible,
            earlier_round_token_anomalies=[
                r["round"] for r in rounds if r["round"] < 3 and not r["token_audit"]["valid"]
            ],
        )
        if third:
            selected = dict(
                selected_round=3,
                valid=eligible,
                success_count=third["success_count"],
                client_wall_s=third["client_wall_s"],
                e2e_s=third["e2e_s"],
                ar_metrics=third["ar_metrics"],
                token_audit=third["token_audit"],
                note=(
                    "Offline reanalysis; original HTTP success is not token consistency. "
                    "Earlier anomalies remain disclosed."
                ),
            )
            write_json(target / "selected_result.json", selected)
            point["selected_result"] = selected
        points.append(point)
    if not points:
        raise ValueError(f"No {deployment} experiment points found in {source}")
    result = dict(
        deployment=deployment, source_root=str(source), tokenizer=tokenizer_metadata(tokenizer_path), points=points
    )
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Deployment directory, e.g. hy3_run01/pd")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; originals are never overwritten")
    parser.add_argument("--deployment", choices=("pd", "npd"), required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    args = parser.parse_args()
    result = reparse(args.source_dir, args.output_dir, args.deployment, args.tokenizer_json)
    for point in result["points"]:
        print(f"{point['group']} C={point['concurrency']}: third_round_eligible={point['third_round_eligible']}")
    print(f"Analysis: {args.output_dir.resolve() / 'summary.json'}")

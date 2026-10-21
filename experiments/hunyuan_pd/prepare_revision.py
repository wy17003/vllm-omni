"""Freeze run02 inputs, retaining run01 PD short/mixed and replacing known risks."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from audit_tokens import audit_tokens, load_tokenizer
from prepare_schedule import ROOT, describe, quotas
from reparse_results import write_json

EXCLUDED = {
    "scene31_short": "NPD short round1: reencoded output grew by one token; HTTP500",
    "scene11_long": "PD long round3 / mixed round2: reencoded output shrank by one token",
    "scene19_long": "PD long round2: same length but different token IDs",
}


def replace_rows(original, pool, token_counts):
    allowed = {row["id"]: row for row in pool if row["id"] not in EXCLUDED}
    occupied = {row["id"] for row in original if row["id"] in allowed}
    result, changes = [], []
    for row in original:
        if row["id"] in allowed:
            result.append(allowed[row["id"]])
            continue
        candidates = [
            candidate
            for candidate in allowed.values()
            if candidate["id"] not in occupied
            and (candidate["group"], candidate["language"]) == (row["group"], row["language"])
        ]
        if not candidates:
            raise ValueError(f"Not enough unique candidates to replace {row['id']}")
        chosen = min(
            candidates,
            key=lambda c: (
                abs(token_counts[c["id"]] - token_counts[row["id"]]),
                abs(c["char_count"] - row["char_count"]),
                c["id"],
            ),
        )
        occupied.add(chosen["id"])
        result.append(chosen)
        changes.append(
            dict(
                old_id=row["id"],
                new_id=chosen["id"],
                old_input_tokens=token_counts[row["id"]],
                new_input_tokens=token_counts[chosen["id"]],
                reason=EXCLUDED.get(row["id"], "not present in revised PD parent batch"),
            )
        )
    return result, changes


def token_stats(rows, counts):
    lengths = [counts[row["id"]] for row in rows]
    return dict(
        n=len(rows), mean=statistics.mean(lengths), p50=statistics.median(lengths), min=min(lengths), max=max(lengths)
    )


def build_revision(baseline_dir, output_dir, tokenizer_path):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen revision: {output_dir}")
    tokenizer = load_tokenizer(tokenizer_path)
    catalog_path = ROOT / "inputs/catalog.jsonl"
    catalog = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in catalog}
    baseline_path = baseline_dir / "schedule_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    counts, sources, audits = {}, {}, []
    for point in baseline["points"]:
        if point["deployment"] != "pd":
            continue
        folder = baseline_dir / f"pd-{point['group']}-c24_pc-on_step-off_s0-32_s1-32_s2-1"
        for batch in point["batches"]:
            number = batch["round"]
            requests_path = folder / f"round{number}_inputs/requests.json"
            requests = json.loads(requests_path.read_text(encoding="utf-8"))
            if [row["id"] for row in requests] != batch["ids"]:
                raise ValueError(f"Baseline request IDs differ from manifest: {requests_path}")
            for row in requests:
                if row["body"]["prompt"] != by_id[row["id"]]["request"]["prompt"]:
                    raise ValueError(f"Baseline prompt differs from catalog: {row['id']}")
            log_path = folder / f"round{number}_server.log"
            audit = audit_tokens(log_path.read_text(encoding="utf-8"), requests, tokenizer)
            if audit["errors"]:
                raise ValueError(f"Cannot establish baseline input lengths: {audit['errors']}")
            if point["group"] in ("short", "mixed") and number == 3 and not audit["roundtrip_consistent"]:
                raise ValueError(f"Cannot reuse {point['group']} third round with token anomalies")
            audits.append(dict(group=point["group"], round=number, mismatches=audit["mismatches"]))
            for row in audit["requests"]:
                if row["id"] in counts and counts[row["id"]] != row["input_tokens"]:
                    raise ValueError(f"Inconsistent input token count for {row['id']}")
                counts[row["id"]] = row["input_tokens"]
            for path in (log_path, requests_path):
                sources[str(path.relative_to(baseline_dir)).replace("\\", "/")] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    if set(counts) != set(by_id):
        raise ValueError("Baseline must cover the catalog to select replacements by observed full input length")
    manifest = dict(
        version=2,
        rounds=3,
        selected_round=3,
        require_token_audit=True,
        selection=(
            "Keep original slots; replace risks within language/short-long strata by nearest observed full input "
            "token count, then chars, then ID. No performance-based selection."
        ),
        catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        baseline_manifest_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        baseline_git_head=(baseline_dir / "git_head.txt").read_text().strip(),
        tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        excluded_from_new_runs=EXCLUDED,
        observed_input_tokens=counts,
        baseline_audits=audits,
        baseline_source_sha256=sources,
        points=[],
    )
    output_dir.mkdir(parents=True)
    (output_dir / ".gitattributes").write_text("*.jsonl text eol=lf\n", encoding="utf-8", newline="\n")
    pd_batches = {}
    # Baseline order is group -> PD, NPD12, NPD16.
    ordered = sorted(
        baseline["points"],
        key=lambda p: (
            ("short", "long", "mixed").index(p["group"]),
            0 if p["deployment"] == "pd" else p["concurrency"],
        ),
    )
    for point in ordered:
        deployment, group, concurrency = point["deployment"], point["group"], point["concurrency"]
        reuse = deployment == "pd" and group in ("short", "mixed")
        revised = dict(
            deployment=deployment,
            group=group,
            concurrency=concurrency,
            action="reuse_run01" if reuse else "run",
            batches=[],
        )
        seen, previous = set(), set()
        for batch in point["batches"]:
            number = batch["round"]
            original = [by_id[pid] for pid in batch["ids"]]
            if reuse:
                rows, changes = original, []
            else:
                pool = catalog if deployment == "pd" else pd_batches[(group, number)]
                rows, changes = replace_rows(original, pool, counts)
            ids = {r["id"] for r in rows}
            if len(rows) != concurrency or len(ids) != concurrency:
                raise ValueError("Wrong batch size or duplicate IDs")
            for (name, language), count in quotas(group, concurrency, number).items():
                if sum((r["group"], r["language"]) == (name, language) for r in rows) != count:
                    raise ValueError("Replacement broke a stratum quota")
            if deployment == "pd":
                pd_batches[(group, number)] = rows
            else:
                if not ids <= {r["id"] for r in pd_batches[(group, number)]}:
                    raise ValueError("NPD must remain a subset of its corresponding PD batch")
            path = batch["path"]
            target = output_dir / path
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n"
            )
            revised["batches"].append(
                dict(
                    round=number,
                    path=path,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    ids=[r["id"] for r in rows],
                    replacements=changes,
                    original_ids=batch["ids"],
                    original_batch_sha256=batch["sha256"],
                    reused_from_earlier_rounds_in_this_point=len(ids & seen),
                    changed_from_previous_round=len(ids - previous) if number > 1 else None,
                    strata=describe(rows),
                    full_input_tokens=token_stats(rows, counts),
                    pd_parent_full_input_tokens=token_stats(pd_batches[(group, number)], counts),
                )
            )
            seen.update(ids)
            previous = ids
        manifest["points"].append(revised)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Original complete run01/pd directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "schedule_v2")
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    args = parser.parse_args()
    result = build_revision(args.baseline_dir, args.output_dir, args.tokenizer_json)
    print(
        f"Frozen revision: {args.output_dir.resolve()}; {len(result['points'])} points "
        "(PD short/mixed are references only)"
    )

"""One or three N=C t2i probe batches against an already-started server.

Checks request completion and PNG dimensions, not GPU headroom or AR timing.
Uses Python standard library and curl. No model/server configuration is changed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import shutil
import struct
import subprocess
import threading
import time
import unicodedata
from pathlib import Path


def normalized_prompt(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def make_batch(input_dir: Path, group: str, concurrency: int, model: str):
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if group == "mixed" and concurrency % 2:
        raise ValueError("Exactly 50% short/long requires even concurrency")
    pools = {
        name: [json.loads(line) for line in (input_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        for name in ("short", "long")
    }
    if not all(pools.values()):
        raise ValueError("Input pools must not be empty")
    for name in ("short", "long") if group == "mixed" else (group,):
        required = concurrency // 2 if group == "mixed" else concurrency
        if len(pools[name]) < required:
            raise ValueError(f"Need {required} unique {name} prompts, have {len(pools[name])}; repetition is forbidden")
        prompts = [normalized_prompt(row["prompt"]) for row in pools[name]]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Duplicate prompt in {name} pool (ignoring case/whitespace/Unicode variants)")
    batch = []
    for index in range(concurrency):
        name = ("short" if index % 2 == 0 else "long") if group == "mixed" else group
        # Interleaved mixed requests use matching scenes for their short/long pair.
        source_index = index // 2 if group == "mixed" else index
        body = dict(pools[name][source_index])
        width, height = body.pop("width"), body.pop("height")
        if (width, height, body.get("size"), body.get("n")) != (1024, 1024, "1024x1024", 1):
            raise ValueError("This experiment requires size=1024x1024, n=1")
        body.update(model=model, response_format="b64_json")
        batch.append(
            {
                "row": index + 1,
                "group": name,
                "source_row": source_index + 1,
                "prompt_sha256": hashlib.sha256(body["prompt"].encode()).hexdigest(),
                "body": body,
            }
        )
    if len({normalized_prompt(item["body"]["prompt"]) for item in batch}) != len(batch):
        raise ValueError("Duplicate prompt across selected short/long inputs")
    return batch


def send_one(url, item, timeout, barrier, curl, request_path, response_path):
    # No shell interpolation; prompt bytes come from a UTF-8 JSON file.
    command = [
        curl,
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout),
        "--header",
        "Content-Type: application/json",
        "--data-binary",
        f"@{request_path}",
        "--output",
        str(response_path),
        "--write-out",
        "%{http_code}",
        url,
    ]
    barrier.wait(timeout=timeout)
    start = time.perf_counter()
    record = {key: value for key, value in item.items() if key != "body"}
    record.update(
        success=False,
        dimensions=[],
        seed=item["body"]["seed"],
        response_file=str(response_path),
        request_file=str(request_path),
    )
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout + 5)
        record["response_latency_s"] = time.perf_counter() - start
        record["curl_exit_code"] = completed.returncode
        record["http_status"] = int(completed.stdout.strip() or b"0")
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[:2000])
        raw = response_path.read_bytes()
        if not 200 <= record["http_status"] < 300:
            raise RuntimeError(f"HTTP {record['http_status']}: {raw.decode('utf-8', errors='replace')[:2000]}")
        result = json.loads(raw)
        images = result.get("data", [])
        if len(images) != 1:
            raise ValueError(f"Expected one image, received {len(images)}")
        for entry in images:
            image = base64.b64decode(entry["b64_json"], validate=True)
            if image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR" or len(image) < 24:
                raise ValueError("Expected a PNG with an IHDR size header")
            width, height = struct.unpack(">II", image[16:24])
            record["dimensions"].append([width, height])
            if (width, height) != (1024, 1024):
                raise ValueError(f"Actual image is {width}x{height}, expected 1024x1024")
        record["success"] = True
        # Preserve any exposed metrics, but never infer missing AR metrics.
        record["server_metrics"] = result.get("metrics")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record.setdefault("response_latency_s", time.perf_counter() - start)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:9000")
    parser.add_argument("--model", default="/new_data/HunyuanImage-3.0-Instruct/")
    parser.add_argument("--group", choices=["short", "long", "mixed"], required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument(
        "--stage-max-num-seqs",
        type=int,
        nargs="+",
        metavar="N",
        help=(
            "Declare stage0/1 (non-PD) or stage0/1/2 (PD) limits; records them and appends a directory tag; "
            "does not configure the server"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        choices=(1, 3),
        default=1,
        help="One feasibility batch by default; 3 reuses identical inputs and selects the last round",
    )
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).with_name("inputs"))
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    parser.add_argument("--timeout", type=float, default=180, help="curl total transfer timeout in seconds")
    parser.add_argument("--prepare-only", action="store_true", help="Save exact requests without contacting the server")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.stage_max_num_seqs is not None and (
        len(args.stage_max_num_seqs) not in (2, 3) or any(n < 1 for n in args.stage_max_num_seqs)
    ):
        parser.error("stage-max-num-seqs requires two (non-PD) or three (PD) positive integers in stage order")
    curl = shutil.which("curl")
    if not args.prepare_only and not curl:
        parser.error("curl executable is required (same --max-time semantics as the reference request)")
    batch = make_batch(args.input_dir, args.group, args.concurrency, args.model)
    stage_limits = None
    config_tag = None
    if args.stage_max_num_seqs is not None:
        stage_limits = {str(stage): n for stage, n in enumerate(args.stage_max_num_seqs)}
        config_tag = "_".join(f"s{stage}-{n}" for stage, n in stage_limits.items())
        if not args.output_dir.name.endswith("_" + config_tag):
            args.output_dir = args.output_dir.with_name(args.output_dir.name + "_" + config_tag)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    request_dir = args.output_dir / "request_bodies"
    request_dir.mkdir()
    for item in batch:
        (request_dir / f"{item['row']:04d}.json").write_text(
            json.dumps(item["body"], ensure_ascii=False), encoding="utf-8"
        )
    # Pin benchmark dimensions too: its backend derives size from width/height.
    # Pass --num-prompts C; CustomDataset otherwise may cycle through this file.
    (args.output_dir / "batch.jsonl").write_text(
        "".join(json.dumps(dict(item["body"], width=1024, height=1024), ensure_ascii=False) + "\n" for item in batch),
        encoding="utf-8",
    )
    (args.output_dir / "requests.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    plan = dict(
        group=args.group,
        concurrency=args.concurrency,
        num_requests_per_round=len(batch),
        rounds=args.rounds,
        selected_round=args.rounds,
        identical_requests_each_round=True,
        timeout_s=args.timeout,
        prefix_caching="must remain enabled on AR/P/D; verify server configuration, not enforced by HTTP client",
        server_config_declaration={
            "max_num_seqs_by_stage": stage_limits,
            "source": "client CLI declaration" if stage_limits is not None else "not provided",
            "verified_against_running_server": False,
        },
        config_tag=config_tag,
        output_dir=str(args.output_dir),
        endpoint=args.base_url.rstrip("/") + "/v1/images/generations",
        scope="t2i completion/size check only; AR and device telemetry require separate collection",
        unique_prompts=len({row["prompt_sha256"] for row in batch}),
        executed=False,
    )
    (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output directory: {args.output_dir}", flush=True)
    if stage_limits is not None:
        print(f"declared max_num_seqs by stage: {stage_limits} (not verified against server)", flush=True)
    if args.prepare_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    plan.update(executed=True, execution_status="running")
    (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    final = None
    for round_number in range(1, args.rounds + 1):
        response_dir = args.output_dir / f"round{round_number}_responses"
        response_dir.mkdir()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            barrier = threading.Barrier(len(batch) + 1)
            futures = [
                pool.submit(
                    send_one,
                    plan["endpoint"],
                    item,
                    args.timeout,
                    barrier,
                    curl,
                    request_dir / f"{item['row']:04d}.json",
                    response_dir / f"{item['row']:04d}.json",
                )
                for item in batch
            ]
            start = time.perf_counter()
            barrier.wait(timeout=args.timeout)
            records = [future.result() for future in futures]
        wall_s = time.perf_counter() - start
        success_count = sum(row["success"] for row in records)
        final = dict(
            round=round_number,
            success_count=success_count,
            request_count=len(batch),
            client_wall_s=wall_s,
            valid=success_count == len(batch),
            records=records,
        )
        (args.output_dir / f"round{round_number}.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"round {round_number}: {success_count}/{len(batch)} passed, client wall={wall_s:.3f}s", flush=True)
        if not final["valid"]:
            plan.update(execution_status="failed", failed_round=round_number)
            (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit("Batch failed; inspect saved round data and server logs. No probe result selected.")
    selected = dict(
        selected_round=args.rounds,
        valid=True,
        success_count=final["success_count"],
        client_wall_s=final["client_wall_s"],
        note="Client wall includes response validation; use instrumented benchmark for formal throughput.",
    )
    (args.output_dir / "selected_result.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plan.update(execution_status="completed")
    (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

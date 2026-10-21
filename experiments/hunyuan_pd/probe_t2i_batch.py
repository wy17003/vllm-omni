"""One or three N=C t2i batches against an already-started server.

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
from datetime import datetime, timezone
from pathlib import Path

from audit_tokens import audit_tokens, load_tokenizer, tokenizer_metadata
from summarize_metrics import parse_stage_metrics, summary


def load_frozen_batch(path, group, concurrency, model):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != concurrency:
        raise ValueError(f"{path}: expected exactly {concurrency} requests")
    batch = []
    for i, row in enumerate(rows, 1):
        body = dict(row["request"])
        expected = dict(
            width=1024,
            height=1024,
            size="1024x1024",
            n=1,
            num_inference_steps=50,
            guidance_scale=0.0,
            seed=42,
            bot_task="think",
            use_system_prompt="en_unified",
        )
        if any(body.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{path}, row {i}: unexpected experiment parameters")
        digest = hashlib.sha256(body["prompt"].encode()).hexdigest()
        if digest != row["prompt_sha256"]:
            raise ValueError(f"{path}, row {i}: prompt hash mismatch")
        if row["group"] not in (("short", "long") if group == "mixed" else (group,)):
            raise ValueError(f"{path}, row {i}: wrong group")
        body.pop("width")
        body.pop("height")
        body.update(model=model, response_format="b64_json")
        batch.append(
            dict(row=i, id=row["id"], group=row["group"], language=row["language"], prompt_sha256=digest, body=body)
        )
    if len({normalized_prompt(item["body"]["prompt"]) for item in batch}) != concurrency:
        raise ValueError(f"{path}: duplicate prompts")
    if group == "mixed" and sum(item["group"] == "short" for item in batch) * 2 != concurrency:
        raise ValueError(f"{path}: mixed batch must contain exactly half short prompts")
    return batch


def save_batch(directory, batch):
    directory.mkdir(parents=True, exist_ok=False)
    request_dir = directory / "request_bodies"
    request_dir.mkdir()
    for item in batch:
        (request_dir / f"{item['row']:04d}.json").write_text(
            json.dumps(item["body"], ensure_ascii=False), encoding="utf-8"
        )
    (directory / "batch.jsonl").write_text(
        "".join(json.dumps(dict(item["body"], width=1024, height=1024), ensure_ascii=False) + "\n" for item in batch),
        encoding="utf-8",
    )
    (directory / "requests.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    return request_dir


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
    parser.add_argument("--batch-files", type=Path, nargs="+", help="One frozen catalog JSONL per round")
    parser.add_argument("--deployment", choices=("pd", "npd"), help="Stage mapping for existing AR log metrics")
    parser.add_argument("--server-log", type=Path, help="Local, append-only server log; requires exclusive traffic")
    parser.add_argument("--tokenizer-json", type=Path, help="Audit logged AR tokens after each batch; fail on mismatch")
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
        help="Select last round; batch-files supplies rotating inputs, otherwise reuse the same batch",
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
    if args.tokenizer_json and not args.server_log and not args.prepare_only:
        parser.error("--tokenizer-json requires --server-log for execution")
    tokenizer = load_tokenizer(args.tokenizer_json) if args.tokenizer_json else None
    if args.server_log and (not args.server_log.is_file() or not args.deployment):
        parser.error("server-log must exist and requires --deployment")
    if args.batch_files and len(args.batch_files) != args.rounds:
        parser.error("Provide exactly one batch-file per round")
    if args.stage_max_num_seqs is not None and (
        len(args.stage_max_num_seqs) not in (2, 3) or any(n < 1 for n in args.stage_max_num_seqs)
    ):
        parser.error("stage-max-num-seqs requires two (non-PD) or three (PD) positive integers in stage order")
    curl = shutil.which("curl")
    if not args.prepare_only and not curl:
        parser.error("curl executable is required (same --max-time semantics as the reference request)")
    if args.batch_files:
        batches = [load_frozen_batch(p, args.group, args.concurrency, args.model) for p in args.batch_files]
    else:
        batch = make_batch(args.input_dir, args.group, args.concurrency, args.model)
        batches = [batch] * args.rounds
    stage_limits = None
    config_tag = None
    if args.stage_max_num_seqs is not None:
        stage_limits = {str(stage): n for stage, n in enumerate(args.stage_max_num_seqs)}
        config_tag = "_".join(f"s{stage}-{n}" for stage, n in stage_limits.items())
        if not args.output_dir.name.endswith("_" + config_tag):
            args.output_dir = args.output_dir.with_name(args.output_dir.name + "_" + config_tag)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    request_dirs = [save_batch(args.output_dir / f"round{r}_inputs", batch) for r, batch in enumerate(batches, 1)]
    plan = dict(
        group=args.group,
        concurrency=args.concurrency,
        num_requests_per_round=args.concurrency,
        rounds=args.rounds,
        selected_round=args.rounds,
        identical_requests_each_round=all(b == batches[0] for b in batches),
        batch_files=[str(p.resolve()) for p in args.batch_files] if args.batch_files else None,
        batch_file_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.batch_files]
        if args.batch_files
        else None,
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
        scope="Full t2i; client E2E and optional native stage TTFT/TPOT from existing logs",
        deployment=args.deployment,
        server_log=str(args.server_log.resolve()) if args.server_log else None,
        unique_prompts_per_round=[len({row["prompt_sha256"] for row in batch}) for batch in batches],
        executed=False,
        token_audit_tokenizer=tokenizer_metadata(args.tokenizer_json) if args.tokenizer_json else None,
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
        batch = batches[round_number - 1]
        request_dir = request_dirs[round_number - 1]
        response_dir = args.output_dir / f"round{round_number}_responses"
        response_dir.mkdir()
        log_start = args.server_log.stat() if args.server_log else None
        started_at = datetime.now(timezone.utc).isoformat()
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
        ended_at = datetime.now(timezone.utc).isoformat()
        success_count = sum(row["success"] for row in records)
        ar_metrics = None
        token_audit = None
        log_text = ""
        if args.server_log:
            log_end = args.server_log.stat()
            if log_end.st_ino != log_start.st_ino or log_end.st_size < log_start.st_size:
                ar_metrics = {"status": "N/A: server log rotated or truncated"}
            else:
                with args.server_log.open("rb") as stream:
                    stream.seek(log_start.st_size)
                    log_bytes = stream.read(log_end.st_size - log_start.st_size)
                (args.output_dir / f"round{round_number}_server.log").write_bytes(log_bytes)
                log_text = log_bytes.decode("utf-8", errors="replace")
                ar_metrics = parse_stage_metrics(log_text, args.deployment, len(batch))
                ar_metrics["log_byte_interval"] = [log_start.st_size, log_end.st_size]
        if tokenizer is not None:
            token_audit = audit_tokens(log_text, batch, tokenizer)
        final = dict(
            round=round_number,
            success_count=success_count,
            request_count=len(batch),
            client_wall_s=wall_s,
            valid=success_count == len(batch) and (token_audit is None or token_audit["valid"]),
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            e2e_s=summary([row["response_latency_s"] for row in records if row["success"]]),
            ar_metrics=ar_metrics,
            records=records,
            token_audit=token_audit,
        )
        (args.output_dir / f"round{round_number}.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"round {round_number}: {success_count}/{len(batch)} passed, client wall={wall_s:.3f}s", flush=True)
        print(json.dumps({"e2e_s": final["e2e_s"], "ar_metrics": ar_metrics}, ensure_ascii=False), flush=True)
        if not final["valid"]:
            plan.update(execution_status="failed", failed_round=round_number)
            (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            raise SystemExit(
                "Batch HTTP/image validation or token audit failed; "
                "inspect saved round data and logs. No result selected."
            )
    selected = dict(
        selected_round=args.rounds,
        valid=True,
        success_count=final["success_count"],
        client_wall_s=final["client_wall_s"],
        e2e_s=final["e2e_s"],
        ar_metrics=final["ar_metrics"],
        token_audit=final["token_audit"],
        note=(
            "E2E is per-request curl latency including response write, excluding PNG validation. "
            "AR metrics are stage-local."
        ),
    )
    (args.output_dir / "selected_result.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plan.update(execution_status="completed")
    (args.output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

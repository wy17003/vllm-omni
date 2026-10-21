"""Run run03 single/waves once each against an already-started local server."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

from audit_tokens import audit_tokens, load_tokenizer, tokenizer_metadata
from prepare_run03 import write_json
from probe_t2i_batch import load_frozen_batch, save_batch, send_one
from summarize_ar_trace import summarize_trace
from summarize_metrics import parse_stage_metrics, summary

ROOT = Path(__file__).resolve().parent


class WaveGate:
    """Release all threads before independently waiting for their wave deadline."""

    def __init__(self, barrier, origin, offset):
        self.barrier, self.origin, self.offset = barrier, origin, offset

    def wait(self, timeout):
        self.barrier.wait(timeout=timeout)
        remaining = self.origin["mono_ns"] / 1e9 + self.offset - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)  # Dedicated client worker, never a server/event-loop thread.


def execute_waves(batch, request_dir, response_dir, endpoint, timeout, curl):
    origin = {}
    barrier = threading.Barrier(len(batch) + 1, action=lambda: origin.update(mono_ns=time.monotonic_ns()))
    # Capacity covers BOTH waves even while wave 1 waits for serialized DiT.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
        futures = [
            pool.submit(
                send_one,
                endpoint,
                item,
                timeout,
                WaveGate(barrier, origin, item["scheduled_offset_s"]),
                curl,
                request_dir / f"{item['row']:04d}.json",
                response_dir / f"{item['row']:04d}.json",
            )
            for item in batch
        ]
        barrier.wait(timeout=timeout)
        records = [future.result() for future in futures]
    end = time.monotonic_ns()
    for record in records:
        record["actual_offset_s"] = (record["submitted_mono_ns"] - origin["mono_ns"]) / 1e9
        record["dispatch_lateness_s"] = record["actual_offset_s"] - record["scheduled_offset_s"]
    return records, origin["mono_ns"], (end - origin["mono_ns"]) / 1e9


def load_schedule(directory, deployment, model, tokenizer_path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 3 or manifest["rounds"] != 1 or manifest["max_num_batched_tokens"] != 8192:
        raise ValueError("Expected frozen run03 schedule, one run, token budget 8192")
    if manifest["tokenizer"]["sha256"] != tokenizer_metadata(tokenizer_path)["sha256"]:
        raise ValueError("Tokenizer differs from frozen schedule")
    points = [p for p in manifest["points"] if p["deployment"] == deployment]
    if [p["workload"] for p in points] != ["single", "waves"]:
        raise ValueError("Expected exactly single then waves")
    seen = set()
    loaded = []
    expected_count = 48 if deployment == "pd" else 24
    for point in points:
        path = (directory / point["path"]).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("Input path escapes schedule directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != point["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
        batch = load_frozen_batch(path, "long", expected_count, model)
        frozen = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for index, (item, row) in enumerate(zip(batch, frozen)):
            wave = 1 if point["workload"] == "single" or index < expected_count // 2 else 2
            offset = 0.0 if wave == 1 else manifest["wave_delay_s"]
            if row["wave_id"] != wave or row["scheduled_offset_s"] != offset:
                raise ValueError("Wave assignment differs from frozen plan")
            if not math.isfinite(offset) or offset < 0 or (wave == 2 and offset == 0):
                raise ValueError("Invalid wave deadline")
            item.update(wave_id=wave, scheduled_offset_s=offset, expected_input_tokens=row["full_input_tokens"])
            if item["prompt_sha256"] in seen:
                raise ValueError("Prompt repeats across run03 points")
            seen.add(item["prompt_sha256"])
        loaded.append((point, batch))
    return manifest, loaded


def save_provenance(destination, manifest, deployment):
    write_json(destination / "schedule_manifest.json", manifest)
    shutil.copyfile(ROOT / f"configs/run03_{deployment}.yaml", destination / "declared_config.yaml")
    repo = ROOT.parents[1]
    for name, cmd in (
        ("git_head.txt", ["git", "rev-parse", "HEAD"]),
        ("git_status.txt", ["git", "status", "--short"]),
        ("working_tree.patch", ["git", "diff", "HEAD"]),
    ):
        result = subprocess.run(cmd, cwd=repo, capture_output=True, check=True)
        (destination / name).write_bytes(result.stdout)
    sources = list(ROOT.glob("*.py")) + [
        repo / p
        for p in (
            "vllm_omni/utils/hunyuan_ar_trace.py",
            "vllm_omni/core/sched/omni_ar_scheduler.py",
            "vllm_omni/engine/orchestrator.py",
            "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py",
        )
    ]
    write_json(
        destination / "source_sha256.json",
        {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    versions = {}
    for package in ("vllm", "vllm-omni", "vllm-ascend", "torch", "tokenizers", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(destination / "package_versions.json", versions)


def run(args):
    manifest, points = load_schedule(args.schedule_dir, args.deployment, args.model, args.tokenizer_json)
    tokenizer = load_tokenizer(args.tokenizer_json)
    curl = shutil.which("curl")
    if not args.prepare_only and (not curl or not args.server_log or not args.server_log.is_file()):
        raise ValueError("Execution needs curl and an existing local --server-log")
    destination = args.output_root.resolve() / args.deployment
    destination.mkdir(parents=True, exist_ok=False)
    save_provenance(destination, manifest, args.deployment)
    endpoint = args.base_url.rstrip("/") + "/v1/images/generations"
    results = []
    if not args.prepare_only:
        # Same independent prompt/parameters on both deployments; warms the common system prefix.
        warm = dict(points[0][1][0])
        warm.update(id="run03_warmup", row=1, body=dict(warm["body"]))
        warm["body"]["prompt"] = (
            "A quiet kitchen with a sleeping orange cat beside a woven basket, "
            "natural window light, square composition."
        )
        warm["prompt_sha256"] = hashlib.sha256(warm["body"]["prompt"].encode()).hexdigest()
        request_dir = save_batch(destination / "warmup_inputs", [warm])
        record = send_one(
            endpoint,
            warm,
            args.timeout,
            threading.Barrier(1),
            curl,
            request_dir / "0001.json",
            destination / "warmup_response.json",
        )
        write_json(destination / "warmup_result.json", record)
        if not record["success"]:
            raise SystemExit("Warmup failed; formal runs not started")
    for point, batch in points:
        tag = "s0-64_s1-64_s2-1" if args.deployment == "pd" else "s0-64_s1-1"
        folder = destination / f"{args.deployment}-{point['workload']}-n{len(batch)}_b8192_pc-on_step-off_{tag}"
        folder.mkdir()
        request_dir = save_batch(folder / "round1_inputs", batch)
        responses = folder / "round1_responses"
        responses.mkdir()
        plan = dict(
            point,
            selected_round=1,
            rounds=1,
            max_num_batched_tokens=8192,
            wave_delay_s=manifest["wave_delay_s"],
            timeout_s=args.timeout,
            endpoint=endpoint,
            max_inflight_http=len(batch),
            trace_clock="single-host monotonic_ns",
            deployment=args.deployment,
            executed=not args.prepare_only,
        )
        write_json(folder / "plan.json", plan)
        if args.prepare_only:
            continue
        before = args.server_log.stat()
        print(f"Running {args.deployment} {point['workload']}: waves={point['wave_sizes']}", flush=True)
        records, origin, wall = execute_waves(batch, request_dir, responses, endpoint, args.timeout, curl)
        # Save transport observations before parsing logs; failures never erase a completed run.
        result = dict(
            round=1,
            request_count=len(batch),
            success_count=sum(r["success"] for r in records),
            client_wall_s=wall,
            origin_mono_ns=origin,
            records=records,
            valid=False,
            e2e_s=summary([r["response_latency_s"] for r in records if r["success"]]),
        )
        write_json(folder / "round1.json", result)
        after = args.server_log.stat()
        if before.st_ino != after.st_ino or after.st_size < before.st_size:
            raise RuntimeError("Server log rotated or truncated; transport records saved")
        with args.server_log.open("rb") as stream:
            stream.seek(before.st_size)
            raw = stream.read(after.st_size - before.st_size)
        (folder / "round1_server.log").write_bytes(raw)
        log = raw.decode("utf-8", errors="replace")
        audit = audit_tokens(log, batch, tokenizer)
        metrics = parse_stage_metrics(log, args.deployment, len(batch))
        trace = summarize_trace(log, batch, records, audit, args.deployment)
        result.update(
            token_audit=audit, ar_metrics=metrics, ar_trace=trace, log_byte_interval=[before.st_size, after.st_size]
        )
        result["valid"] = result["success_count"] == len(batch) and audit["valid"]
        result["analysis_eligible"] = (
            result["valid"]
            and trace["complete"]
            and all(metrics[k]["status"] == "complete" for k in ("ttft_ms", "tpot_ms"))
        )
        write_json(folder / "round1.json", result)
        results.append(
            dict(
                workload=point["workload"],
                output_dir=str(folder),
                valid=result["valid"],
                analysis_eligible=result["analysis_eligible"],
                ar_trace=trace,
            )
        )
        write_json(destination / "summary.json", dict(deployment=args.deployment, points=results))
        if not result["analysis_eligible"]:
            raise SystemExit("Run failed transport/token/trace validation; inspect saved data. No automatic retry.")
        write_json(folder / "selected_result.json", dict(result, selected_round=1))
        print(f"Completed: AR window={trace['ar_window_s']:.3f}s; warnings={len(audit['warnings'])}", flush=True)
    print(f"Output: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", choices=("pd", "npd"), required=True)
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "schedule_run03")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--base-url", default="http://localhost:9000")
    parser.add_argument("--model", default="/new_data/HunyuanImage-3.0-Instruct/")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    run(args)

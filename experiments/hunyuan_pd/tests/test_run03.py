"""CPU regressions for independent waves, frozen inputs and conservative AR tracing."""

import argparse
import base64
import copy
import hashlib
import json
import shutil
import statistics
import struct
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe_t2i_batch import load_frozen_batch, save_batch  # noqa: E402
from run_ar_experiment import execute_waves, load_schedule, run  # noqa: E402
from summarize_ar_trace import summarize_trace  # noqa: E402

SCHEDULE = ROOT / "schedule_run03"


def trace_fixture(deployment):
    events, batch, records, requests = [], [], [], []

    def emit(kind, stage, seconds, **fields):
        events.append(dict(event=kind, stage=stage, mono_ns=int((seconds + 100) * 1e9), host="node", **fields))

    ar_stage = 1 if deployment == "pd" else 0
    for wave, offset in ((1, 0), (2, 5)):
        rid = f"request{wave}"
        internal = f"internal{wave}"
        batch.append(dict(id=rid, wave_id=wave, expected_input_tokens=1900))
        records.append(
            dict(
                id=rid,
                wave_id=wave,
                submitted_mono_ns=int((100 + offset) * 1e9),
                dispatch_lateness_s=0.0,
                client_host="node",
            )
        )
        requests.append(dict(id=rid, server_request_id=rid, input_tokens=1900, output_tokens=4 + ar_stage))
        emit("request_start", 0, offset + 0.1, request_id=rid)
        for stage in range(ar_stage + 1):
            emit(
                "admit",
                stage,
                offset + 0.15 + stage,
                request_id=internal,
                external_request_id=rid,
                input_tokens=1900,
                max_num_seqs=64,
                max_num_batched_tokens=8192,
                prefix_caching=True,
            )
            emit("forward_submit", stage, offset + 0.2 + stage, scheduled_tokens={internal: 1900 if stage == 0 else 1})
        if deployment == "pd":
            emit(
                "outputs",
                0,
                offset + 1.0,
                requests=[
                    dict(request_id=internal, delta_tokens=1, output_tokens=1, cached_tokens=1200, finished=True)
                ],
            )
            emit("stage_done", 0, offset + 1.1, request_id=rid)
            emit("kv_ready", 1, offset + 1.18, request_id=internal)
        for index, timestamp in enumerate((2, 4, 6, 8)):
            emit(
                "outputs",
                ar_stage,
                offset + timestamp,
                requests=[
                    dict(
                        request_id=internal,
                        delta_tokens=1 + ar_stage if index == 0 else 1,
                        output_tokens=index + 1 + ar_stage,
                        cached_tokens=1200 if index == 0 else None,
                        finished=index == 3,
                    )
                ],
            )
        emit("stage_done", ar_stage, offset + 9, request_id=rid)
    return events, batch, records, dict(requests=requests)


def summarize_fixture(fixture, deployment):
    events, *args = fixture
    # Deliberately reverse line order: independent process logging may be reordered.
    log = "\n".join("[HY3_AR_TRACE] " + json.dumps(e) for e in reversed(events))
    return summarize_trace(log, *args, deployment)


class Run03Tests(unittest.TestCase):
    def test_frozen_schedule_balanced_unique_matched_and_lf(self):
        manifest = json.loads((SCHEDULE / "manifest.json").read_text(encoding="utf-8"))
        groups = {}
        for point in manifest["points"]:
            raw = (SCHEDULE / point["path"]).read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), point["sha256"])
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            groups[(point["deployment"], point["workload"])] = rows
            self.assertEqual(len(rows), point["requests"])
            self.assertEqual(len({r["prompt_sha256"] for r in rows}), len(rows))
            for wave in sorted({r["wave_id"] for r in rows}):
                subset = [r for r in rows if r["wave_id"] == wave]
                self.assertEqual(sum(r["language"] == "en" for r in subset), len(subset) // 4)
                self.assertTrue(all(r["scheduled_offset_s"] == (0 if wave == 1 else 5) for r in subset))
        pd_rows = groups[("pd", "single")] + groups[("pd", "waves")]
        self.assertEqual(len({r["prompt_sha256"] for r in pd_rows}), 96)
        self.assertEqual(
            [min(r["full_input_tokens"] for r in pd_rows), max(r["full_input_tokens"] for r in pd_rows)],
            manifest["input_tokens_range"],
        )
        for workload in ("single", "waves"):
            parent, subset = groups[("pd", workload)], groups[("npd", workload)]
            self.assertLessEqual({r["id"] for r in subset}, {r["id"] for r in parent})
            for wave in {r["wave_id"] for r in parent}:
                for lang in ("zh", "en"):
                    means = [
                        statistics.mean(
                            r["full_input_tokens"] for r in rows if r["wave_id"] == wave and r["language"] == lang
                        )
                        for rows in (parent, subset)
                    ]
                    self.assertLess(abs(means[0] - means[1]), 10)

    def test_prepare_only_and_changed_input_rejection(self):
        manifest = json.loads((SCHEDULE / "manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch("run_ar_experiment.tokenizer_metadata", return_value=manifest["tokenizer"]),
                patch("run_ar_experiment.load_tokenizer"),
                patch("run_ar_experiment.send_one") as send,
            ):
                for deployment in ("pd", "npd"):
                    args = argparse.Namespace(
                        schedule_dir=SCHEDULE,
                        deployment=deployment,
                        tokenizer_json=root / "unused.json",
                        model="model",
                        prepare_only=True,
                        output_root=root / "out",
                        server_log=None,
                        base_url="http://localhost:9000",
                        timeout=3600,
                    )
                    run(args)
                    plans = list((args.output_root / deployment).glob("*/plan.json"))
                    self.assertEqual(len(plans), 2)
                    for path in plans:
                        plan = json.loads(path.read_text())
                        self.assertFalse(plan["executed"])
                        self.assertEqual(plan["rounds"], 1)
                        self.assertEqual(plan["max_num_batched_tokens"], 8192)
                        self.assertEqual(plan["max_inflight_http"], 48 if deployment == "pd" else 24)
                send.assert_not_called()
                shutil.copytree(SCHEDULE, root / "bad")
                with (root / "bad/pd-single.jsonl").open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    load_schedule(root / "bad", "pd", "model", root / "unused.json")

    @unittest.skipUnless(shutil.which("curl"), "curl required")
    def test_wave2_arrives_before_wave1_image_returns(self):
        arrivals, lock, wave2 = {}, threading.Lock(), threading.Event()
        png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1024, 1024)
        payload = json.dumps({"data": [{"b64_json": base64.b64encode(png).decode()}]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                wave = int(body["prompt"])
                with lock:
                    arrivals.setdefault(wave, []).append(time.monotonic_ns())
                    if len(arrivals.get(2, [])) == 2:
                        wave2.set()
                unblocked = wave != 1 or wave2.wait(3)
                self.send_response(200 if unblocked else 500)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                batch = load_frozen_batch(SCHEDULE / "npd-waves.jsonl", "long", 24, "model")[:4]
                for index, item in enumerate(batch):
                    wave = 1 if index < 2 else 2
                    item.update(wave_id=wave, scheduled_offset_s=0 if wave == 1 else 0.2)
                    item["body"]["prompt"] = str(wave)
                request_dir = save_batch(root / "inputs", batch)
                responses = root / "responses"
                responses.mkdir()
                records, _, _ = execute_waves(
                    batch, request_dir, responses, f"http://127.0.0.1:{server.server_port}", 5, shutil.which("curl")
                )
                self.assertTrue(all(r["success"] for r in records), records)
                self.assertLess(max(arrivals[2]), min(r["response_end_mono_ns"] for r in records if r["wave_id"] == 1))
                self.assertTrue(all(r["actual_offset_s"] >= 0.19 for r in records if r["wave_id"] == 2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_pd_and_npd_counts_clock_mapping_and_crossing_itl(self):
        for deployment in ("pd", "npd"):
            result = summarize_fixture(trace_fixture(deployment), deployment)
            self.assertTrue(result["complete"], result["errors"])
            self.assertEqual(result["ar_window_s"], 14)
            self.assertEqual(result["output_tokens"], 10 if deployment == "pd" else 8)
            self.assertEqual(result["requests"][0]["uncached_tokens"], 700)
            overlap = result["two_wave_overlap"]
            self.assertTrue(overlap["target_window_observed"])
            self.assertEqual(overlap["first_wave_decoding_count"], 1)
            self.assertEqual(overlap["first_wave_itl_after_ms"]["n"], 2)
            self.assertEqual(overlap["first_wave_itl_after_ms"]["mean"], 2000)
            if deployment == "pd":
                # The future second wave cannot count as pending P at the first D token.
                self.assertEqual(result["pipeline_overlap"]["pending_prefill_count"], 0)

    def test_incomplete_trace_cannot_report_throughput(self):
        fixture = trace_fixture("pd")
        mutations = []
        for kind in ("kv_ready", "forward_submit", "outputs", "request_start"):
            bad = copy.deepcopy(fixture)
            index = next(
                i
                for i, e in enumerate(bad[0])
                if e["event"] == kind and e["stage"] == (0 if kind == "request_start" else 1)
            )
            del bad[0][index]
            mutations.append(bad)
        for key, value in (("max_num_seqs", 32), ("prefix_caching", False), ("max_num_batched_tokens", 4096)):
            bad = copy.deepcopy(fixture)
            next(e for e in bad[0] if e["event"] == "admit")[key] = value
            mutations.append(bad)
        bad = copy.deepcopy(fixture)
        bad[0][0]["host"] = "another-node"
        mutations.append(bad)
        bad = copy.deepcopy(fixture)
        bad[0].append(dict(mono_ns=0, host="node", stage=0, event="outputs", requests=None))
        mutations.append(bad)
        bad = copy.deepcopy(fixture)
        bad[2][0]["dispatch_lateness_s"] = 1.0
        mutations.append(bad)
        for bad in mutations:
            result = summarize_fixture(bad, "pd")
            self.assertFalse(result["complete"])
            self.assertIsNone(result["ar_window_s"])
            self.assertIsNone(result["output_tokens_per_s"])

    def test_early_prefill_metadata_is_not_a_first_token(self):
        fixture = trace_fixture("npd")
        fixture[0].append(
            dict(
                event="outputs",
                stage=0,
                host="node",
                mono_ns=100_500_000_000,
                requests=[
                    dict(request_id="internal1", delta_tokens=0, output_tokens=0, cached_tokens=1200, finished=False)
                ],
            )
        )
        for event in fixture[0]:
            if event["event"] == "outputs" and event["mono_ns"] == 102_000_000_000:
                event["requests"][0]["cached_tokens"] = None
        result = summarize_fixture(fixture, "npd")
        self.assertTrue(result["complete"], result["errors"])
        row = result["requests"][0]
        self.assertEqual(row["cached_tokens"], 1200)
        self.assertEqual(row["prefill_done_mono_ns"], 102_000_000_000)


if __name__ == "__main__":
    unittest.main()

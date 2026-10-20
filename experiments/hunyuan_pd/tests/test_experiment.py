"""CPU-only experiment regression checks: python -m unittest discover -s this_directory."""

import base64
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_schedule import build_schedule  # noqa: E402
from summarize_metrics import parse_stage_metrics  # noqa: E402


class ExperimentTests(unittest.TestCase):
    def test_frozen_inputs_are_reproducible_unique_balanced_and_matched(self):
        committed = json.loads((ROOT / "schedule/manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = build_schedule(Path(temporary) / "schedule")
        self.assertEqual(committed, rebuilt)
        parents = {p["group"]: p for p in committed["points"] if p["deployment"] == "pd"}
        for point in committed["points"]:
            for index, batch in enumerate(point["batches"]):
                raw = (ROOT / "schedule" / batch["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), batch["sha256"])
                rows = [json.loads(line) for line in raw.decode().splitlines()]
                self.assertEqual(len(rows), point["concurrency"])
                self.assertEqual(len({r["request"]["prompt"] for r in rows}), len(rows))
                self.assertEqual(sum(r["language"] == "en" for r in rows), len(rows) // 4)
                if point["group"] == "mixed":
                    self.assertEqual(sum(r["group"] == "short" for r in rows), len(rows) // 2)
                parent = parents[point["group"]]["batches"][index]
                self.assertLessEqual(set(batch["ids"]), set(parent["ids"]))
                if index:
                    self.assertNotEqual(set(batch["ids"]), set(point["batches"][index - 1]["ids"]))

    def test_metrics_mapping_missing_zeros_and_duplicate_tables(self):
        table = "[StageRequestStats [request_id=x]]\n| Field | 0 | 1 | 2 |\n"
        table += "| vllm_ttft_ms | 1,200.0 | 40 | 0 |\n| vllm_tpot_ms | 0 | 15 | 0 |\n"
        result = parse_stage_metrics(table + table, "pd", 1)
        self.assertEqual(result["ttft_ms"]["mean"], 1200)
        self.assertEqual(result["tpot_ms"]["mean"], 15)
        self.assertIsNone(parse_stage_metrics(table, "npd", 1)["tpot_ms"]["mean"])
        self.assertIsNone(parse_stage_metrics(table, "pd", 2)["ttft_ms"]["mean"])
        self.assertIsNone(parse_stage_metrics("", "pd", 1)["ttft_ms"]["mean"])

    @unittest.skipUnless(shutil.which("curl"), "curl required")
    def test_real_curl_round_rotation_and_failure_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "server.log"
            log.write_text("", encoding="utf-8")
            received = []
            lock = threading.Lock()
            failed = threading.Event()
            # Only an IHDR fixture: the client explicitly checks PNG header dimensions.
            png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1024, 1024)
            response = json.dumps({"data": [{"b64_json": base64.b64encode(png).decode()}]}).encode()

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    with lock:
                        received.append(body)
                        with log.open("a", encoding="utf-8") as stream:
                            stream.write(
                                f"[StageRequestStats [request_id=req-{len(received)}]]\n"
                                "| Field | 0 | 1 | 2 |\n"
                                "| vllm_ttft_ms | 123 | 44 | 0 |\n"
                                "| vllm_tpot_ms | 0 | 12 | 0 |\n"
                            )
                    payload = b'{"error":"fixture failure"}' if failed.is_set() else response
                    self.send_response(500 if failed.is_set() else 200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                batch_files = [ROOT / f"schedule/pd-long-c24/round{r}.jsonl" for r in (1, 2, 3)]
                command = [
                    sys.executable,
                    str(ROOT / "probe_t2i_batch.py"),
                    "--deployment",
                    "pd",
                    "--group",
                    "long",
                    "--concurrency",
                    "24",
                    "--rounds",
                    "3",
                    "--stage-max-num-seqs",
                    "32",
                    "32",
                    "1",
                    "--server-log",
                    str(log),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--timeout",
                    "20",
                    "--batch-files",
                    *map(str, batch_files),
                ]
                result = subprocess.run(command + ["--output-dir", str(root / "good")], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                output = root / "good_s0-32_s1-32_s2-1"
                selected = json.loads((output / "selected_result.json").read_text())
                self.assertEqual(len(received), 72)
                self.assertEqual(selected["selected_round"], 3)
                self.assertEqual(selected["e2e_s"]["n"], 24)
                self.assertEqual(selected["ar_metrics"]["ttft_ms"]["mean"], 123)
                self.assertEqual(selected["ar_metrics"]["tpot_ms"]["mean"], 12)
                for index, path in enumerate(batch_files):
                    expected = {
                        json.loads(line)["request"]["prompt"] for line in path.read_text(encoding="utf-8").splitlines()
                    }
                    self.assertEqual({r["prompt"] for r in received[index * 24 : (index + 1) * 24]}, expected)
                self.assertTrue(all(r["guidance_scale"] == 0 and "width" not in r for r in received))
                failed.set()
                result = subprocess.run(command + ["--output-dir", str(root / "bad")], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                output = root / "bad_s0-32_s1-32_s2-1"
                self.assertTrue((output / "round1.json").exists())
                self.assertFalse((output / "round2.json").exists())
                self.assertFalse((output / "selected_result.json").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()

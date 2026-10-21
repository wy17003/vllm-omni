"""Supplement regression: PD reuse, NPD32 selection, profile guards and dry run."""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_run03_1p5 import prepare  # noqa: E402
from run_ar_experiment import load_schedule, run  # noqa: E402

SOURCE = ROOT / "schedule_run03"
SCHEDULE = ROOT / "schedule_run03_1p5"


class SupplementTests(unittest.TestCase):
    def test_reproducible_superset_balanced_and_pd_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "schedule"
            manifest = prepare(SOURCE, output)
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), (SCHEDULE / path.name).read_bytes())
            for point in manifest["points"]:
                path = output / point["path"]
                raw = path.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), point["sha256"])
                self.assertNotIn(b"\r\n", raw)
                if point["deployment"] == "pd":
                    self.assertEqual(raw, (SOURCE / path.name).read_bytes())
                    continue
                rows = [json.loads(line) for line in raw.decode().splitlines()]
                old = [json.loads(line) for line in (SOURCE / path.name).read_text(encoding="utf-8").splitlines()]
                parent = [
                    json.loads(line)
                    for line in (SOURCE / path.name.replace("npd", "pd")).read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 32)
                self.assertEqual(len({r["prompt_sha256"] for r in rows}), 32)
                by_id = {r["id"]: r for r in rows}
                self.assertTrue(all(by_id[r["id"]] == r for r in old))
                self.assertTrue(all(r in parent for r in rows))
                for wave in {r["wave_id"] for r in rows}:
                    subset = [r for r in rows if r["wave_id"] == wave]
                    self.assertEqual(sum(r["language"] == "en" for r in subset), len(subset) // 4)

    def test_npd32_dry_run_and_reused_pd_not_executed(self):
        manifest = json.loads((SCHEDULE / "manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                schedule_dir=SCHEDULE,
                deployment="npd",
                tokenizer_json=Path(temporary) / "unused",
                model="model",
                prepare_only=True,
                output_root=Path(temporary) / "out",
                server_log=None,
                base_url="http://localhost:9000",
                timeout=3600,
            )
            with (
                patch("run_ar_experiment.tokenizer_metadata", return_value=manifest["tokenizer"]),
                patch("run_ar_experiment.load_tokenizer"),
                patch("run_ar_experiment.send_one") as send,
            ):
                run(args)
                plans = list((args.output_root / "npd").glob("*/plan.json"))
                self.assertEqual(len(plans), 2)
                for path in plans:
                    plan = json.loads(path.read_text())
                    self.assertEqual(plan["max_inflight_http"], 32)
                    self.assertEqual(plan["comparison_profile"], "total_cards_1p5x")
                    self.assertEqual(plan["ar_cards"], 4)
                    self.assertEqual(plan["total_cards"], 8)
                args.deployment, args.prepare_only = "pd", False
                with self.assertRaisesRegex(ValueError, "reuses completed run03 PD"):
                    run(args)
                send.assert_not_called()

    def test_c32_without_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads((SCHEDULE / "manifest.json").read_text(encoding="utf-8"))
            manifest.pop("comparison_profile")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("run_ar_experiment.tokenizer_metadata", return_value=manifest["tokenizer"]):
                with self.assertRaisesRegex(ValueError, "counts differ"):
                    load_schedule(root, "npd", "model", root / "unused")


if __name__ == "__main__":
    unittest.main()

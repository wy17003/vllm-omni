"""Count actual Hunyuan input tokens on the inference host without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local directory containing the experiment model tokenizer")
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("token_calibration.json"))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    for line in args.catalog.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        req = item["request"]
        built = build_prompt_tokens(
            req["prompt"], tokenizer, task="t2i", bot_task=req["bot_task"], sys_type=req["use_system_prompt"]
        )
        rows.append(
            dict(
                id=item["id"],
                pair_id=item["pair_id"],
                group=item["group"],
                language=item["language"],
                prompt_sha256=item["prompt_sha256"],
                user_token_count=len(tokenizer.encode(req["prompt"], add_special_tokens=False)),
                full_prompt_token_count=len(built.token_ids),
            )
        )
    summary = {}
    for group in ("short", "long"):
        values = [row["full_prompt_token_count"] for row in rows if row["group"] == group]
        summary[group] = dict(
            count=len(values),
            min_full_tokens=min(values),
            max_full_tokens=max(values),
            mean_full_tokens=sum(values) / len(values),
            count_ge_2048=sum(v >= 2048 for v in values),
            count_ge_4096=sum(v >= 4096 for v in values),
            count_ge_8192=sum(v >= 8192 for v in values),
        )
    report = dict(
        model=str(Path(args.model).resolve()),
        catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        method="branch build_prompt_tokens; task=t2i, bot_task=think, sys_type=en_unified",
        full_token_ranges_overlap=summary["short"]["max_full_tokens"] >= summary["long"]["min_full_tokens"],
        summary=summary,
        rows=rows,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": summary,
                "full_token_ranges_overlap": report["full_token_ranges_overlap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

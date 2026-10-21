"""Audit the known AR text round-trip defect using existing HY3_EQ logs.

This checks the logged AR sequence, not numerical KV or image equivalence.
No model weights, torch, or inference calls are needed.
"""

import hashlib
import json
import re
from pathlib import Path

TOKEN_AUDIT_POLICY = "warn_equal_length_v1"


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Token audit requires the inference environment's tokenizers package") from exc
    return Tokenizer.from_file(str(path))


def read_ar_events(log_text):
    events = {}
    errors = []
    for line in log_text.splitlines():
        match = re.search(r"\[HY3_EQ\] ar request_id=(\S+)", line)
        if not match:
            continue
        request_id = match[1]
        event = events.setdefault(request_id, {})
        for key in ("prompt_token_ids", "output_token_ids"):
            match = re.search(key + r"=(\[[^\]]*\])", line)
            if not match:
                continue
            try:
                ids = json.loads(match[1])
                if not isinstance(ids, list) or not all(type(i) is int for i in ids):
                    raise ValueError("expected integer IDs")
                if key in event and event[key] != ids:
                    errors.append(f"{request_id}: conflicting {key}")
                event[key] = ids
                if key == "output_token_ids":
                    generated_text = json.JSONDecoder().raw_decode(line.split("ar_generated_text=", 1)[1])[0]
                    if not isinstance(generated_text, str):
                        raise ValueError("expected generated text string")
                    event["text"] = generated_text
            except (ValueError, IndexError, TypeError) as exc:
                errors.append(f"{request_id}: malformed {key}: {exc}")
    return events, errors


def audit_tokens(log_text, requests, tokenizer):
    events, errors = read_ar_events(log_text)
    rows = []
    expected_ids = [r.get("id", str(r["row"])) for r in requests]
    for request_id, event in events.items():
        if "output_token_ids" not in event:
            # PD's prefill request has a separate ID and does not supply CoT.
            continue
        if "prompt_token_ids" not in event or "text" not in event:
            errors.append(f"{request_id}: missing AR prompt IDs or generated text")
            continue
        prompt_text = tokenizer.decode(event["prompt_token_ids"], skip_special_tokens=False)
        matches = [r for r in requests if r["body"]["prompt"] in prompt_text]
        if len(matches) != 1:
            errors.append(f"{request_id}: cannot uniquely map logged prompt to actual request")
            continue
        request = matches[0]
        original = event["output_token_ids"]
        decoded = tokenizer.decode(original, skip_special_tokens=False)
        rebuilt = tokenizer.encode(event["text"], add_special_tokens=False).ids
        equal = decoded == event["text"] and original == rebuilt
        first_difference = next((i for i, (a, b) in enumerate(zip(original, rebuilt)) if a != b), None)
        if first_difference is None and len(original) != len(rebuilt):
            first_difference = min(len(original), len(rebuilt))
        rows.append(
            dict(
                id=request.get("id", str(request["row"])),
                server_request_id=request_id,
                input_tokens=len(event["prompt_token_ids"]),
                output_tokens=len(original),
                reencoded_tokens=len(rebuilt),
                decoded_text_matches_log=decoded == event["text"],
                roundtrip_equal=equal,
                first_different_output_token=first_difference,
            )
        )
    mapped = [r["id"] for r in rows]
    if sorted(mapped) != sorted(expected_ids) or len(set(mapped)) != len(mapped):
        errors.append("AR output coverage does not match the actual batch exactly")
    mismatches = [r for r in rows if not r["roundtrip_equal"]]
    warnings = [r for r in mismatches if r["decoded_text_matches_log"] and r["output_tokens"] == r["reencoded_tokens"]]
    blocking = [
        r for r in mismatches if not r["decoded_text_matches_log"] or r["output_tokens"] != r["reencoded_tokens"]
    ]
    return dict(
        policy=TOKEN_AUDIT_POLICY,
        valid=not errors and not blocking,
        roundtrip_consistent=not errors and not mismatches,
        expected_count=len(requests),
        observed_count=len(rows),
        errors=errors,
        mismatches=mismatches,
        warnings=warnings,
        blocking_mismatches=blocking,
        requests=rows,
        scope=(
            "Equal-length ID differences with matching decoded text are warnings under the experiment policy. "
            "Passing does not prove token, KV numerical, or image equivalence."
        ),
    )


def tokenizer_metadata(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

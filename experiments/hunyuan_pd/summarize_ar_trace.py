"""Summarize opt-in host AR events; never label host intervals as kernel time."""

import json
import math
import re
import statistics
from collections import defaultdict


def describe(values):
    values = sorted(values)
    return dict(
        n=len(values),
        mean=statistics.mean(values) if values else None,
        p50=statistics.median(values) if values else None,
        p95=values[max(0, math.ceil(len(values) * 0.95) - 1)] if values else None,
        max=max(values) if values else None,
    )


def validate_event(event):
    """Reject broken/interleaved lines without discarding the saved transport result."""
    if not isinstance(event, dict) or type(event.get("mono_ns")) is not int:
        raise ValueError("invalid timestamp/event")
    if type(event.get("stage")) is not int or not isinstance(event.get("host"), str):
        raise ValueError("invalid stage/host")
    kind = event.get("event")
    if kind == "forward_submit":
        scheduled = event.get("scheduled_tokens")
        if not isinstance(scheduled, dict) or not all(
            isinstance(rid, str) and type(n) is int and n > 0 for rid, n in scheduled.items()
        ):
            raise ValueError("invalid forward token counts")
    elif kind == "outputs":
        entries = event.get("requests")
        if not isinstance(entries, list) or not all(
            isinstance(row, dict)
            and isinstance(row.get("request_id"), str)
            and type(row.get("delta_tokens")) is int
            and row["delta_tokens"] >= 0
            and type(row.get("output_tokens")) is int
            and row["output_tokens"] >= 0
            and type(row.get("finished")) is bool
            for row in entries
        ):
            raise ValueError("invalid output records")
    elif kind not in ("request_start", "admit", "stage_done", "kv_ready"):
        raise ValueError("unknown event kind")
    elif not isinstance(event.get("request_id"), str):
        raise ValueError("invalid request ID")
    if kind == "admit" and not isinstance(event.get("external_request_id"), str):
        raise ValueError("missing admission ID mapping")


def summarize_trace(log, batch, records, audit, deployment):
    errors = []
    events = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", log).splitlines():
        if "[HY3_AR_TRACE] " not in line:
            continue
        try:
            event, _ = json.JSONDecoder().raw_decode(line.split("[HY3_AR_TRACE] ", 1)[1])
            validate_event(event)
            events.append(event)
        except (ValueError, TypeError) as exc:
            errors.append(f"Malformed trace event: {exc}")
    events.sort(key=lambda e: e["mono_ns"])
    hosts = {e.get("host") for e in events} | {r.get("client_host") for r in records}
    if len(hosts) != 1 or None in hosts:
        errors.append("Trace and client must use one host's monotonic clock")
    if not events:
        errors.append("No HY3_AR_TRACE events: restart server with VLLM_OMNI_HY3_AR_TRACE=1")
    mapping = {r["server_request_id"]: r for r in audit["requests"]}
    by_id = {r["id"]: r for r in batch}
    record_by_id = {r["id"]: r for r in records}
    if (
        len(mapping) != len(batch)
        or len(record_by_id) != len(batch)
        or {r["id"] for r in mapping.values()} != set(by_id)
    ):
        errors.append("Audit/client/request ID coverage differs")
    aliases = {}
    for event in events:
        if event["event"] == "admit":
            key = (event["request_id"], event["stage"])
            external = event["external_request_id"]
            if key in aliases and aliases[key] != external:
                errors.append("Conflicting admission ID mapping")
            aliases[key] = external
    starts, admits, forwards, tokens, dones, ready = (defaultdict(list) for _ in range(6))
    for event in events:
        kind, stage, timestamp = event["event"], event["stage"], event["mono_ns"]
        if kind == "outputs":
            entries = event["requests"]
        elif kind == "forward_submit":
            entries = [dict(request_id=rid, scheduled_tokens=count) for rid, count in event["scheduled_tokens"].items()]
        else:
            entries = [event]
        for row in entries:
            rid = aliases.get((row["request_id"], stage), row["request_id"])
            if rid not in mapping:
                errors.append(f"Unmapped trace request {rid}")
                continue
            key = (rid, stage)
            target = {
                "request_start": starts,
                "admit": admits,
                "forward_submit": forwards,
                "outputs": tokens,
                "stage_done": dones,
                "kv_ready": ready,
            }.get(kind)
            if target is None:
                errors.append(f"Unknown event kind {kind}")
            else:
                target[key].append(dict(row, mono_ns=timestamp))
    ar_stage = 1 if deployment == "pd" else 0
    rows = []
    for rid, audit_row in mapping.items():
        if audit_row["id"] not in by_id or audit_row["id"] not in record_by_id:
            errors.append(f"{rid}: unknown input/client ID")
            continue
        item = by_id[audit_row["id"]]
        record = record_by_id[item["id"]]
        needed = [
            ("start", starts, 0),
            ("P/AR admission", admits, 0),
            ("P/AR forward", forwards, 0),
            ("P/AR outputs", tokens, 0),
            ("AR admission", admits, ar_stage),
            ("AR forward", forwards, ar_stage),
            ("AR outputs", tokens, ar_stage),
            ("AR completion", dones, ar_stage),
        ]
        if deployment == "pd":
            needed += [("P completion", dones, 0), ("D KV ready", ready, 1)]
        missing = [name for name, target, stage in needed if not target[(rid, stage)]]
        if missing:
            errors.append(f"{rid}: missing {', '.join(missing)}")
            continue
        for stage in range(ar_stage + 1):
            admission = admits[(rid, stage)]
            if len(admission) != 1 or any(
                a.get("max_num_seqs") != 64
                or a.get("max_num_batched_tokens") != 8192
                or a.get("prefix_caching") is not True
                for a in admission
            ):
                errors.append(f"{rid}: running stage {stage} config/admission differs from run03")
        if (
            len(starts[(rid, 0)]) != 1
            or len(dones[(rid, ar_stage)]) != 1
            or (deployment == "pd" and len(dones[(rid, 0)]) != 1)
        ):
            errors.append(f"{rid}: duplicate start/completion")
        ar_outputs = [t for t in tokens[(rid, ar_stage)] if t["delta_tokens"] > 0]
        prefill_outputs = [t for t in tokens[(rid, 0)] if t["delta_tokens"] > 0]
        if not ar_outputs or not prefill_outputs:
            errors.append(f"{rid}: no P/AR token events")
            continue
        cumulative = 0
        for index, event in enumerate(ar_outputs):
            cumulative += event["delta_tokens"]
            if cumulative != event["output_tokens"] or (index > 0 and event["delta_tokens"] != 1):
                errors.append(f"{rid}: missing/duplicate/multi-token AR event; ITL is ambiguous")
        if cumulative != audit_row["output_tokens"] or not tokens[(rid, ar_stage)][-1]["finished"]:
            errors.append(f"{rid}: terminal token coverage differs from audit")
        cached = next((t["cached_tokens"] for t in tokens[(rid, 0)] if t.get("cached_tokens") is not None), None)
        if not isinstance(cached, int) or not 0 <= cached <= audit_row["input_tokens"]:
            errors.append(f"{rid}: invalid/missing local cached-token count")
            cached = None
        if audit_row["input_tokens"] != item["expected_input_tokens"]:
            errors.append(f"{rid}: runtime full input length differs from frozen template")
        start = starts[(rid, 0)][0]["mono_ns"]
        done = dones[(rid, ar_stage)][0]["mono_ns"]
        first = ar_outputs[0]["mono_ns"]
        prefill_end = dones[(rid, 0)][0]["mono_ns"] if deployment == "pd" else prefill_outputs[0]["mono_ns"]
        first_forward = forwards[(rid, 0)][0]["mono_ns"]
        ar_forward = forwards[(rid, ar_stage)][0]["mono_ns"]
        kv_ready = ready[(rid, 1)][0]["mono_ns"] if deployment == "pd" else None
        if not record["submitted_mono_ns"] <= start <= first_forward <= prefill_end <= done:
            errors.append(f"{rid}: invalid host event ordering")
        if not start <= ar_forward <= first <= done:
            errors.append(f"{rid}: invalid AR forward/token ordering")
        if kv_ready is not None and not prefill_end <= kv_ready <= ar_forward:
            errors.append(f"{rid}: D KV ready is outside P handoff / D forward interval")
        rows.append(
            dict(
                id=item["id"],
                server_request_id=rid,
                wave_id=item["wave_id"],
                input_tokens=audit_row["input_tokens"],
                cached_tokens=cached,
                uncached_tokens=audit_row["input_tokens"] - cached if cached is not None else None,
                output_tokens=audit_row["output_tokens"],
                submitted_mono_ns=record["submitted_mono_ns"],
                request_start_mono_ns=start,
                prefill_forward_submit_mono_ns=first_forward,
                prefill_first_token_mono_ns=prefill_outputs[0]["mono_ns"],
                prefill_done_mono_ns=prefill_end,
                d_kv_ready_mono_ns=kv_ready,
                ar_forward_submit_mono_ns=ar_forward,
                first_ar_output_mono_ns=first,
                ar_complete_mono_ns=done,
                ar_complete_s=(done - start) / 1e9,
                client_to_ar_complete_s=(done - record["submitted_mono_ns"]) / 1e9,
                token_output_mono_ns=[e["mono_ns"] for e in ar_outputs],
            )
        )
    if len(rows) != len(batch):
        errors.append("Incomplete request coverage")
    timing = describe([r["dispatch_lateness_s"] for r in records])
    # A delayed launcher would silently change the intended arrival workload.
    if any(abs(r["dispatch_lateness_s"]) > 0.5 for r in records):
        errors.append("Client dispatch missed a wave deadline by more than 0.5s")
    total_time = (
        ((max(r["ar_complete_mono_ns"] for r in rows) - min(r["submitted_mono_ns"] for r in records)) / 1e9)
        if rows
        else None
    )
    if total_time is not None and total_time <= 0:
        errors.append("Non-positive AR window")
    waves = {}
    for wave in sorted({r["wave_id"] for r in rows}):
        group = [r for r in rows if r["wave_id"] == wave]
        waves[str(wave)] = dict(
            ar_complete_s=describe([r["ar_complete_s"] for r in group]),
            client_to_ar_complete_s=describe([r["client_to_ar_complete_s"] for r in group]),
            uncached_input_tokens=sum(r["uncached_tokens"] for r in group)
            if all(r["uncached_tokens"] is not None for r in group)
            else None,
        )
    overlap = None
    if any(r["wave_id"] == 2 for r in records):
        second = min(r["submitted_mono_ns"] for r in records if r["wave_id"] == 2)
        first_wave = [r for r in rows if r["wave_id"] == 1]
        before, after, per_request = [], [], []
        for row in first_wave:
            times = row["token_output_mono_ns"]
            pairs = [(b, (b - a) / 1e6) for a, b in zip(times, times[1:])]
            pre = [gap for end, gap in pairs if second - 5e9 <= end < second]
            post = [gap for end, gap in pairs if second <= end < second + 5e9]
            before.extend(pre)
            after.extend(post)
            per_request.append(dict(id=row["id"], before_itl_ms=describe(pre), after_itl_ms=describe(post)))
        overlap = dict(
            second_wave_submitted_mono_ns=second,
            first_wave_decoding_count=sum(
                r["first_ar_output_mono_ns"] < second < r["token_output_mono_ns"][-1] for r in first_wave
            ),
            first_wave_itl_before_ms=describe(before),
            first_wave_itl_after_ms=describe(after),
            per_request=per_request,
            scope="Host token intervals classified by END in +/-5s windows; crossing intervals included",
        )
        overlap["target_window_observed"] = overlap["first_wave_decoding_count"] > 0
    pipeline = None
    if deployment == "pd" and rows:
        earliest = min(r["first_ar_output_mono_ns"] for r in rows)
        pipeline = dict(
            first_d_output_mono_ns=earliest,
            pending_prefill_count=sum(r["request_start_mono_ns"] < earliest < r["prefill_done_mono_ns"] for r in rows),
            submitted_prefill_not_done_count=sum(
                r["prefill_forward_submit_mono_ns"] < earliest < r["prefill_done_mono_ns"] for r in rows
            ),
            scope="Host observations, not simultaneous-device-kernel proof",
        )
    complete = not errors
    total_tokens = sum(r["output_tokens"] for r in rows)
    return dict(
        complete=complete,
        errors=sorted(set(errors)),
        requests=rows,
        waves=waves,
        dispatch_lateness_s=timing,
        two_wave_overlap=overlap,
        pipeline_overlap=pipeline,
        ar_window_s=total_time if complete else None,
        output_tokens=total_tokens if complete else None,
        output_tokens_per_s=total_tokens / total_time if complete else None,
        output_tokens_per_s_per_ar_card=total_tokens / total_time / (8 if deployment == "pd" else 4)
        if complete
        else None,
        scope="One host monotonic clock. One output sequence per request; PD imported first token counted once.",
    )

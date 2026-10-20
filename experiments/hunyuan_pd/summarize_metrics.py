"""Read existing stage log tables; never substitute missing AR metrics with zero."""

import math
import re
import statistics


def summary(values):
    values = [float(value) for value in values if math.isfinite(value) and value > 0]
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_stage_metrics(log_text, deployment, expected_count):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
    by_request = {}
    request_id, columns = None, []
    ambiguous = False
    for line in clean.splitlines():
        match = re.search(r"\[StageRequestStats \[request_id=([^\]]+)\]\]", line)
        if match:
            request_id, columns = match[1], []
            by_request.setdefault(request_id, {})
            continue
        if request_id is None or "|" not in line:
            continue
        cells = [cell.strip() for cell in line[line.index("|") :].split("|")[1:-1]]
        if not cells:
            continue
        if cells[0] == "Field":
            columns = cells[1:]
            continue
        if cells[0] not in ("vllm_ttft_ms", "vllm_tpot_ms"):
            continue
        for column, cell in zip(columns, cells[1:]):
            # Multiple events for a stage are ambiguous; do not silently
            # count a repeated table/stage as another completed request.
            if "_" in column:
                ambiguous = True
                continue
            try:
                value = float(cell.replace(",", ""))
            except ValueError:
                continue
            key = (column, cells[0])
            previous = by_request[request_id].get(key)
            if previous is not None and previous != value:
                ambiguous = True
            by_request[request_id][key] = value
    result = {
        "request_table_count": len(by_request),
        "expected_count": expected_count,
        "assignment": "round log byte interval; requires exclusive service traffic and an unrotated log",
        "request_ids": list(by_request),
    }
    for name, stage, field in (
        ("ttft_ms", "0", "vllm_ttft_ms"),
        ("tpot_ms", "1" if deployment == "pd" else "0", "vllm_tpot_ms"),
    ):
        observed = summary([rows.get((stage, field), 0) for rows in by_request.values()])
        complete = len(by_request) == expected_count and observed["n"] == expected_count and not ambiguous
        result[name] = dict(
            observed if complete else summary([]),
            observed_n=observed["n"],
            status="complete" if complete else "N/A: missing, zero, ambiguous, or mismatched log coverage",
            stage=int(stage),
        )
    return result

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
    tables = []
    table = None
    for line in clean.splitlines():
        match = re.search(r"\[StageRequestStats \[request_id=([^\]]+)\]\]", line)
        if match:
            request_id, columns = match[1], []
            by_request.setdefault(request_id, {})
            table = {}
            tables.append((request_id, table))
            continue
        # Do not consume a later E2E/transfer table as part of this stage table.
        if re.search(r"\[(?:RequestE2EStats|TransferStats|OmniTiming)\b", line):
            request_id, columns = None, []
        if request_id is None or "|" not in line:
            continue
        cells = [cell.strip() for cell in line[line.index("|") :].split("|")[1:-1]]
        if not cells:
            continue
        if cells[0] == "Field":
            if columns:
                # Another table began without a StageRequestStats title.
                request_id, columns = None, []
                continue
            columns = cells[1:]
            continue
        if not columns or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", cells[0]):
            continue
        for column, cell in zip(columns, cells[1:]):
            table.setdefault(column, {})[cells[0]] = cell

    ambiguous = False
    duplicate_columns = 0
    conflicts = []
    stage_rows = {}
    for request_id, table in tables:
        for column, fields in table.items():
            match = re.fullmatch(r"(\d+)(?:_([2-9]\d*|1\d+))?", column)
            if not match:
                ambiguous = True
                conflicts.append(f"{request_id}: invalid stage column {column}")
                continue
            stage = match[1]
            if match[2]:
                # The suffix is added by metrics.utils._format_table. Collapse
                # only if ALL printed fields match, not just TTFT/TPOT.
                if stage not in table or table[stage] != fields:
                    ambiguous = True
                    conflicts.append(f"{request_id}: {column} differs from {stage}")
                else:
                    duplicate_columns += 1
                continue
            key = (request_id, stage)
            if key in stage_rows and stage_rows[key] != fields:
                ambiguous = True
                conflicts.append(f"{request_id}: conflicting repeated stage {stage}")
            stage_rows[key] = fields
        for column, fields in table.items():
            if not column.isdigit():
                continue
            for field in ("vllm_ttft_ms", "vllm_tpot_ms"):
                cell = fields.get(field, "")
                try:
                    value = float(cell.replace(",", ""))
                except ValueError:
                    continue
                by_request[request_id][(column, field)] = value
    result = {
        "request_table_count": len(by_request),
        "expected_count": expected_count,
        "assignment": "round log byte interval; requires exclusive service traffic and an unrotated log",
        "request_ids": list(by_request),
        "identical_duplicate_columns_collapsed": duplicate_columns,
        "conflicts": conflicts,
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

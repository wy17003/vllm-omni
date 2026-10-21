# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host observations; not kernel timings. Never synchronize devices."""

import json
import logging
import os
import socket
import time

ENABLED = os.environ.get("VLLM_OMNI_HY3_AR_TRACE", "0") == "1"
HOST = socket.gethostname()
logger = logging.getLogger("vllm")


def emit(event: str, stage: int, fields: dict[str, object], timestamp_ns: int | None = None) -> None:
    if ENABLED:
        row = dict(
            event=event,
            stage=stage,
            mono_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
            host=HOST,
            pid=os.getpid(),
            **fields,
        )
        logger.info("[HY3_AR_TRACE] %s", json.dumps(row, separators=(",", ":")))

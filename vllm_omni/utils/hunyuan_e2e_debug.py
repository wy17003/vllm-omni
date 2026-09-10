# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped, read-only diagnostics for Hunyuan's ordinary DiT forward."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch

from vllm_omni.utils.debug_fingerprint import (
    bytes_fingerprint,
    full_kv_fingerprint,
    tensor_fingerprint,
)

logger = logging.getLogger(__name__)


class HunyuanE2EProbe:
    def __init__(self, request_id, tp_rank, config):
        self.request_id = request_id
        self.tp_rank = tp_rank
        self.steps = set(config.get("debug_denoise_steps", [1, 10, 25, 50]))
        self.dump_dir = config.get("debug_e2e_dump_dir")

    def emit(self, event, **data):
        logger.info(
            "[HY3_E2E] event=%s request_id=%s tp_rank=%s data=%s",
            event,
            self.request_id,
            self.tp_rank,
            json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )

    def record(self, event, *, tensors=None, kv=None, images=None, **data):
        """A failed observation is explicit in logs, never an inference failure."""
        try:
            for name, tensor in (tensors or {}).items():
                if tensor is not None:
                    data[name] = tensor_fingerprint(tensor)
            if kv is not None:
                data["kv"] = full_kv_fingerprint(*kv)
            if images is not None:
                data["images"] = [
                    {"size": list(image.size), "mode": image.mode, "pixel_sha256": bytes_fingerprint(image.tobytes())}
                    for image in images
                ]
            self.emit(event, **data)
            # Optional artifacts permit actual cross-run errors to be measured;
            # min/mean/std and unequal hashes alone cannot provide those errors.
            if self.dump_dir and event in {"dit_initial", "dit_final"}:
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(self.request_id))
                path = Path(self.dump_dir) / f"{name}.tp{self.tp_rank}.{event}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.detach().cpu().clone() for k, v in (tensors or {}).items() if v is not None}, path)
                self.emit("artifact", source_event=event, path=str(path.resolve()))
        except Exception:
            logger.exception(
                "[HY3_E2E] event=probe_error request_id=%s tp_rank=%s source_event=%s",
                self.request_id,
                self.tp_rank,
                event,
            )

    def denoise(self, step, total_steps, timestep, prediction, latents):
        if step in self.steps or step == total_steps:
            self.record(
                "dit_step",
                step=step,
                total_steps=total_steps,
                tensors={"timestep": timestep, "prediction": prediction, "latents": latents},
            )

    def injected_kv(self, layers):
        # Do not call _snapshot_injected_ar_kv: it clears the live layer state.
        try:
            keys, values = [], []
            for layer in layers:
                entries = layer.self_attn.image_attn._injected_ar_kv
                if entries is None or len(entries) != 1:
                    raise ValueError("Expected one injected AR KV branch per layer (CFG disabled)")
                key, value = entries[0]
                keys.append(key)
                values.append(value)
            self.record("dit_injected_kv", kv=(keys, values))
        except Exception:
            logger.exception(
                "[HY3_E2E] event=probe_error request_id=%s tp_rank=%s source_event=dit_injected_kv",
                self.request_id,
                self.tp_rank,
            )

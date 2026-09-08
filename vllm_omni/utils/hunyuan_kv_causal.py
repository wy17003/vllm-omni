# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-request AR prompt-tail KV intervention (diagnostics only)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import torch

from vllm_omni.utils.debug_fingerprint import gather_paged_tensor_tokens, int_sequence_fingerprint


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"[HY3_KV_CAUSAL] invalid experiment: {message}")


class HunyuanKVCausalProbe:
    """Exchange CPU tail snapshots between co-located P/D workers by request/rank.

    ``check`` only compares; ``restore`` writes the producer tail after the
    consumer's first forward. Neither mode changes hidden states or sampling.
    The runner must invoke this on every TP rank, with PP/CP=1 and batch=1.
    """

    def __init__(self, mode: str, directory: str, role: str, rank: int, tp_size: int):
        _require(mode in {"check", "restore"}, f"unsupported mode {mode!r}")
        _require(role in {"kv_producer", "kv_consumer", "non_pd"}, f"unsupported role {role!r}")
        _require(role != "non_pd" or mode == "check", "non-PD supports check only")
        _require(bool(directory), "VLLM_OMNI_HY3_KV_CAUSAL_DIR must be set")
        self.mode, self.role, self.rank, self.tp_size = mode, role, rank, tp_size
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        self.pending: dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)

    def _log(self, event: str, request_id: str, **fields) -> None:
        self.logger.info(
            "[HY3_KV_CAUSAL] %s",
            json.dumps(
                dict(event=event, request_id=request_id, role=self.role, mode=self.mode, tp_rank=self.rank, **fields),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _path(self, request_id: str) -> Path:
        key = hashlib.sha256(request_id.encode()).hexdigest()
        return self.directory / f"{key}.tp{self.rank}.pt"

    def finish(self, request_ids) -> None:
        for request_id in request_ids:
            _require(request_id not in self.pending, f"{request_id}: finished before first-token verification")
            self.done.discard(request_id)

    @torch.inference_mode()
    def after_forward(
        self,
        request_id: str,
        prompt_ids: list[int],
        computed: int,
        scheduled: int,
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        if request_id in self.done:
            return
        length = len(prompt_ids)
        _require(length > 0, "empty prompt")
        # This experiment intentionally excludes chunking, prefix-cache hits,
        # and schedules that already include generated tokens.
        expected = (length - 1, 1) if self.role == "kv_consumer" else (0, length)
        _require((computed, scheduled) == expected, f"first schedule {(computed, scheduled)} != {expected}")
        _require(bool(layers), "no KV layers")
        _require(block_size > 0, "invalid block size")
        _require(len(block_ids) == (length + block_size - 1) // block_size, "invalid block table length")
        _require(len(set(block_ids)) == len(block_ids), "aliased prompt blocks")
        metadata = dict(
            schema=1,
            request_id=request_id,
            prompt_len=length,
            prompt_sha256=int_sequence_fingerprint(prompt_ids),
            tp_rank=self.rank,
            tp_size=self.tp_size,
            layer_count=len(layers),
            block_size=block_size,
        )
        entries, tails, targets = [], [], []
        for layer_index, pair in enumerate(layers):
            _require(pair is not None and len(pair) == 2, f"layer {layer_index}: unsupported KV layout")
            for cache_name, blocks in zip(("key", "value"), pair, strict=True):
                _require(blocks.ndim == 4 and blocks.shape[1] == block_size, "expected [blocks, tokens, heads, dim]")
                # Gather only valid logical prompt positions, independent of
                # the producer/consumer physical block allocations.
                logical = gather_paged_tensor_tokens(blocks, block_ids, list(range(length)), block_size).cpu()
                _require(bool(torch.isfinite(logical).all()), f"layer {layer_index}/{cache_name}: non-finite KV")
                tail = logical[-1].clone()
                entries.append(
                    dict(
                        layer=layer_index,
                        cache=cache_name,
                        shape=list(tail.shape),
                        dtype=str(tail.dtype),
                        prefix_sha256=_digest(logical[:-1]),
                        tail_sha256=_digest(tail),
                    )
                )
                tails.append(tail)
                targets.append(blocks[block_ids[(length - 1) // block_size], (length - 1) % block_size])
        self._log("post_forward", request_id, metadata=metadata, entries=entries, tail_position=length - 1)

        if self.role == "kv_producer":
            _require(
                not self._path(request_id).exists(), "snapshot already exists; use a fresh run directory/request ID"
            )
            # Publish only after sampling: by then the snapshot also contains
            # the producer's actual first output token, before D is submitted.
            self.pending[request_id] = dict(metadata=metadata, entries=entries, tails=tails)
        elif self.role == "kv_consumer":
            path = self._path(request_id)
            _require(path.is_file(), f"producer snapshot missing: {path}; P/D need the same filesystem directory")
            reference = torch.load(path, map_location="cpu", weights_only=True)
            _require(reference.get("metadata") == metadata, "producer snapshot metadata mismatch")
            source_entries, source_tails = reference["entries"], reference["tails"]
            _require(len(source_entries) == len(entries) == len(source_tails), "incomplete producer layers")
            _require(isinstance(reference.get("first_token"), int), "missing producer first token")
            # Validate ALL layers before the first write, including the entire
            # untouched prefix [0, length-1), not just selected token samples.
            for entry, source, tail in zip(entries, source_entries, source_tails, strict=True):
                for field in ("layer", "cache", "shape", "dtype", "prefix_sha256"):
                    _require(entry[field] == source[field], f"{entry['layer']}/{entry['cache']}: {field} mismatch")
                _require(
                    list(tail.shape) == entry["shape"] and str(tail.dtype) == entry["dtype"], "tail layout mismatch"
                )
                _require(_digest(tail) == source["tail_sha256"], "producer tail checksum mismatch")
            changed = [
                dict(layer=e["layer"], cache=e["cache"])
                for e, s in zip(entries, source_entries, strict=True)
                if e["tail_sha256"] != s["tail_sha256"]
            ]
            self._log(
                "prefix_verified", request_id, prefix_tokens=length - 1, kv_tensor_count=len(entries), changed=changed
            )
            if self.mode == "restore":
                for target, tail in zip(targets, source_tails, strict=True):
                    target.copy_(tail.to(device=target.device))
                # CPU readback synchronizes the writes before subsequent decode.
                restored = [_digest(target) for target in targets]
                _require(restored == [entry["tail_sha256"] for entry in source_entries], "tail readback mismatch")
                self._log(
                    "tail_restored",
                    request_id,
                    tail_position=length - 1,
                    kv_tensor_count=len(restored),
                    tail_sha256=restored,
                    readback_equal=True,
                )
            self.pending[request_id] = dict(first_token=reference["first_token"])
        self.done.add(request_id)

    def after_sample(self, request_ids: list[str], sampled_token_ids: torch.Tensor) -> None:
        for row, request_id in enumerate(request_ids):
            if request_id not in self.pending:
                continue
            _require(sampled_token_ids.ndim == 2 and sampled_token_ids.shape[1] == 1, "expected one sampled token")
            token = int(sampled_token_ids[row, 0].item())
            _require(token >= 0, "invalid first sampled token")
            state = self.pending[request_id]
            if self.role == "kv_producer":
                state["first_token"] = token
                fd, tmp = tempfile.mkstemp(prefix=".hy3-", suffix=".tmp", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        torch.save(state, stream)
                    os.replace(tmp, self._path(request_id))
                finally:
                    Path(tmp).unlink(missing_ok=True)
                self._log("producer_ready", request_id, first_token=token)
            else:
                expected = state["first_token"]
                self._log(
                    "first_token_check",
                    request_id,
                    first_token=token,
                    producer_first_token=expected,
                    equal=token == expected,
                )
                _require(token == expected, f"consumer first token {token} != producer {expected}")
                self._path(request_id).unlink()
            del self.pending[request_id]

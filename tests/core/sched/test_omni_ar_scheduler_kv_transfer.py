from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_to_dit_transfer_keeps_prefill_and_decode_blocks():
    """The DiT export must include blocks allocated for the imported prefix."""
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.requests_needing_kv_transfer = {}
    scheduler._should_transfer_kv_for_request = lambda _request_id: True
    scheduler.kv_cache_manager = SimpleNamespace(
        # Blocks 10 and 11 represent the imported Prefill prefix; block 12
        # contains locally generated Decode tokens. Block 13 is unused tail.
        get_block_ids=lambda _request_id: ([10, 11, 12, 13],),
    )
    scheduler.cache_config = SimpleNamespace(block_size=4)

    scheduler._mark_request_for_kv_transfer("req", seq_len=10)

    assert scheduler.requests_needing_kv_transfer["req"] == {
        "seq_len": 10,
        "block_ids": [10, 11, 12],
    }

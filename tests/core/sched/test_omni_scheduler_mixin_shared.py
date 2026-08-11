from collections import defaultdict
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniChunkRecvHandle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    pass


def test_schedule_lifecycle_helpers_process_and_restore_both_input_paths():
    calls = []
    scheduler = _Scheduler()
    scheduler.waiting = ["waiting"]
    scheduler.running = ["running"]
    scheduler.requests = {"request": object()}
    scheduler._consume_pending_connector_output = lambda mode: calls.append(("consume", mode))
    scheduler._process_pending_input_timeouts = lambda: calls.append(("timeouts",))
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        process_pending_chunks=lambda waiting, running, scheduler_requests: calls.append(
            ("process", waiting, running, scheduler_requests)
        ),
        restore_queues=lambda waiting, running, scheduler_requests: calls.append(
            ("restore-chunks", waiting, running, scheduler_requests)
        ),
    )
    scheduler.input_coordinator = SimpleNamespace(
        restore_queues=lambda waiting: calls.append(("restore-full", waiting))
    )

    scheduler._process_pending_omni_inputs("ar")
    scheduler._restore_omni_wait_queues()

    assert calls == [
        ("consume", "ar"),
        ("timeouts",),
        ("process", scheduler.waiting, scheduler.running, scheduler.requests),
        ("restore-chunks", scheduler.waiting, scheduler.running, scheduler.requests),
        ("restore-full", scheduler.waiting),
    ]


@pytest.mark.parametrize(
    ("synthesize_abort_outputs", "expected_finish_reason"),
    [(False, None), (True, FinishReason.ABORT)],
)
def test_finished_request_attachment_keeps_ar_abort_policy_explicit(
    synthesize_abort_outputs,
    expected_finish_reason,
):
    scheduler = _Scheduler()
    scheduler.finished_req_ids_dict = defaultdict(set, {2: {"req-finished"}})
    outputs = {}

    scheduler._attach_finished_request_sets(
        outputs,
        synthesize_abort_outputs=synthesize_abort_outputs,
    )

    assert outputs[2].finished_requests == {"req-finished"}
    if expected_finish_reason is None:
        assert outputs[2].outputs == []
    else:
        assert outputs[2].outputs[0].finish_reason == expected_finish_reason
    assert scheduler.finished_req_ids_dict == {}


def test_chunk_receive_handle_carries_minimal_registration_fields():
    handle = OmniChunkRecvHandle(request_id="req", external_req_id="external")
    assert handle.request_id == "req"
    assert handle.external_req_id == "external"


def test_output_helper_preserves_required_nan_counter_default():
    scheduler = _Scheduler()
    request = SimpleNamespace(
        request_id="req-output",
        trace_headers=None,
        take_events=lambda: [],
    )

    output = scheduler._make_omni_engine_output(request, new_token_ids=[])

    assert output.num_nans_in_logits == 0


def test_update_helpers_skip_disabled_metrics_and_missing_kv_failures():
    scheduler = _Scheduler()
    scheduler.perf_metrics = SimpleNamespace(is_enabled=lambda: False)
    scheduler._handle_invalid_blocks = lambda *_args: pytest.fail("unexpected KV recovery")

    assert scheduler._take_step_perf_stats(SimpleNamespace()) is None
    assert scheduler._get_failed_kv_load_request_ids(None, {}) is None
    assert scheduler._get_failed_kv_load_request_ids(SimpleNamespace(invalid_block_ids=[]), {}) is None


def test_output_assembly_keeps_completion_stats_and_connector_capture_order():
    scheduler = _Scheduler()
    calls = []
    scheduler._attach_finished_request_sets = lambda outputs, **kwargs: calls.append(
        ("finished", outputs, kwargs)
    )
    scheduler._attach_scheduler_stats = lambda *args: calls.append(("stats", args))
    scheduler._capture_omni_connector_output = lambda output: calls.append(("connector", output))
    runner_output = SimpleNamespace()

    assembled = scheduler._assemble_engine_core_outputs(
        {3: []},
        synthesize_abort_outputs=True,
        spec_decoding_stats=None,
        kv_connector_stats=None,
        cudagraph_stats=None,
        perf_stats=None,
        model_runner_output=runner_output,
    )

    assert set(assembled) == {3}
    assert [call[0] for call in calls] == ["finished", "stats", "connector"]
    assert calls[0][2] == {"synthesize_abort_outputs": True}

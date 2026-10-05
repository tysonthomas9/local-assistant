"""The priority gate's ordering rules (pure logic; `llm/priority_gate.yaml` runs it for real)."""

import asyncio

import pytest

from assistant_brain.adapters.priority_gate import PriorityGate, RequestClass

pytestmark = pytest.mark.unit


async def _queue(gate: PriorityGate, request_id: str, cls: RequestClass) -> asyncio.Task[None]:
    task = asyncio.create_task(gate.acquire(request_id, cls))
    await asyncio.sleep(0)
    return task


def _admitted(gate: PriorityGate) -> list[str]:
    return [e.request_id for e in gate.log if e.kind == "admitted"]


async def test_never_more_than_max_in_flight() -> None:
    gate = PriorityGate(3, 0)
    tasks = [await _queue(gate, f"b{i}", "background") for i in range(5)]
    assert len(gate.in_flight) == 3
    assert gate.waiting == 2
    gate.release("b0")
    await asyncio.sleep(0)
    assert len(gate.in_flight) == 3
    assert gate.max_in_flight_seen == 3
    for task in tasks[3:]:
        task.cancel()


async def test_reserved_voice_slot_is_never_given_to_other_classes() -> None:
    gate = PriorityGate(3, 1)
    background = [await _queue(gate, f"b{i}", "background") for i in range(3)]
    proactive = await _queue(gate, "p0", "proactive")
    assert sorted(gate.in_flight) == ["b0", "b1"]
    assert not background[2].done()
    assert not proactive.done()
    voice = await _queue(gate, "v0", "voice")
    assert voice.done()
    assert sorted(gate.in_flight) == ["b0", "b1", "v0"]
    gate.release("v0")
    await asyncio.sleep(0)
    assert "v0" not in gate.in_flight
    assert len(gate.in_flight) == 2  # the freed slot is the reserved one: nobody else takes it
    for task in (background[2], proactive):
        task.cancel()


async def test_waiters_are_served_by_class_then_fifo() -> None:
    gate = PriorityGate(1, 0)
    await _queue(gate, "first", "background")
    tasks = [
        await _queue(gate, "b1", "background"),
        await _queue(gate, "p1", "proactive"),
        await _queue(gate, "b2", "background"),
        await _queue(gate, "v1", "voice"),
        await _queue(gate, "p2", "proactive"),
        await _queue(gate, "v2", "voice"),
    ]
    order = ["first"]
    while len(order) < 7:
        gate.release(order[-1])
        await asyncio.sleep(0)
        (current,) = gate.in_flight
        order.append(current)
    assert order == ["first", "v1", "v2", "p1", "p2", "b1", "b2"]
    assert _admitted(gate) == order
    assert all(t.done() for t in tasks)


async def test_a_new_request_does_not_overtake_waiters_of_its_class_or_higher() -> None:
    gate = PriorityGate(2, 1)
    await _queue(gate, "b0", "background")
    waiting = await _queue(gate, "b1", "background")
    assert not waiting.done()
    voice = await _queue(gate, "v0", "voice")
    assert voice.done()  # voice is above every waiter: straight in
    gate.release("v0")
    await asyncio.sleep(0)
    late = await _queue(gate, "b2", "background")
    gate.release("b0")
    await asyncio.sleep(0)
    assert waiting.done()
    assert not late.done()  # FIFO: b1 waited first
    late.cancel()


async def test_a_cancelled_waiter_gives_up_its_place() -> None:
    gate = PriorityGate(1, 0)
    await _queue(gate, "a", "background")
    gone = await _queue(gate, "b", "background")
    next_one = await _queue(gate, "c", "background")
    gone.cancel()
    await asyncio.sleep(0)
    assert [e.kind for e in gate.log if e.request_id == "b"] == ["queued", "cancelled"]
    gate.release("a")
    await asyncio.sleep(0)
    assert next_one.done()
    assert list(gate.in_flight) == ["c"]


def test_reserved_slots_must_leave_room() -> None:
    with pytest.raises(ValueError, match="reserved_voice_slots"):
        PriorityGate(2, 2)

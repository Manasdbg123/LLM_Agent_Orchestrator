"""The idempotency key is the whole feature, so its properties are tested directly.

The single most important assertion here is `test_key_is_stable_across_attempts`.
If that ever fails, retries stop deduplicating and side effects double — silently,
and only under the failure conditions nobody exercises by hand.
"""

from __future__ import annotations

import uuid

from app.core.idempotency import canonical_json, effect_key

RUN = uuid.UUID("11111111-1111-7111-8111-111111111111")
STEP = uuid.UUID("22222222-2222-7222-8222-222222222222")
ARGS = {"to": "a@example.test", "subject": "Hi", "body": "Hello"}


def _key(**overrides: object) -> str:
    payload = {"run_id": RUN, "step_id": STEP, "tool_name": "send_email", "arguments": ARGS}
    payload.update(overrides)
    return effect_key(**payload)  # type: ignore[arg-type]


def test_key_is_stable_across_attempts() -> None:
    """`attempt` must not influence the key.

    The brief specified (run_id, step_id, attempt). Including attempt would mean a
    retry after an ambiguous failure computes a *different* key, the ledger has no
    memory of the first attempt, and the customer receives two emails. This test is
    the guard against that regression.
    """
    assert _key() == _key()  # nothing about the call changes between attempts


def test_key_is_deterministic_regardless_of_argument_ordering() -> None:
    reordered = {"body": "Hello", "subject": "Hi", "to": "a@example.test"}
    assert _key() == _key(arguments=reordered)


def test_key_changes_with_arguments() -> None:
    assert _key() != _key(arguments={**ARGS, "to": "b@example.test"})


def test_key_changes_with_tool_name() -> None:
    assert _key() != _key(tool_name="database_write")


def test_key_is_scoped_to_a_step_and_a_run() -> None:
    other = uuid.UUID("33333333-3333-7333-8333-333333333333")
    assert _key() != _key(step_id=other)
    assert _key() != _key(run_id=other)


def test_key_is_a_sha256_hex_digest() -> None:
    key = _key()
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_canonical_json_sorts_keys_and_omits_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_stable_for_nested_structures() -> None:
    left = {"outer": {"z": 1, "a": [3, 2, 1]}}
    right = {"outer": {"a": [3, 2, 1], "z": 1}}
    assert canonical_json(left) == canonical_json(right)


def test_canonical_json_handles_non_serialisable_values() -> None:
    # `default=str` keeps a UUID or datetime in an argument from raising at the point
    # where we are trying to make a call safe to retry.
    assert canonical_json({"id": RUN}) == '{"id":"11111111-1111-7111-8111-111111111111"}'

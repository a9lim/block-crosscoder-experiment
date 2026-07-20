from block_crosscoder_experiment.cli.harvest_store import _close_iterators

import pytest


def test_close_iterators_closes_entire_generator_chain_in_order():
    events = []

    class Closable:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(self.name)

    _close_iterators(
        Closable("batches"),
        Closable("rows"),
        Closable("documents"),
        Closable("stream"),
    )
    assert events == ["batches", "rows", "documents", "stream"]


def test_close_iterators_ignores_nonclosable_objects():
    _close_iterators(object(), None)


def test_close_iterators_closes_inner_streams_after_outer_error():
    events = []

    class BrokenOuter:
        def close(self):
            events.append("outer")
            raise RuntimeError("close failed")

    class Inner:
        def close(self):
            events.append("inner")

    with pytest.raises(RuntimeError, match="close failed"):
        _close_iterators(BrokenOuter(), Inner())
    assert events == ["outer", "inner"]


def test_close_iterators_is_idempotent_for_generators():
    events = []

    def stream():
        try:
            yield 1
        finally:
            events.append("closed")

    iterator = stream()
    next(iterator)
    _close_iterators(iterator)
    _close_iterators(iterator)
    assert events == ["closed"]

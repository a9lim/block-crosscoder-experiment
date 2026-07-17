"""CodeStore round-trip, packing, labelers, and store-backed battery parity."""

import torch

from block_crosscoder_experiment.phase0.battery import (
    cluster_restricted_reconstruction,
)
from block_crosscoder_experiment.phase0.harvest import CodeStore, pack_token_rows
from block_crosscoder_experiment.phase0.labels import build_label_map, label_tokens
from block_crosscoder_experiment.phase0.nulls import random_member_sets

T, F = 5000, 96


def _sparse_codes(gen: torch.Generator) -> torch.Tensor:
    codes = torch.rand(T, F, generator=gen)
    return torch.where(codes > 0.95, codes, torch.zeros(()))


def test_code_store_round_trip(tmp_path, device):
    gen = torch.Generator().manual_seed(1)
    codes = _sparse_codes(gen)
    token_ids = torch.randint(0, 50000, (T,), generator=gen)

    writer = CodeStore.open_writer(tmp_path / "store", F, {"note": "test"})
    for start in range(0, T, 1500):  # uneven shards on purpose
        writer.add_shard(codes[start : start + 1500], token_ids[start : start + 1500])
    store = writer.finalize()

    assert store.n_tokens == T and store.n_features == F
    assert torch.equal(store.token_ids(), token_ids.to(torch.int32))
    assert torch.equal(store.firing_counts(), codes.gt(0).sum(0))

    members = torch.tensor([3, 17, 40, 95])
    dense = store.select_members(members, device=device)
    assert torch.allclose(dense.cpu(), codes[:, members])

    chunks = torch.cat(
        [c.cpu() for c in store.iter_dense_chunks(chunk=700, device=device)]
    )
    assert torch.allclose(chunks, codes)


def test_store_backed_battery_matches_dense(tmp_path, device):
    gen = torch.Generator().manual_seed(2)
    codes = _sparse_codes(gen)
    writer = CodeStore.open_writer(tmp_path / "store", F, {})
    writer.add_shard(codes, torch.zeros(T, dtype=torch.int32))
    store = writer.finalize()

    decoder = torch.randn(F, 16, generator=gen).to(device)
    members = torch.tensor([1, 2, 3, 4, 5])
    r_dense, k_dense = cluster_restricted_reconstruction(
        codes.to(device), decoder, members.to(device)
    )
    r_store, k_store = cluster_restricted_reconstruction(store, decoder, members)
    assert torch.equal(k_dense.cpu(), k_store.cpu())
    assert torch.allclose(r_dense.cpu(), r_store.cpu(), atol=1e-5)


def test_pack_token_rows():
    docs = iter([list(range(10, 300)), list(range(1000, 1200))])
    rows = list(pack_token_rows(docs, ctx=64, bos_id=7, n_rows=5))
    assert len(rows) == 5
    for row in rows:
        assert row.shape == (64,) and int(row[0]) == 7
    flat = torch.cat([r[1:] for r in rows]).tolist()
    assert flat == (list(range(10, 300)) + list(range(1000, 1200)))[: 63 * 5]


def test_frequency_matched_nulls():
    gen = torch.Generator().manual_seed(3)
    freqs = torch.cat([torch.full((50,), 10000.0), torch.full((50,), 3.0)])
    members = torch.tensor([0, 1, 2])  # all high-frequency
    draws = random_member_sets(
        100, 3, n_draws=20, seed=0, exclude=members,
        frequencies=freqs, match_to=members,
    )
    for d in draws:
        assert (d < 50).all(), "null member left the candidate's frequency bucket"
        assert not set(d.tolist()) & {0, 1, 2}
    del gen


def test_weekday_label_map_gpt2():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    mapping = build_label_map(tok, "weekday")
    # every weekday has at least the leading-space capitalized form
    assert set(mapping.values()) == set(range(7))
    monday = tok.encode(" Monday")[0]
    assert mapping[monday] == 0

    ids = torch.tensor([monday, 42, tok.encode(" Sunday")[0]])
    labels = label_tokens(ids, mapping)
    assert labels.tolist() == [0, -1, 6]

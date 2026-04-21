"""Tests for rule canonicalization."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tca_sim.canonical import (
    canonical_form,
    rule_hash,
    NEG_PERM,
    langton_lambda,
    rule_entropy,
)


def _random_rule(seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=51).astype(np.int8)


# ─────────────────────────────────────── NEG_PERM properties ─────────────

def test_neg_perm_is_involution():
    """Applying NEG_PERM twice should return the original index."""
    roundtrip = NEG_PERM[NEG_PERM]
    expected  = np.arange(51, dtype=np.int64)
    np.testing.assert_array_equal(roundtrip, expected)


def test_neg_perm_length():
    assert len(NEG_PERM) == 51


def test_neg_perm_is_permutation():
    assert sorted(NEG_PERM.tolist()) == list(range(51))


# ─────────────────────────────────────── canonical_form ──────────────────

def test_canonical_idempotent():
    """canonical_form(canonical_form(r)) == canonical_form(r)."""
    rule  = _random_rule(seed=7)
    canon = canonical_form(rule)
    np.testing.assert_array_equal(canonical_form(canon), canon)


def test_canonical_negated_gives_same():
    """A rule and its negation-equivalent must canonicalise to the same array."""
    rule     = _random_rule(seed=3)
    neg_rule = (-rule[NEG_PERM]).astype(np.int8)
    np.testing.assert_array_equal(canonical_form(rule), canonical_form(neg_rule))


def test_canonical_lex_order():
    """The canonical form must be ≤ its negation-equivalent lexicographically."""
    for seed in range(50):
        rule  = _random_rule(seed=seed)
        canon = canonical_form(rule)
        neg   = (-canon[NEG_PERM]).astype(np.int8)
        assert canon.tobytes() <= neg.tobytes(), (
            f"seed={seed}: canonical is not lex-smallest"
        )


def test_canonical_dtype():
    rule  = _random_rule(seed=1)
    canon = canonical_form(rule)
    assert canon.dtype == np.int8


# ─────────────────────────────────────── rule_hash ───────────────────────

def test_rule_hash_symmetry_invariant():
    """Symmetry-equivalent rules must produce the same hash."""
    rule     = _random_rule(seed=11)
    neg_rule = (-rule[NEG_PERM]).astype(np.int8)
    assert rule_hash(rule) == rule_hash(neg_rule)


def test_rule_hash_collision_rare():
    """Distinct rules (not related by symmetry) should almost never hash-collide."""
    hashes = {rule_hash(_random_rule(seed=i)) for i in range(200)}
    assert len(hashes) >= 190, "Unexpectedly many hash collisions"


def test_rule_hash_length():
    assert len(rule_hash(_random_rule())) == 12


# ─────────────────────────────────────── lambda / entropy ────────────────

def test_langton_lambda_zero():
    rule = np.zeros(51, dtype=np.int8)
    assert langton_lambda(rule) == 0.0


def test_langton_lambda_one():
    rule = np.ones(51, dtype=np.int8)
    assert langton_lambda(rule) == 1.0


def test_rule_entropy_range():
    for seed in range(20):
        ent = rule_entropy(_random_rule(seed=seed))
        assert 0.0 <= ent <= np.log2(3) + 0.001

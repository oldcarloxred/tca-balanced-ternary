"""
Rule canonicalization for 2D ternary outer-totalistic CAs.

Rule table layout: 51 entries.
  flat_index = state_idx * 17 + sum_idx
  state_idx ∈ {0,1,2}  ←→  state ∈ {-1, 0, +1}
  sum_idx   ∈ {0..16}  ←→  neighbour-sum ∈ {-8..+8}

Negation symmetry
-----------------
Swapping -1 ↔ +1 (negating every cell) maps a rule to an equivalent
dynamical system.  Under negation:
  state_idx   : 0 ↔ 2,  1 stays (2 - state_idx)
  sum_idx     : 0 ↔ 16  (16 - sum_idx)
  output      : negate

The permutation is an involution (its own inverse), so:
  neg_rule = -rule[NEG_PERM]

The canonical form is the lexicographically smaller of rule and neg_rule.
Hashing the canonical form lets us detect symmetry-equivalent discoveries.
"""
import hashlib
import numpy as np


def _build_neg_perm() -> np.ndarray:
    perm = np.empty(51, dtype=np.int64)
    for s in range(3):
        for sig in range(17):
            perm[s * 17 + sig] = (2 - s) * 17 + (16 - sig)
    return perm


NEG_PERM: np.ndarray = _build_neg_perm()


def canonical_form(rule: np.ndarray) -> np.ndarray:
    """Return lex-minimal rule among rule and its negation-equivalent."""
    r = np.asarray(rule, dtype=np.int8)
    n = (-r[NEG_PERM]).astype(np.int8)
    return r if r.tobytes() <= n.tobytes() else n


def rule_hash(rule: np.ndarray) -> str:
    """12-hex-char SHA-1 of the canonical form; stable across symmetries."""
    return hashlib.sha1(canonical_form(rule).tobytes()).hexdigest()[:12]


def langton_lambda(rule: np.ndarray) -> float:
    """Fraction of non-quiescent (non-zero) outputs."""
    return float(np.count_nonzero(rule)) / 51.0


def rule_entropy(rule: np.ndarray) -> float:
    """Shannon entropy (bits) of the output distribution."""
    counts = np.bincount(np.asarray(rule, dtype=np.int8) + 1, minlength=3).astype(float)
    p = counts / 51.0
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def rule_balance(rule: np.ndarray) -> tuple:
    """(f_neg, f_zero, f_pos) fractions of each output state."""
    r = np.asarray(rule, dtype=np.int8)
    return (float(np.mean(r == -1)), float(np.mean(r == 0)), float(np.mean(r == 1)))

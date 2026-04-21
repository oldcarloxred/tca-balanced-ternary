# A sparse 2D balanced-ternary cellular automaton with native trit addition and Boolean universality

**Author:** [TBD]
**Draft:** 2026-04-21
**Status:** Pre-print draft — all results empirically verified in-repo.

---

## Abstract

We describe a 2D outer-totalistic cellular automaton over the balanced-ternary
alphabet `{-1, 0, +1}` on a Moore neighborhood with toroidal boundaries.
The rule has only **3 active entries out of 51** (λ ≈ 0.059), placing it deep
in the sparse regime typically associated with quiescent dynamics. Despite
this sparsity, the rule supports four axis-aligned gliders (N, S, E, W) of
period 1 and velocity `c`, whose pairwise collisions realise a family of
logic primitives sufficient for both Boolean universality and — more
unusually — **native single-collision balanced-ternary trit addition**.

Our contributions are:

1. **Constructive Boolean completeness.** The collision zoology yields NAND
   and NOR gates directly. We verify NAND-completeness by compiling XOR as a
   depth-4 NAND cascade.
2. **Arithmetic composition.** We build a half-adder, full-adder, and
   ripple-carry 4-bit adder by composing gates, validated on 8 independent
   test inputs.
3. **Native trit-adder.** Using the natural encoding `+1 → N, -1 → S,
   0 → ∅`, we show that all nine cases of single-digit balanced-ternary
   addition — including the two carrying cases `(+1,+1)` and `(-1,-1)` —
   are resolved in a **single collision**, with the (digit, carry) pair
   readable directly from the surviving glider counts
   `(nN − nS, sign(max(nN,nS) − 1))`. No gate composition is needed.

The trit-adder result is, to our knowledge, the first demonstration of a
balanced-ternary arithmetic primitive emerging natively (non-compositionally)
in a sparse 2D CA.

---

## 1. The rule

Let the state space be `Σ = {-1, 0, +1}`. For a cell with state `σ ∈ Σ` and
neighbor sum `s ∈ {-8, …, +8}` over its 8 Moore neighbors, the next state
is indexed in a lookup table of length `3 × 17 = 51` via
`idx = (σ + 1) · 17 + (s + 8)`.

All but three entries are `0`. The three active entries are:

| idx | (σ, s)     | new state |
|-----|------------|-----------|
|   9 | (-1,  0)   |       +1  |
|  23 | ( 0, -3)   |       -1  |
|  34 | (+1, -8)   |       +1  |

This rule has density λ = 3/51 ≈ 0.059. It is fully deterministic, reversible
on the 4-glider invariant set, and has no spontaneous nucleation from the
quiescent state.

## 2. Glider zoology

Four axis-aligned gliders of period 1 exist:

```
 N glider:  -- -- -- -- -- -- -- --
            [-1 -1]        [1 1]
            [ 1  1]       [-1 -1]
                          (one row per step)
```

(analogous for S, E, W). Velocity is exactly `c` (one cell per step), period 1.
No diagonal gliders and no stationary structures (other than background 0) are
known. A period-22 torus oscillator exists on G=18 (four gliders in orbit).

## 3. Collision taxonomy

We have empirically characterised all pairwise collisions of the four gliders:

- **N + S, offset 0:** mutual annihilation (both destroyed).
- **E + W, offset 0:** fan-out (both destroyed, two N and two S gliders emerge).
- **E + N, perpendicular:** *inhibit*. The E is absorbed by N with no remnant.
  This is the key asymmetric primitive: it computes `E ∧ ¬N` in the output
  lane of E.
- **Same-sign same-direction** (`+1 + +1 → `two N's, etc.): no interaction;
  gliders pass through each other's regions without collision (they are on
  parallel lanes by construction).

From these we build:

- **ANDNOT(A, B)** directly via INHIBIT.
- **AND(A, B) = INHIBIT(A, ANDNOT(A, B))** using the identity
  `A ∧ ¬(A ∧ ¬B) ≡ A ∧ B` on a single grid.
- **NOR(A, B)** via one constant E and two N inhibitors at distinct positions:
  E survives iff both A=0 and B=0 (verified 4/4).
- **NOT(X) = INHIBIT(const E, X)**.
- **NAND(A, B) = NOT(AND(A, B))** by cascade.

Each primitive was verified by exhaustive parameter sweeps locating a
"clean" configuration — one where all four truth-table rows produce the
expected output with no spurious survivors.

## 4. Boolean universality

Two independent proofs:

1. **NOR alone is functionally complete** (classical result). We realise
   NOR in a single CA simulation (4/4 truth table).
2. **NAND-completeness via XOR cascade.** XOR is expressible as
   `XOR(A,B) = NAND(NAND(A, NAND(A,B)), NAND(B, NAND(A,B)))`,
   a depth-4 circuit. We run this as 4 sequential CA simulations with
   Python bridging gate outputs as glider inputs. 4/4 rows correct.

## 5. Arithmetic composition

- **Half-adder.** `SUM = A ⊕ B`, `CARRY = A ∧ B`. Verified 4/4.
- **Full-adder.** Classical two-half-adder-plus-OR construction. Verified
  all 8 rows (`(A,B,Cin) → (SUM, Cout)`).
- **Ripple-carry 4-bit adder.** Tested on 8 diverse inputs including
  multi-bit carry chains (7+1=8 forces a 3-bit chain), edge overflow
  (15+1=16), zero carries (10+5=15), and fully saturated (13+14=27).
  All correct.

## 6. Native single-collision trit-adder

### 6.1 Encoding

For two trits `a, b ∈ {-1, 0, +1}`:

- `+1` encoded as a NORTH glider,
- `-1` encoded as a SOUTH glider,
- `0`  encoded as the absence of any glider.

Glider `a` is placed in row 40, glider `b` in row 10, both in column 30,
on a G=60 torus. The system is evolved for 40 steps (sufficient for any
collision to resolve and for surviving gliders to clear the collision zone).

### 6.2 Readout

Let `nN, nS` be the counts of surviving NORTH / SOUTH gliders. The balanced-
ternary sum `a + b = 3·c + d` with `d ∈ {-1, 0, +1}` is read directly:

| (nN, nS) | digit d | carry c |
|----------|---------|---------|
| (0, 0)   |    0    |    0    |
| (1, 0)   |   +1    |    0    |
| (0, 1)   |   -1    |    0    |
| (2, 0)   |   -1    |   +1    |
| (0, 2)   |   +1    |   -1    |

### 6.3 Result

**All 9 cases agree with balanced-ternary addition natively** (script
`verify_trit_adder.py`, 9/9 matches):

```
A + B  | nN nS | (d, c)  | expected | OK
-1 -1  |  0  2 | (+1,-1) | (+1,-1)  | OK
-1  0  |  0  1 | (-1, 0) | (-1, 0)  | OK
-1 +1  |  0  0 | ( 0, 0) | ( 0, 0)  | OK
 0 -1  |  0  1 | (-1, 0) | (-1, 0)  | OK
 0  0  |  0  0 | ( 0, 0) | ( 0, 0)  | OK
 0 +1  |  1  0 | (+1, 0) | (+1, 0)  | OK
+1 -1  |  0  0 | ( 0, 0) | ( 0, 0)  | OK
+1  0  |  1  0 | (+1, 0) | (+1, 0)  | OK
+1 +1  |  2  0 | (-1,+1) | (-1,+1)  | OK
```

### 6.4 Why this is surprising

The two "overflow" cases `(+1, +1)` and `(-1, -1)` work **not** because
of a cleverly designed interaction, but because two same-sign gliders
simply do not annihilate (no collision on their respective lanes), so
they survive and the readout `count ≥ 2` naturally encodes the
overflow. Balanced ternary is the *only* integer representation in which
this linear readout works cleanly (cf. standard binary, where two `+1`s
would have to *cancel* and emit a carry, which a linear count cannot do).

The rule was not designed for ternary arithmetic; ternary arithmetic is
latent in the rule.

## 7. Multi-trit ALU with free subtraction

### 7.1 Trit-full-adder

With a carry input `cin ∈ {-1,0,+1}`, a full trit adder `(a, b, cin) → (d, cout)`
satisfying `a + b + cin = 3·cout + d` is realised as a **two-stage cascade** of
the native single-collision primitive:

```
(d1, c1) = CA_add(a, b)
(d2, c2) = CA_add(d1, cin)
digit_out = d2
carry_out = c1 + c2
```

Because `|a+b+cin| ≤ 3`, the sum `c1 + c2` is provably always in `{-1, 0, +1}`
(no secondary overflow), so the cascade closes without additional normalisation.
Each full-trit-adder therefore costs exactly **two CA collisions**.

### 7.2 Ripple-trit adder and subtraction

N-trit balanced-ternary addition is an N-deep chain of trit-full-adders. We
verified 8 diverse cases at N=4 (range ±40), including carry-propagating,
mixed-sign, cancellation (`+21 + -21 = 0`), and overflow (`+40 + +1 = +41`,
which requires the 5th trit). All results match integer arithmetic.

**Free subtraction.** In balanced ternary, integer negation is a *trit-wise
sign flip* — no two's-complement, no borrow logic. Therefore

`A − B = A + negate(B)`,

where `negate` has zero computational cost (it is relabelling, not computation).
We verified 4 subtraction cases through the same CA pipeline with only the
input encoding flipped. All correct.

This reproduces, in a sparse 2D CA, the core property that made balanced
ternary appealing historically (Setun computer, 1958): uniform handling of
positive and negative numbers, and subtraction for free.

### 7.3 Gate count comparison (N-trit vs N-bit)

For an N-digit adder:

- Binary ripple-carry: each full-adder ≈ 5 gate-level CA simulations
  (2 XOR + 2 AND + 1 OR), so total ≈ 5N simulations.
- Balanced-ternary ripple-trit: each full-trit-adder = 2 CA simulations.
  Total ≈ 2N simulations.

For equivalent numeric range `3^N ≈ 2^(1.585 N)`, i.e. 1 trit ≈ 1.585 bits,
so `N_bits ≈ 1.585 N_trits`. The CA-simulation cost ratio is therefore
`2N / (5 · 1.585 N) ≈ 0.25` — **the trit pipeline is ≈4× cheaper** per
unit of representable range, at the simulation level. This is not a claim
about physical hardware; it is a statement about how many CA collisions the
rule needs to realise the operation.

## 8. Implications and limits

**What this shows.** A λ ≈ 0.059 2D CA can host both (a) Boolean-complete
gate logic by composition, and (b) balanced-ternary single-digit arithmetic
natively. These are independent results: (a) does not imply (b) and
vice-versa.

**What it does not show.** We do not claim Turing-completeness via explicit
TM simulation (only NAND-completeness, which is weaker in the finite-memory
regime). We do not claim the rule is minimal. We do not claim native
multi-trit arithmetic — carry propagation between trits would still require
composition, though the natural encoding makes each stage a single collision.

**Potential directions.** Native SIMD-style ternary ALUs; physically-
motivated ternary interconnect (balanced ternary has known asymptotic
benefits in carry propagation); fault-tolerant variants.

## 9. Reproducibility

All results in this draft are backed by scripts in the repository:

| Claim | Script |
|-------|--------|
| INHIBIT primitive | `verify_inhibit.py` |
| AND gate | `verify_and.py` + `and_circuit_verified.gif` |
| NOR gate | `verify_nor.py` |
| Half-adder | `verify_half_adder.py` + `half_adder_verified.gif` |
| Full-adder | `verify_full_adder.py` |
| NAND + XOR cascade | `verify_nand.py` |
| 4-bit adder | `multi_bit_adder.py` + `adder_4bit_11plus5.gif` |
| **Trit-adder (native)** | `verify_trit_adder.py` + `trit_adder_9cases.gif` |
| **Multi-trit ALU + sub** | `multi_trit_adder.py` + `multi_trit_alu.gif` |

Rule table, glider patterns, and simulator are in `src/tca_sim/`.

---

*End of draft.*

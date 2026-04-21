# TCA_Explorer — a sparse 2D balanced-ternary cellular automaton

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19684656.svg)](https://doi.org/10.5281/zenodo.19684656)

A 2D outer-totalistic CA over `{-1, 0, +1}` on a Moore neighborhood with toroidal
boundaries. The rule has only **3 active entries out of 51** (λ ≈ 0.059), yet
supports:

1. **Four axis-aligned gliders** (N, S, E, W) of velocity `c` and period 1.
2. **Boolean universality** via NAND and NOR gates built from glider collisions.
3. **Native single-collision balanced-ternary trit-adder** — all 9 single-digit
   cases resolved without gate composition.
4. **Multi-trit ALU** with carry propagation and free subtraction (balanced-
   ternary negation is a trit-wise sign flip, cost 0).

See `PAPER_DRAFT.md` for the full write-up.

## Repository layout

```
src/tca_sim/              simulator + canonicalization
verify_*.py               per-primitive empirical verifications
multi_bit_adder.py        4-bit ripple-carry adder (binary, composed)
multi_trit_adder.py       N-trit ALU (balanced-ternary, native)
gen_*.py                  GIF generators for paper figures
tests/                    canonicalization/metric unit tests
archive/                  earlier exploration, non-reproducible artifacts
```

## Reproducibility map

| Paper claim | Script | Artifact |
|---|---|---|
| INHIBIT primitive (E+N perpendicular) | `verify_inhibit.py` | — |
| AND gate via cascade | `verify_and.py` · `gen_and_circuit_gif.py` | `and_circuit_verified.gif` |
| NOR gate (functional completeness) | `verify_nor.py` | — |
| NAND + XOR depth-4 cascade | `verify_nand.py` | — |
| Half-adder | `verify_half_adder.py` · `gen_half_adder_gif.py` | `half_adder_verified.gif` |
| Full-adder (8/8 truth table) | `verify_full_adder.py` | — |
| 4-bit ripple-carry adder | `multi_bit_adder.py` · `gen_4bit_adder_gif.py` | `adder_4bit_11plus5.gif` |
| **Trit-adder (9/9 native)** | `verify_trit_adder.py` · `gen_trit_adder_gif.py` | `trit_adder_9cases.gif` |
| **Multi-trit ALU + free subtraction** | `multi_trit_adder.py` · `gen_multi_trit_alu_gif.py` | `multi_trit_alu.gif` |

Every script is self-contained — run directly with `python <script>.py` and it
prints a truth table, asserts correctness, and (for `gen_*.py`) writes a GIF.

## Quick start

```bash
pip install -r requirements.txt
python verify_trit_adder.py     # 9/9 balanced-ternary addition
python multi_trit_adder.py      # multi-trit ALU + subtraction
python -m pytest tests/ -q      # sanity checks
```

## The rule

Cells take values in `Σ = {-1, 0, +1}`. For state `σ` and Moore neighbor sum
`s`, the next state is looked up at index `(σ+1)·17 + (s+8)` in a 51-entry
table. All entries are 0 except:

| idx | (σ, s) | → |
|---|---|---|
| 9  | (-1,  0) | +1 |
| 23 | ( 0, -3) | -1 |
| 34 | (+1, -8) | +1 |

## Status

Pre-print draft. All numerical claims in `PAPER_DRAFT.md` reproduce directly
from the scripts in this repo.

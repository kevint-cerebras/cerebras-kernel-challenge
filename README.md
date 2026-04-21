# Cerebras Kernel Challenge — Top-K k-NN

## What you're building

A CSL kernel that, for a given query vector `q` and a sharded database
`D` (N rows, d dims, spread across a P×P grid of PEs), returns the `K`
rows of `D` closest to `q` in Euclidean distance, sorted ascending, with
deterministic tie-breaking by original index.

Trivial in NumPy (`reference.py` is ~10 lines). The interesting work is
mapping it onto the wafer: sharding, routing, packing `(dist, idx)`
pairs through collectives, surviving ties through a distributed
reduction, and fitting in ~48 KB of PE SRAM.

**Skim `SPEC.md` now** — that's the authoritative statement. This README
is the onramp.

## Time budget

72 hours from when you open this packet. Log your start time in
`DESIGN.md`.

## Before you write any CSL (≈30 min)

```bash
# 1. Toolchain check
cslc --help
cs_python -c 'import numpy; print(numpy.__version__)'

# 2. Eyeball the oracle so you know exactly what "correct" means
python3 reference.py
# Prints expected top-K for each of 6 deterministic test cases.

# 3. Compile and run ONE tutorial end-to-end before touching your own code
cd <sdk>/examples/benchmarks/gemv-collectives_2d
bash commands_wse2.sh
# If this prints SUCCESS, your toolchain is good. If not, fix this first.
```

## Recommended path through the SDK tutorials

You do not need to read all of them. This order is enough to build the
challenge:

| # | Tutorial | What you learn |
|---|---|---|
| 1 | `tutorials/gemv-00-basic-syntax` | CSL syntax, tasks, DSDs — 10 minutes |
| 2 | `tutorials/gemv-05-multiple-pes` | Multi-PE layout, memcpy H2D/D2H |
| 3 | `tutorials/topic-11-collectives` | scatter / broadcast / gather / reduce |
| 4 | `benchmarks/gemv-collectives_2d` | **Best template for a P×P kernel — start from this.** |
| 5 | `tutorials/topic-05-sentinels` | Needed only if you design a streaming reduce |

## Suggested implementation order

1. **Single-PE version first.** Compute distances + local top-K on one PE,
   memcpy result straight back. No routing. Gets you a working
   compile-run-verify loop.
2. **Go to P×P.** Shard `D`, broadcast `q` to every PE, have each PE
   compute its own local top-K. Still no reduction — have every PE
   memcpy its slice back to host; merge on host to prove correctness.
3. **Move the merge onto the wafer.** Gather local top-Ks to PE(0,0) and
   do the final selection there.
4. **Tighten.** Replace the scalar distance loop with DSD-based
   `@fsubs` / `@fmacs`. Instrument with timestamps, read `csdb`, tune.

Iterations 1–3 are about correctness, 4 is about perf. Do not try to do
them all at once.

## Gotchas that will eat a morning if you don't know them

- **Always pass `--memcpy --channels=1` to `cslc`.** The default is the
  deprecated CSELFRunner and will refuse to compile for SdkRuntime.
- **Task IDs and colors share a 32-slot namespace.** Collectives-2d
  reserves 10-13, memcpy reserves 21-23 and 27-30, system reserves
  29/31. That leaves 14-20, 24-26. Collision errors compile as
  "task ID N bound to more than one task" and point at a library file.
- **PE SRAM is ~48 KB.** Oversized per-PE buffers fail at *link*, not
  compile, with a `.data.hi range` error. The `K=256` edge case is
  deliberately tight — pick your buffer sizes carefully.
- **Collective callbacks fire on every PE in the group**, not just the
  destination. Your task state machine needs to handle "I'm not the
  root, but the callback just fired me too."
- **Gather preserves source order on the destination**, but your final
  selection still has to tie-break by the original global index that
  each pair carries as its second `u32`.

## Minimum viable submission

Everything from `src-starter/` plus:

- `layout.csl` and one or more PE program files of your design.
- `run.py` that reads test cases from `reference.py`, memcpys inputs,
  launches, reads back outputs, compares to the oracle, prints
  `PASS: <case>` on success.
- `commands.sh` that compiles *and* runs the `baseline` case on WSE-2.
- `DESIGN.md` — one page. See SPEC §5. **This is the most important
  artifact. We read it closely.** Candidates who used AI write good
  memos because they understood the output. Candidates who pasted
  without understanding cannot.

## Running the grader locally

```bash
cd tests
python3 -m pytest test_correctness.py -v --submission=../src-starter
```

For each of the 6 cases, the grader:
1. Compiles your `layout.csl` with that case's params.
2. Runs your `run.py --case <case>`.
3. Asserts the `PASS: <case>` marker appears in stdout.

The CI machine runs this same script. If it passes locally, it passes
grading.

## What "good" looks like

- **Correctness**: all 6 cases bit-identical indices, distances
  `allclose` within 1e-3. `all_equal` and `duplicates` are the
  tie-breaking tests — don't skip them.
- **Determinism**: run twice, same output.
- **Perf**: within the cycle budget in SPEC §2.5 (set at `1.3×` our
  reference).
- **DESIGN.md**: specific, honest, and short. Diagrams beat paragraphs.
  "What I'd do with 2× more time" is the section we weight highest.

## Using AI

Expected and encouraged. Every Cerebras engineer does. The challenge is
calibrated on the assumption you will — the things it tests (PE SRAM
budgets, routing topology, reading `csdb` output, carrying tie-break
keys through a reduction) are things AI can help with but not one-shot.

You will find that pasting `SPEC.md` verbatim into an LLM produces code
that fails to compile. That is on purpose and not a bug.

## Questions

Email `Daniel.kim@cerebras.net` and `Kevin.taylor@cerebras.net` — we
answer within 4 hours on business days. Spec clarifications: fine.
"How do I write CSL": that's what the tutorials are for.

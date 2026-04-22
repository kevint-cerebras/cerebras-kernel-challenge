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

SDK download: [Cerebras-SDK-1.4.0](http://dropbox.com/scl/fi/2lgy7q7rwg366wf9m9r1l/Cerebras-SDK-1.4.0-202505230211-4-d9070058.tar.gz?dl=0&e=1&rlkey=fk8egs6dop9n3prrgnmz5ji6u&st=8681pzzf)

The SDK runs only on Linux with Apptainer (or Singularity). Three ways
to get that environment:

- **Linux + Apptainer/Singularity** — native, fastest.
- **Mac or Windows + Docker** — build the image in `docker/` and work
  inside it locally. See `docker/README.md`.
- **Cloud Linux VM on Render** — sign up at render.com and redeem promo
  code `RENDER-CODETV` for $50 in free credits, which covers well over
  the 72-hour window on a small instance. Then install Apptainer and
  follow the Linux path.

If none of those work, ping your contact and a hosted VM will be
arranged.

```bash
# 1. Toolchain check
cslc --help
cs_python -c 'import numpy; print(numpy.__version__)'

# 2. Examine exactly what "correct" means
python3 reference.py
# Prints expected top-K for each of 6 deterministic test cases.

# 3. Compile and run ONE tutorial end-to-end before touching your own code
cd <sdk>/examples/benchmarks/gemv-collectives_2d
bash commands_wse2.sh
# If this prints SUCCESS, your toolchain is good. If not, fix this first.
```

## Minimum viable submission

Everything from `src-starter/` plus:

- `layout.csl` and one or more PE program files of your design.
- `run.py` that reads test cases from `reference.py`, memcpys inputs,
  launches, reads back outputs, compares to the oracle, prints
  `PASS: <case>` on success.
- `commands.sh` that compiles *and* runs the `baseline` case.
- `DESIGN.md` — one page. See SPEC §5. **This is the most important
  artifact. We read it closely.** Candidates who used AI write good
  memos because they understood the output. Candidates who pasted
  without understanding cannot. Understand your code!

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
  tie-breaking tests. Don't skip them.
- **Determinism**: run twice, same output.
- **Perf**: per-case cycle gate in SPEC §2.5 (1.3× our reference, per
  test case). Reference counts live in `cycles.json`; entries currently
  set to `null` are waived for your submission with no retroactive
  enforcement.
- **DESIGN.md**: specific, honest, and short. Diagrams beat paragraphs.

## Submission

Invite {your username} + `danielkim-cerebras` to your solution repo and send an email to us with the private github repo to submit.

## Questions

Email `Daniel.kim@cerebras.net` and `Kevin.taylor@cerebras.net` — we
answer within 4 hours on business days. Spec clarifications: fine.
"How do I write CSL": that's what the tutorials are for.

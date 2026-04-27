# Cerebras Kernel Challenge: Wafer-Scale Top-K k-NN

**Role fit:** kernel / HPC / ML-systems engineer
**Expected effort:** 1–3 days for a strong candidate
**Tools allowed:** any AI assistant, any documentation, any reference code
**What we're testing:** can you design, debug, and profile a non-trivial CSL kernel — not whether you memorized CSL syntax

---

## 1. Problem

Implement a kernel that computes the **top-K nearest neighbors** (by L2 distance) of a query vector against a sharded database, running on the Cerebras Wafer-Scale Engine.

### Inputs (provided by host)
- `q`: query vector, `fp32`, shape `(d,)`
- `D`: database matrix, `fp32`, shape `(N, d)`
- `K`: integer, number of neighbors to return
- `P`: integer, the kernel is configured for a `P × P` PE grid

### Output (returned to host)
- `indices`: `int32`, shape `(K,)` — indices into `D` of the `K` closest rows, sorted ascending by distance
- `distances`: `fp32`, shape `(K,)` — the corresponding L2 distances, sorted ascending

### Semantics
- Distance metric is **Euclidean** (L2). **Return squared L2 distances** (`||x − q||²`). The NumPy oracle in `reference.py` returns squared distances and the grader's `allclose` (atol=1e-3, rtol=1e-3) compares against them; returning Euclidean (sqrt-applied) distances will fail the tolerance check at any non-trivial scale.
- `indices[0]` must be the globally-nearest row.
- **Tie-breaking is required and deterministic**: if two rows have equal distance, the row with the smaller original index in `D` comes first.
- Output must be identical across runs (no nondeterminism from fabric arrival order).

---

## 2. Constraints

### Hard constraints (enforced by grader)
1. `D` is sharded across a `P × P` PE grid, each PE owning `ceil(N / P²)` rows. No host involvement between `q` broadcast and result readback.
2. Works for the **baseline workload**: `N=2048, d=32, K=16, P=4` on WSE-2. (A PE has ~48 KB of SRAM. The baseline is intentionally small so a single-pass design fits, but scaling `N` is a stretch goal: bigger grid, host streaming, or multi-pass reduction — document whatever path you take.)
3. Works for the **edge-case workloads**:
   - `K=1` (argmin)
   - `K=256` (large K)
   - `N` not divisible by `P²` (last PE is short)
   - All-equal rows (tie-breaking test)
   - Rows with exact duplicates at different indices
4. Deterministic output (run twice, bit-identical result).
5. **Performance budget**: For each of the 6 test cases, your kernel must satisfy `candidate_cycles ≤ 1.3 × reference_cycles`, measured by `csdb`. Per-case reference cycle counts live in `cycles.json` at the packet root and are authoritative — the grader reads them directly. If an entry is `null`, the gate is waived for that case for the duration of your submission; there is no retroactive enforcement once the reference is published. Failing the gate on a case loses that case's share of the perf points (see §4); it does not fail your submission overall.

### Soft constraints (graded qualitatively)
6. Memory footprint per PE should fit comfortably — no swapping to host between phases.
7. Kernel should compose cleanly: separate `layout.csl` and PE program file(s), parameterized by `N, d, K, P`.

---

## 3. Deliverables

Submit by inviting `kevint-cerebras` and `danielkim-cerebras` as collaborators on your **private** solution GitHub repo, then email both contacts with the repo link.

**Layout requirement:** `layout.csl` and `run.py` must be in the same directory at the level you point the grader at. `tests/test_correctness.py` is invoked with `--submission=<dir>` and asserts both files exist directly in `<dir>`. Recommended flat layout:

```
<your-repo>/
├── layout.csl                # required at this level
├── pe_program.csl            # or multiple PE program files
├── run.py                    # required at this level — host: memcpy D + q, launch, readback
├── commands.sh               # cslc + cs_python invocation for the baseline
└── DESIGN.md                 # 1-page design memo (see §5)
```

You do not need to re-vendor `reference.py`, `tests/`, or this `SPEC.md` — the grader brings its own copy. If you place CSL files under a subdirectory like `src/`, point the grader at that subdirectory and put `run.py` there too.

---

## 4. Grading (100 pts)

| Category | Points | What we look for |
|---|---|---|
| Correctness — baseline | 25 | Matches NumPy top-K exactly |
| Correctness — edge cases | 25 | All 5 edge cases pass |
| Determinism | 10 | Bit-identical across 3 runs |
| Performance — within budget | 20 | `csdb` cycles ≤ 1.3× reference per case (see `cycles.json`). Awarded proportionally across the 6 cases; cases with a `null` reference count as pass. |
| Design memo quality | 15 | See §5 |
| Code clarity | 5 | Readable, commented where non-obvious |

A submission that fails correctness on baseline scores 0 regardless of other categories. You can fail perf and still do well overall — we care that you *understand* where your cycles went.

---

## 5. Design memo (`DESIGN.md`)

Exactly **one page** (roughly 400–600 words). Must include:

1. **Routing topology.** A diagram or plain-text sketch. Which colors, which directions, which wavelets carry what payload.
2. **Local top-K algorithm** and why you picked it. Expected work per PE in Big-O terms, and an estimate of cycles per element for the dominant loop.
3. **Fabric bandwidth accounting.** How many wavelets cross each edge of the grid in the worst case? Is your bottleneck compute or routing?
4. **Tie-breaking argument.** A short proof that your merge reduction produces the same output regardless of the order wavelets arrive at a given PE.
5. **What you'd do with 2× more time.** One or two concrete improvements, with rough cycle payoff.

**This memo is the most important anti-plagiarism device in the challenge.** We read it closely. Candidates who used AI heavily but understood the output will write a good memo. Candidates who pasted without understanding cannot.

---

## 6. Scaffolding policy

You receive:
- This spec
- `reference.py` — NumPy oracle for top-K
- `tests/test_correctness.py` — runs your compiled kernel against the oracle
- The SDK's `csl-extras` tarball with all tutorials and benchmarks
- A pointer to read first: `tutorials/gemv-05-multiple-pes`, `tutorials/topic-11-collectives`, `tutorials/topic-05-sentinels`, `tutorials/topic-08-filters`, and `benchmarks/histogram-torus/README.rst` (for wavelet encoding patterns)

You do **not** receive:
- A reference top-K implementation
- A skeleton that already has routing set up

---

## 7. Rules

- You may use any AI assistant. We encourage it.
- You must run your own code. We will compile and run your submission on a fresh machine; if it doesn't run, it doesn't score.
- You may not share the challenge externally or seek help from current Cerebras employees.
- Time-boxed at 72 hours from the moment you open the challenge. Log start time in `README.md`.

---

## 8. Why this is designed to resist "paste the spec into a chatbot"

Because CSL is a small-corpus language and because this problem does not appear verbatim in any public tutorial or benchmark, an AI assistant cannot produce a working one-shot. A candidate using AI will have to:

- Read tutorials to understand fabric routing semantics
- Compose patterns from four different tutorials into one kernel
- Run the compiler, read `csdb` output, and iterate on cycle counts
- Debug tie-breaking that only fails on inputs with collisions
- Defend their routing and algorithm choices in a memo

We read the memo last and the code first. If the code works but the memo is vacuous, we follow up in the onsite with pointed architectural questions. If the candidate cannot answer those, the signal is clear.

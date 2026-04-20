# Cerebras Kernel Challenge — Candidate Packet

Welcome. Read `SPEC.md` first. Everything else in this directory is support
material.

## What's in this packet

```
candidate-packet/
├── SPEC.md                       # the challenge itself — READ FIRST
├── reference.py                  # NumPy oracle + deterministic test cases
├── tests/
│   └── test_correctness.py       # pytest-style grader that will be run on
│                                 # your submission
├── src-starter/                  # empty directory — put your CSL here
│   └── README.md                 # pointers to SDK tutorials
└── README.md                     # this file
```

## Getting set up (30 minutes)

You need a Linux machine with Singularity/Apptainer installed and the
Cerebras SDK 1.4.0 tarball. If you don't have a working SDK install, ping
your contact and one will be arranged.

```bash
# verify toolchain
cslc --help              # should print usage and exit 0
cs_python -c 'import numpy; print(numpy.__version__)'

# verify the oracle (no SDK needed; just numpy)
python3 reference.py
```

## Minimum viable submission

1. `src-starter/layout.csl` and PE program file(s) of your design.
2. `run.py` that loads test cases from `reference.py`, memcpys to device,
   launches, reads back, compares against oracle.
3. `commands.sh` that compiles **and** runs the baseline.
4. `DESIGN.md` — one page, see SPEC §5. **Do not skip this.**

## Running the grader against your submission

```bash
cd tests
python3 -m pytest test_correctness.py -v --submission=../src-starter
```

The grader compiles with your `commands.sh` params for each of the 6 test
cases and diffs output against the oracle. It also measures wall-clock time
as a crude perf sanity check (real grading uses `csdb` cycle counts).

## What we look for

- **All 6 cases pass bit-identical indices and distances `allclose` within 1e-3.**
  The `all_equal` and `duplicates` cases specifically test tie-breaking.
- **Determinism**: run twice, same output.
- **One-page DESIGN.md** with the five sections in SPEC §5.

## Time budget

72 hours from when you open this packet. Log your start time in your
`DESIGN.md`. We do not track keystrokes; the honor system applies.

## Questions

Email `<hiring-manager>@cerebras.net` — we answer within 4 hours on
business days. Questions about clarifying the spec are fine; questions
about how to write CSL are not (that's what the SDK tutorials are for).

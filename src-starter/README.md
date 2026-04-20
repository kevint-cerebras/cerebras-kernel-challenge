# Start here

Your CSL goes in this directory. Suggested starting reading order:

1. `examples/tutorials/gemv-00-basic-syntax` — CSL in ~100 lines
2. `examples/tutorials/gemv-05-multiple-pes` — multi-PE layout + memcpy
3. `examples/tutorials/gemv-06-routes-1` through `gemv-08-routes-3` — routing
4. `examples/tutorials/topic-11-collectives` — scatter / broadcast / gather / reduce
5. `examples/tutorials/topic-05-sentinels` — marking end-of-stream
6. `examples/benchmarks/gemv-collectives_2d` — best template for a P×P grid
7. `examples/benchmarks/histogram-torus` — wavelet bit-packing and
   termination detection, if you end up needing those

Common pitfalls (not a full list, just the ones we've seen):

- **Task IDs live in a 32-slot namespace shared with colors.** Many IDs are
  reserved by memcpy and collectives. Check the color-map comment in
  `topic-11-collectives/layout.csl` before picking task IDs.
- **PE SRAM is ~48 KB.** Oversized per-PE buffers fail at link, not compile.
- **`cslc` invocations need `--memcpy --channels=1`** for SdkRuntime. The
  default is the deprecated CSELFRunner and will reject your compile.
- **Collective callbacks fire on every PE in the group**, not just the
  destination. Structure your task state machine accordingly.

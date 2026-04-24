# Docker image for Mac / Windows candidates

The Cerebras SDK runs only on Linux with Apptainer (or Singularity) installed.
This directory packages both into a Docker image so you can run the full
toolchain on any host that runs Docker (macOS, Windows, or Linux).

If you already have Linux + Apptainer, you don't need this. Follow the main
`../README.md` instead.

## Build (once, ~10 minutes, ~3 GB image)

1. Download the SDK tarball using the link in `../README.md` and extract it
   into this directory as `./sdk`:

   ```bash
   # Run from this docker/ directory.
   cp ~/Downloads/Cerebras-SDK-1.4.0-*.tar.gz .
   tar xzf Cerebras-SDK-1.4.0-*.tar.gz
   mv Cerebras-SDK-1.4.0-* sdk
   ```

   Result: `./sdk/cslc`, `./sdk/cs_python`, `./sdk/sdk-cbcore-*.sif`, etc.

2. Build:

   ```bash
   docker build --platform=linux/amd64 -t topk-knn-sdk .
   ```

   `--platform=linux/amd64` is required: the SDK is x86_64-only. On Intel Macs,
   Linux x86 hosts, and Windows (Docker Desktop) this runs natively. On Apple
   Silicon it runs under emulation — see the Apple Silicon note below.

## Apple Silicon (M1/M2/M3/M4): turn Rosetta off

The Cerebras compiler uses AVX2 instructions. Docker Desktop's default
Rosetta 2 emulator does **not** translate AVX2, so the compiler crashes with
`Illegal instruction` (exit code 132). Fix:

1. Open Docker Desktop → Settings → General.
2. Turn OFF "Use Rosetta for x86_64/amd64 emulation on Apple Silicon".
3. Apply & Restart.

Docker falls back to QEMU, which supports the full x86 instruction set but is
roughly 3–5× slower than Rosetta. Expect simulator runs that take ~10 s on
native x86 to take ~30–50 s in this configuration. Functional, not fast.

If you'd rather not trade speed for compatibility, ask about a hosted Linux
VM instead.

## Use

From your submission directory (containing `layout.csl`, `run.py`, …):

```bash
docker run --rm -it --platform=linux/amd64 --privileged \
  -v "$PWD:/work" \
  topk-knn-sdk
```

Inside the container, run `cslc`, `cs_python`, `csdb`, and `pytest` as
documented in the main README and SPEC.

`--privileged` is needed for Apptainer's kernel features (namespaces, overlay,
fuse mounts) inside Docker. If your Docker host has user-namespace remapping
configured, you can drop it to `--cap-add SYS_ADMIN --device /dev/fuse`, but
most Docker Desktop setups just want `--privileged`.

## Smoke test

```bash
docker run --rm --platform=linux/amd64 --privileged topk-knn-sdk cslc --help
```

Should print `cslc` usage and exit 0. If you see `Illegal instruction` or
`EXIT=132` on an Apple Silicon Mac, revisit the Rosetta toggle above. If it
errors with "SIF not found" or "singularity not in \$PATH", the image is
wrong — rebuild.

## Cleanup

```bash
docker rmi topk-knn-sdk
rm -rf sdk Cerebras-SDK-1.4.0-*.tar.gz
```

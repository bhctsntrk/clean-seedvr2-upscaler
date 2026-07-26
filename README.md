# Clean SeedVR2 Upscaler

Run [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) locally with one Python script, batch support, hardware preflight, and a choice between disposable or reusable dependencies.

```text
uv run seedvr2_upscale.py INPUT OUTPUT OUTPUT_RES [OPTIONS]
```

The wrapper supports:

- Windows with an NVIDIA GPU
- Linux with an NVIDIA GPU
- Apple Silicon macOS with PyTorch MPS

Unit tests run on Windows, Ubuntu, and macOS in GitHub Actions. End-to-end GPU inference has been tested on Windows with a supported NVIDIA GPU. The macOS and Linux paths should be treated as supported but hardware-dependent until they receive broader real-GPU testing.

## Quick start

Check the machine before downloading anything:

```text
uv run seedvr2_upscale.py --preflight-only
```

Upscale one image to exact wallpaper dimensions:

```text
uv run seedvr2_upscale.py image.png image-2560x1440.png 2560x1440
```

Upscale every supported image in a folder with one model load:

```text
uv run seedvr2_upscale.py input output 2560x1440
```

`OUTPUT_RES` accepts:

- `1440`: set the short side to 1440 pixels and preserve aspect ratio.
- `2560x1440`: create that exact size using a centered fill crop after SeedVR2.

Supported inputs: PNG, JPEG, BMP, TIFF, and WebP. Outputs are PNG.

## Cache modes

The default is still the clean, leave-no-clutter behavior:

```text
uv run seedvr2_upscale.py input output 2560x1440 --cache-mode clean
```

`clean` creates a cryptographically owned private session and removes its SeedVR2 source, Python environment, models, package caches, task caches, and staging files on success or ordinary failure.

For repeated use, retain downloads between runs:

```text
uv run seedvr2_upscale.py input output 2560x1440 --cache-mode reuse
```

`reuse` keeps the source, versioned Python environment, model files, and package caches. Output staging and process-temporary data still live in a separate private session that is safely removed after each run.

Default persistent cache roots:

- Windows: `%LOCALAPPDATA%\clean-seedvr2-upscaler`
- Linux: `$XDG_CACHE_HOME/clean-seedvr2-upscaler` or `~/.cache/clean-seedvr2-upscaler`
- macOS: `~/Library/Caches/clean-seedvr2-upscaler`

Choose another persistent cache root:

```text
uv run seedvr2_upscale.py input output 2560x1440 \
  --cache-mode reuse --cache-dir /mnt/fast/seedvr2-cache
```

On PowerShell, the same command can be written on one line or continued with backticks. `--cache-dir` is accepted only with `--cache-mode reuse`.

The wrapper never automatically deletes or prunes a persistent cache. For concurrent runs, use a different `--cache-dir` for each process.

## Persistent model location

Keep only model weights at a location you control while allowing the rest of the environment to remain disposable:

```text
uv run seedvr2_upscale.py input output 2560x1440 \
  --model-dir /mnt/models/seedvr2
```

An explicit `--model-dir` is always persistent and is never a cleanup target, even in `--cache-mode clean`. In `reuse` mode without this option, models are stored below the persistent cache root.

## Temporary work location

Select the parent directory for the private per-run staging session:

```text
uv run seedvr2_upscale.py input output 2560x1440 --temp-dir /mnt/fast/tmp
```

The `SEEDVR2_TEMP_ROOT` environment variable remains supported as a fallback:

```powershell
$env:SEEDVR2_TEMP_ROOT = "E:\AI-Temp"
uv run seedvr2_upscale.py .\input .\output 2560x1440
```

The script creates a random `seedvr2-batch-*` child inside the selected parent. It never deletes the selected parent or neighboring files.

## Hardware preflight

Before creating a session or downloading dependencies, the wrapper checks:

- supported OS and accelerator backend
- selected NVIDIA GPU index on Windows/Linux
- Apple Silicon and MPS eligibility on macOS
- logical CPU count
- physical RAM
- GPU VRAM or Apple unified memory
- free space on every work/cache/model filesystem

Default refusal thresholds:

- 4 logical CPU cores
- 16 GiB system RAM
- 8 GiB VRAM or unified memory
- 20 GiB free work/cache disk

Run only the check:

```text
uv run seedvr2_upscale.py --preflight-only
```

Choose GPU 1 and stricter limits:

```text
uv run seedvr2_upscale.py input output 2560x1440 \
  --device 1 --min-ram-gb 32 --min-vram-gb 12 --min-disk-gb 30
```

Set an individual limit to `0` to disable that refusal. `--skip-preflight` bypasses all resource checks and is intended only for users who knowingly accept installation failures, out-of-memory errors, or very slow execution.

## What each run guarantees

- Accepts one image or a whole folder.
- Uses a pinned, tested SeedVR2 revision.
- Uses Python 3.12 in an isolated environment managed by `uv`.
- Installs CUDA PyTorch on Windows/Linux or MPS PyTorch on Apple Silicon macOS.
- Reuses the loaded DiT and VAE across every image in a folder batch.
- Produces PNG files and optionally normalizes them to exact dimensions.
- Refuses to overwrite existing outputs.
- Generates inside private staging and publishes only validated outputs.
- Never deletes a user-selected output, persistent cache, or model directory.

## Cleanup boundaries

The disposable session is protected by a per-run cryptographic ownership marker. Before removal, the wrapper confirms the original resolved parent, atomically quarantines the session, verifies filesystem identity, builds a no-follow deletion manifest, and rechecks every entry immediately before removal. Links and Windows junctions are removed as links and are never traversed.

Cleanup is deliberately fail-closed: if ownership, containment, filesystem type, volume/mount boundaries, or identity cannot be proven, the script raises an error and leaves the quarantined data in place. It never calls `shutil.rmtree`, never deletes the selected temporary parent, and never treats a matching directory name alone as proof of ownership.

Normal failures and `Ctrl+C` run cleanup. An abrupt power loss or forced process kill cannot execute Python's `finally` block; in that case, a clearly named `seedvr2-batch-*` directory may remain under the selected temporary parent.

Output publication is stricter still: generated images remain inside the private session until validation succeeds. Publishing uses a new exclusive partial file and an atomic no-overwrite operation. An unusual filesystem interruption may leave a clearly named `.clean-seedvr2-*.partial` file instead of risking deletion of user data.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository:

```text
git clone https://github.com/bhctsntrk/clean-seedvr2-upscaler.git
cd clean-seedvr2-upscaler
```

No system Python, Git checkout of SeedVR2, CUDA Toolkit, manual model download, or manually managed virtual environment is required. Internet access is required on the first `reuse` run and on every `clean` run. A pre-populated `--model-dir` avoids downloading the model weights again, but a clean run still downloads its disposable source and dependencies.

## Reproducibility and upstream

The wrapper pins upstream SeedVR2 commit [`4490bd1`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/commit/4490bd1f482e026674543386bb2a4d176da245b9), which identifies itself as CLI v2.5.24. SeedVR2 is maintained separately and licensed under Apache-2.0. This repository contains only the wrapper; it does not redistribute SeedVR2 or model weights.

## License

The wrapper is released under the [MIT License](LICENSE). Upstream code and downloaded model assets retain their own licenses and terms.

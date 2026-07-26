# Clean SeedVR2 Upscaler

Run [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) locally with one Python script, three arguments, batch support, and no persistent model or virtual-environment clutter.

```powershell
uv run seedvr2_upscale.py INPUT OUTPUT OUTPUT_RES
```

The wrapper is Windows-first and tuned for an NVIDIA GPU with limited VRAM. It was tested on Windows 11 with an RTX 3060 12 GB.

## What it does

- Accepts one image or a whole folder.
- Downloads a pinned, tested SeedVR2 revision into an isolated session directory.
- Creates a temporary Python 3.12 environment with CUDA PyTorch and SeedVR2 dependencies.
- Downloads the 3B FP8 model and VAE once per invocation.
- Reuses the loaded DiT and VAE across every image in a folder batch.
- Produces PNG files and optionally normalizes them to exact dimensions.
- Refuses to overwrite existing outputs.
- Removes partial outputs after a failed run.
- Deletes its venv, model weights, source tree, package caches, and task caches on success or failure.

The script only removes the uniquely named `seedvr2-batch-*` directory that it created. It does not accept an arbitrary cache or cleanup path.

## Requirements

- Windows 10 or 11
- NVIDIA GPU and a current driver
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access for each invocation
- Roughly 15–20 GB of temporary free disk space

No separate Python, Git, CUDA Toolkit, model, or manual virtual environment is required. `uv` obtains Python 3.12 when necessary, and every task-owned download is redirected into the disposable session directory.

## Usage

Clone the repository:

```powershell
git clone https://github.com/bhctsntrk/clean-seedvr2-upscaler.git
cd clean-seedvr2-upscaler
```

Upscale one image to exact wallpaper dimensions:

```powershell
uv run seedvr2_upscale.py image.png image-2560x1440.png 2560x1440
```

Upscale every supported image in a folder with one model load:

```powershell
uv run seedvr2_upscale.py .\input .\output 2560x1440
```

Preserve aspect ratio and set only the short side:

```powershell
uv run seedvr2_upscale.py image.jpg upscaled.png 1440
```

`OUTPUT_RES` accepts:

- `1440`: makes the short side 1440 pixels and preserves the original aspect ratio.
- `2560x1440`: creates that exact size using a centered fill crop after SeedVR2.

Supported inputs: PNG, JPEG, BMP, TIFF, and WebP. Outputs are PNG.

## Put temporary data on another drive

Every invocation downloads several gigabytes because persistent caching is intentionally disabled. To keep temporary data off the system drive, select a parent directory:

```powershell
$env:SEEDVR2_TEMP_ROOT = "D:\Temp"
uv run seedvr2_upscale.py .\input .\output 2560x1440
```

The script creates a random `seedvr2-batch-*` child inside that directory and removes only that child when finished.

## Cleanup boundaries

The following task-owned locations are redirected into the disposable session:

- uv cache and uv-managed Python
- pip cache
- Hugging Face cache
- Torch and TorchInductor caches
- Triton and CUDA caches
- process temporary files
- SeedVR2 source, venv, models, and dependencies

Normal failures and `Ctrl+C` run cleanup. An abrupt power loss or forced process kill cannot execute Python's `finally` block; in that case, a clearly named `seedvr2-batch-*` directory may remain under the system temp directory or `SEEDVR2_TEMP_ROOT`.

## Reproducibility and upstream

The wrapper pins upstream SeedVR2 commit [`4490bd1`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/commit/4490bd1f482e026674543386bb2a4d176da245b9), which identifies itself as CLI v2.5.24. SeedVR2 is maintained separately and licensed under Apache-2.0. This repository contains only the wrapper; it does not redistribute SeedVR2 or its model weights.

## License

The wrapper is released under the [MIT License](LICENSE). Upstream code and downloaded model assets retain their own licenses and terms.

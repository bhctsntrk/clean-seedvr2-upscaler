"""Three-argument, self-cleaning SeedVR2 image upscaler.

Usage:
    uv run seedvr2_upscale.py INPUT OUTPUT OUTPUT_RES

Examples:
    uv run seedvr2_upscale.py image.png image-4k.png 2560x1440
    uv run seedvr2_upscale.py input-folder output-folder 2560x1440
    uv run seedvr2_upscale.py image.png output-folder 1440

OUTPUT_RES accepts either:
    1440        Target short side; preserves the source aspect ratio.
    2560x1440   Exact size; applies a centered fill crop after SeedVR2.

Each invocation creates an isolated temporary SeedVR2/CUDA environment. A whole
input directory is processed in one batch, then the environment, model weights,
and package caches are deleted automatically. Successful runs publish only
validated outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


# SeedVR2 prints Unicode status icons. Force UTF-8 so a redirected Windows
# cp1252 console cannot crash inference while writing progress messages.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


SEEDVR2_REPOSITORY = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler"
SEEDVR2_REVISION = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR2_ARCHIVE = f"{SEEDVR2_REPOSITORY}/archive/{SEEDVR2_REVISION}.zip"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/nightly/cu130"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WINDOWS_LATE_EXIT_CODES = {0xC0000409, -1073740791}
SESSION_PREFIX = "seedvr2-batch-"
QUARANTINE_PREFIX = ".clean-seedvr2-delete-"
SESSION_MARKER = ".clean-seedvr2-session"
SESSION_MAGIC = "clean-seedvr2-upscaler-owned-session-v1"


@dataclass(frozen=True)
class Resolution:
    short_side: int
    exact_size: tuple[int, int] | None


@dataclass(frozen=True)
class TemporaryEnvironment:
    cache: Path
    root: Path
    parent: Path
    token: str


@dataclass(frozen=True)
class CleanupEntry:
    path: Path
    identity: os.stat_result
    kind: str


def parse_resolution(value: str) -> Resolution:
    normalized = value.strip().lower().replace("×", "x")
    if normalized.isdigit():
        short_side = int(normalized)
        if short_side < 64:
            raise argparse.ArgumentTypeError("output_res must be at least 64 pixels")
        return Resolution(short_side=short_side, exact_size=None)

    match = re.fullmatch(r"(\d+)x(\d+)", normalized)
    if not match:
        raise argparse.ArgumentTypeError(
            "output_res must be a short side such as 1440 or exact dimensions "
            "such as 2560x1440"
        )

    width, height = (int(part) for part in match.groups())
    if min(width, height) < 64:
        raise argparse.ArgumentTypeError("both output dimensions must be at least 64 pixels")
    return Resolution(short_side=min(width, height), exact_size=(width, height))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale an image or folder with SeedVR2 using three arguments."
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("output", type=Path, help="Output PNG or directory")
    parser.add_argument(
        "output_res",
        type=parse_resolution,
        help="Short side (1440) or exact dimensions (2560x1440)",
    )
    return parser.parse_args()


def find_uv() -> Path:
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)

    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            [
                Path.home() / ".local" / "bin" / "uv.exe",
                Path.home()
                / "AppData"
                / "Local"
                / "Programs"
                / "Python"
                / "Python312"
                / "Scripts"
                / "uv.exe",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("uv is required but was not found on PATH")


def run(command: list[os.PathLike[str] | str], cwd: Path | None = None) -> None:
    values = [os.fspath(part) for part in command]
    print(f"\n> {subprocess.list2cmdline(values)}", flush=True)
    subprocess.run(values, cwd=cwd, check=True)


def create_temporary_environment() -> TemporaryEnvironment:
    """Create an owned session directory and redirect task caches into it."""
    parent = Path(
        os.environ.get("SEEDVR2_TEMP_ROOT", tempfile.gettempdir())
    ).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    cleanup_root = Path(tempfile.mkdtemp(prefix=SESSION_PREFIX, dir=parent))
    cache = cleanup_root / "runtime"
    session_temp = cleanup_root / "tmp"
    token = secrets.token_hex(32)
    cache.mkdir()
    session_temp.mkdir()
    marker = cleanup_root / SESSION_MARKER
    marker.write_text(
        f"{SESSION_MAGIC}\n{token}\n{cleanup_root}\n",
        encoding="utf-8",
    )

    redirected = {
        "UV_CACHE_DIR": cleanup_root / "uv-cache",
        "UV_PYTHON_INSTALL_DIR": cleanup_root / "uv-python",
        "PIP_CACHE_DIR": cleanup_root / "pip-cache",
        "HF_HOME": cleanup_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cleanup_root / "huggingface" / "hub",
        "TORCH_HOME": cleanup_root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cleanup_root / "torchinductor",
        "TRITON_CACHE_DIR": cleanup_root / "triton",
        "CUDA_CACHE_PATH": cleanup_root / "cuda",
        "XDG_CACHE_HOME": cleanup_root / "xdg-cache",
        "TMP": session_temp,
        "TEMP": session_temp,
        "TMPDIR": session_temp,
    }
    for name, path in redirected.items():
        os.environ[name] = os.fspath(path)
    return TemporaryEnvironment(
        cache=cache,
        root=cleanup_root,
        parent=parent,
        token=token,
    )


def cleanup_kind(info: os.stat_result) -> str:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or attributes & reparse_flag:
        return "link"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    if stat.S_ISREG(info.st_mode):
        return "file"
    return "special"


def is_filesystem_boundary(
    path: Path,
    identity: os.stat_result | None = None,
) -> bool:
    """Detect roots and mount/volume boundaries without pathlib.is_mount()."""
    parent = path.parent
    if parent == path:
        return True

    current = identity if identity is not None else os.lstat(path)
    parent_identity = os.lstat(parent)
    if current.st_dev != parent_identity.st_dev:
        return True

    # POSIX filesystem roots have the same inode as their parent. Windows
    # volume roots are covered by parent == path; junctions/reparse points are
    # classified as links by cleanup_kind() and are never traversed.
    return os.name != "nt" and current.st_ino == parent_identity.st_ino


def same_cleanup_identity(
    expected: os.stat_result,
    current: os.stat_result,
    kind: str,
) -> bool:
    if not os.path.samestat(expected, current):
        return False

    stable_fields = (
        "st_birthtime_ns",
        "st_file_attributes",
        "st_reparse_tag",
    )
    for field in stable_fields:
        if getattr(expected, field, None) != getattr(current, field, None):
            return False

    if stat.S_IFMT(expected.st_mode) != stat.S_IFMT(current.st_mode):
        return False

    # Removing children changes a directory's ctime, so directory identity
    # relies on the filesystem ID plus creation/reparse metadata. Files and
    # links can use the stronger mutation-sensitive fields as well.
    if kind != "directory":
        mutable_fields = ("st_ctime_ns", "st_mtime_ns", "st_size")
        for field in mutable_fields:
            if getattr(expected, field, None) != getattr(current, field, None):
                return False
    return True


def verify_cleanup_entry(entry: CleanupEntry) -> os.stat_result:
    current = os.lstat(entry.path)
    if not same_cleanup_identity(entry.identity, current, entry.kind):
        raise RuntimeError(f"Cleanup path identity changed: {entry.path}")
    if cleanup_kind(current) != entry.kind:
        raise RuntimeError(f"Cleanup path type changed: {entry.path}")
    return current


def build_cleanup_manifest(root: Path, marker: Path) -> list[CleanupEntry]:
    manifest: list[CleanupEntry] = []
    root_identity = os.lstat(root)

    def scan(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name.casefold())
        for item in entries:
            path = Path(item.path)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"Cleanup entry escaped its session root: {path}") from error
            if path == marker:
                continue

            # Windows DirEntry.stat() may report st_dev/st_ino as zero even
            # when os.lstat() exposes the stable filesystem identity. Use the
            # same identity API for both manifest creation and deletion checks.
            info = os.lstat(path)
            kind = cleanup_kind(info)
            if kind == "special":
                raise RuntimeError(f"Refusing to remove a special filesystem entry: {path}")
            if info.st_dev != root_identity.st_dev:
                raise RuntimeError(f"Refusing to cross a filesystem boundary: {path}")
            if kind == "directory":
                if is_filesystem_boundary(path, info):
                    raise RuntimeError(f"Refusing to cross a mounted filesystem: {path}")
                scan(path)
            manifest.append(CleanupEntry(path=path, identity=info, kind=kind))
    scan(root)
    return manifest


def remove_manifest_entry(entry: CleanupEntry) -> None:
    current = verify_cleanup_entry(entry)
    attributes = getattr(current, "st_file_attributes", 0)
    directory_flag = getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0)

    def remove() -> None:
        if entry.kind == "directory" or (
            entry.kind == "link" and os.name == "nt" and attributes & directory_flag
        ):
            os.rmdir(entry.path)
        else:
            os.unlink(entry.path)

    try:
        remove()
    except PermissionError:
        if entry.kind == "link":
            raise
        os.chmod(entry.path, current.st_mode | stat.S_IWRITE)
        changed_mode = os.lstat(entry.path)
        if not os.path.samestat(current, changed_mode):
            raise RuntimeError(f"Cleanup path identity changed after chmod: {entry.path}")
        remove()


def remove_quarantined_tree(
    root: Path,
    expected_root_identity: os.stat_result,
    marker: Path,
    expected_marker: str,
) -> None:
    root_identity = os.lstat(root)
    if not same_cleanup_identity(expected_root_identity, root_identity, "directory"):
        raise RuntimeError(f"Quarantined cleanup root identity changed: {root}")
    if cleanup_kind(root_identity) != "directory" or is_filesystem_boundary(
        root,
        root_identity,
    ):
        raise RuntimeError(f"Quarantined cleanup root is not a plain directory: {root}")

    marker_identity = os.lstat(marker)
    if cleanup_kind(marker_identity) != "file":
        raise RuntimeError(f"Cleanup ownership marker is not a regular file: {marker}")
    actual_marker = marker.read_text(encoding="utf-8")
    if not secrets.compare_digest(actual_marker, expected_marker):
        raise RuntimeError(f"Cleanup ownership marker did not match: {root}")

    manifest = build_cleanup_manifest(root, marker)
    for entry in manifest:
        remove_manifest_entry(entry)

    remove_manifest_entry(
        CleanupEntry(path=marker, identity=marker_identity, kind="file")
    )
    verify_cleanup_entry(
        CleanupEntry(path=root, identity=root_identity, kind="directory")
    )
    os.rmdir(root)


def remove_temporary_environment(session: TemporaryEnvironment) -> None:
    cleanup_root = session.root
    quarantine_container = session.parent / f"{QUARANTINE_PREFIX}{session.token}"
    quarantine = quarantine_container / "session"
    root_exists = os.path.lexists(cleanup_root)
    container_exists = os.path.lexists(quarantine_container)
    if not root_exists and not container_exists:
        return
    if root_exists and container_exists:
        raise RuntimeError("Both live and quarantined cleanup paths exist; refusing deletion")

    expected_marker = f"{SESSION_MAGIC}\n{session.token}\n{cleanup_root}\n"
    if root_exists:
        if cleanup_root.is_symlink() or (
            hasattr(cleanup_root, "is_junction") and cleanup_root.is_junction()
        ):
            raise RuntimeError(f"Refusing to remove a linked cleanup path: {cleanup_root}")
        resolved_root = cleanup_root.resolve()
        if resolved_root != cleanup_root or cleanup_root.parent != session.parent:
            raise RuntimeError(f"Refusing to remove an unsafe cleanup path: {cleanup_root}")
        if not cleanup_root.name.startswith(SESSION_PREFIX):
            raise RuntimeError(f"Refusing to remove an unsafe cleanup path: {cleanup_root}")

        marker = cleanup_root / SESSION_MARKER
        if marker.is_symlink() or not marker.is_file():
            raise RuntimeError(f"Refusing to remove an unowned cleanup path: {cleanup_root}")
        if not secrets.compare_digest(marker.read_text(encoding="utf-8"), expected_marker):
            raise RuntimeError(f"Cleanup ownership marker did not match: {cleanup_root}")

        root_identity = os.lstat(cleanup_root)
        quarantine_container.mkdir(mode=0o700)
        container_identity = os.lstat(quarantine_container)
        os.rename(cleanup_root, quarantine)
        moved_identity = os.lstat(quarantine)
        if not same_cleanup_identity(root_identity, moved_identity, "directory"):
            raise RuntimeError("Cleanup root identity changed during quarantine")
    else:
        container_identity = os.lstat(quarantine_container)
        if cleanup_kind(container_identity) != "directory":
            raise RuntimeError("Cleanup quarantine container is not a plain directory")
        if not os.path.lexists(quarantine):
            raise RuntimeError("Cleanup quarantine exists without its owned session")
        root_identity = os.lstat(quarantine)

    print(f"\nRemoving quarantined SeedVR2 environment: {quarantine}", flush=True)
    quarantined_marker = quarantine / SESSION_MARKER
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            if os.path.lexists(quarantine):
                remove_quarantined_tree(
                    quarantine,
                    root_identity,
                    quarantined_marker,
                    expected_marker,
                )
            current_container = os.lstat(quarantine_container)
            if not same_cleanup_identity(
                container_identity,
                current_container,
                "directory",
            ):
                raise RuntimeError("Cleanup quarantine container identity changed")
            if cleanup_kind(current_container) != "directory":
                raise RuntimeError("Cleanup quarantine container type changed")
            os.rmdir(quarantine_container)
            return
        except OSError as error:
            last_error = error
            if attempt < 4:
                time.sleep(1.0)

    raise RuntimeError(
        f"Temporary environment could not be fully removed: {last_error}"
    )


def ensure_source(cache_directory: Path) -> Path:
    source = cache_directory / "source"
    archive_path = cache_directory / "seedvr2.zip"
    extraction_root = cache_directory / "source.extracting"

    print(
        f"Downloading official SeedVR2 source at revision {SEEDVR2_REVISION[:12]}",
        flush=True,
    )
    request = urllib.request.Request(
        SEEDVR2_ARCHIVE,
        headers={"User-Agent": "clean-seedvr2-upscaler"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with archive_path.open("wb") as file:
            shutil.copyfileobj(response, file)

    extraction_root.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe path in SeedVR2 archive: {member.filename}")
        archive.extractall(extraction_root)

    candidates = list(extraction_root.glob("*/inference_cli.py"))
    if len(candidates) != 1:
        raise RuntimeError("Downloaded SeedVR2 archive has an unexpected layout")
    candidates[0].parent.replace(source)
    extraction_root.rmdir()
    archive_path.unlink()
    return source


def environment_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    digest.update(SEEDVR2_REVISION.encode("ascii"))
    digest.update(CUDA_WHEEL_INDEX.encode("utf-8"))
    for name in ("requirements.txt", "pyproject.toml"):
        path = source / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_environment(cache_directory: Path, source: Path, uv: Path) -> Path:
    venv = cache_directory / "venv"
    python = venv_python(venv)
    marker = venv / ".seedvr2-wrapper-ready"
    fingerprint = environment_fingerprint(source)

    if python.is_file() and marker.is_file() and marker.read_text().strip() == fingerprint:
        return python

    run([uv, "venv", "--python", "3.12", venv])
    print(
        "Installing CUDA PyTorch and SeedVR2 dependencies for this batch.",
        flush=True,
    )
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--pre",
            "torch",
            "torchvision",
            "--index-url",
            CUDA_WHEEL_INDEX,
        ]
    )
    run([uv, "pip", "install", "--python", python, "-r", source / "requirements.txt"])
    run(
        [
            python,
            "-c",
            (
                "import torch, cv2; "
                "assert torch.cuda.is_available(), 'CUDA-enabled PyTorch was not installed'; "
                "print('CUDA ready:', torch.cuda.get_device_name(0))"
            ),
        ]
    )
    marker.write_text(fingerprint, encoding="utf-8")
    return python


def input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise RuntimeError(f"Unsupported input image type: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise RuntimeError(f"No supported images found in: {input_path}")
        return images

    raise RuntimeError(f"Input does not exist: {input_path}")


def plan_outputs(
    input_path: Path,
    output_path: Path,
    images: list[Path],
) -> list[Path]:
    if input_path.is_file() and output_path.suffix:
        if output_path.suffix.lower() != ".png":
            raise RuntimeError("A single output file must use the .png extension")
        targets = [output_path]
    else:
        if output_path.exists() and not output_path.is_dir():
            raise RuntimeError(f"Output must be a directory: {output_path}")
        targets = [output_path / f"{image.stem}.png" for image in images]

    conflicts = [target for target in targets if os.path.lexists(target)]
    if conflicts:
        joined = "\n".join(f"  {path}" for path in conflicts[:10])
        raise RuntimeError(f"Refusing to overwrite existing output files:\n{joined}")
    return targets


def stage_outputs(
    session: TemporaryEnvironment,
    input_path: Path,
    final_targets: list[Path],
) -> tuple[Path, list[Path]]:
    directory = session.cache / "generated"
    directory.mkdir()
    staged_targets = [directory / target.name for target in final_targets]
    if input_path.is_file() and len(staged_targets) == 1:
        return staged_targets[0], staged_targets
    return directory, staged_targets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_outputs(
    staged_targets: list[Path],
    final_targets: list[Path],
    token: str,
) -> None:
    if len(staged_targets) != len(final_targets):
        raise RuntimeError("Internal error: staged and final output counts differ")

    conflicts = [target for target in final_targets if os.path.lexists(target)]
    if conflicts:
        raise RuntimeError(f"Output appeared during processing; refusing overwrite: {conflicts[0]}")

    for staged, final in zip(staged_targets, final_targets, strict=True):
        final.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(final):
            raise RuntimeError(f"Output appeared during processing; refusing overwrite: {final}")

        partial = final.with_name(f".{final.name}.clean-seedvr2-{token}.partial")
        if os.path.lexists(partial):
            raise RuntimeError(f"Refusing to replace an existing publish staging file: {partial}")

        expected_hash = sha256_file(staged)
        with staged.open("rb") as source, partial.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
            partial_identity = os.fstat(destination.fileno())

        if os.name == "nt":
            os.rename(partial, final)
            if not os.path.samestat(partial_identity, os.lstat(final)):
                raise RuntimeError(f"Published output identity changed: {final}")
        else:
            os.link(partial, final)
            if not os.path.samefile(partial, final):
                raise RuntimeError(f"Published output identity changed: {final}")
            if not os.path.samestat(partial_identity, os.lstat(partial)):
                raise RuntimeError(f"Publish staging identity changed: {partial}")
            os.unlink(partial)

        if sha256_file(final) != expected_hash:
            raise RuntimeError(f"Published output failed hash verification: {final}")


def resize_fill(
    paths: list[Path],
    target_size: tuple[int, int],
    python: Path,
) -> None:
    resize_program = r"""
import os
import sys
import cv2

target_width = int(sys.argv[1])
target_height = int(sys.argv[2])
target_ratio = target_width / target_height

for value in sys.argv[3:]:
    image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read generated image: {value}")
    source_height, source_width = image.shape[:2]
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        crop_width = round(source_height * target_ratio)
        left = max(0, (source_width - crop_width) // 2)
        cropped = image[:, left:left + crop_width]
    else:
        crop_height = round(source_width / target_ratio)
        top = max(0, (source_height - crop_height) // 2)
        cropped = image[top:top + crop_height, :]

    resized = cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    temporary = os.path.join(
        os.path.dirname(value),
        f".{os.path.basename(value)}.resizing.png",
    )
    if not cv2.imwrite(temporary, resized, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise RuntimeError(f"Cannot save resized image: {value}")
    os.replace(temporary, value)
"""
    run(
        [
            python,
            "-c",
            resize_program,
            str(target_size[0]),
            str(target_size[1]),
            *paths,
        ]
    )


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Output is not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def upscale(
    source: Path,
    cli_output: Path,
    targets: list[Path],
    resolution: Resolution,
    seedvr2_source: Path,
    python: Path,
) -> None:
    command: list[os.PathLike[str] | str] = [
        python,
        seedvr2_source / "inference_cli.py",
        source,
        "--output",
        cli_output,
        "--output_format",
        "png",
        "--resolution",
        str(resolution.short_side),
        "--cuda_device",
        "0",
        "--seed",
        "42",
        "--color_correction",
        "lab",
        "--input_noise_scale",
        "0",
        "--latent_noise_scale",
        "0",
        "--dit_offload_device",
        "cpu",
        "--vae_offload_device",
        "cpu",
        "--tensor_offload_device",
        "cpu",
        "--blocks_to_swap",
        "16",
        "--swap_io_components",
        "--vae_encode_tiled",
        "--vae_decode_tiled",
    ]

    if resolution.exact_size:
        command.extend(["--max_resolution", str(max(resolution.exact_size))])
    if source.is_dir():
        command.extend(["--cache_dit", "--cache_vae"])

    try:
        run(command, cwd=seedvr2_source)
    except subprocess.CalledProcessError as error:
        missing = [target for target in targets if not target.is_file()]
        if error.returncode not in WINDOWS_LATE_EXIT_CODES or missing:
            raise
        print(
            f"WARNING: SeedVR2 exited with {error.returncode} after writing all "
            "outputs; continuing with validation.",
            file=sys.stderr,
            flush=True,
        )

    missing = [target for target in targets if not target.is_file()]
    if missing:
        raise RuntimeError(f"SeedVR2 did not create the expected output: {missing[0]}")

    if resolution.exact_size:
        resize_fill(targets, resolution.exact_size, python)


def main() -> int:
    args = parse_arguments()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    images = input_images(source)

    if source.is_dir() and (output == source or source in output.parents):
        raise RuntimeError("For directory input, output must be outside the input directory")

    final_targets = plan_outputs(source, output, images)
    session = create_temporary_environment()
    try:
        uv = find_uv()
        seedvr2_source = ensure_source(session.cache)
        python = ensure_environment(session.cache, seedvr2_source, uv)
        cli_output, staged_targets = stage_outputs(session, source, final_targets)

        destination = args.output_res.exact_size or (
            f"short side {args.output_res.short_side}"
        )
        print(f"\nUpscaling {len(images)} image(s) to {destination}", flush=True)
        upscale(
            source,
            cli_output,
            staged_targets,
            args.output_res,
            seedvr2_source,
            python,
        )

        for target in staged_targets:
            width, height = png_size(target)
            if args.output_res.exact_size and (width, height) != args.output_res.exact_size:
                raise RuntimeError(
                    f"Generated output has wrong dimensions: {target} [{width}x{height}]"
                )
        publish_outputs(staged_targets, final_targets, session.token)

        print("\nCompleted:", flush=True)
        for target in final_targets:
            width, height = png_size(target)
            print(f"  {target}  [{width}x{height}]", flush=True)
        return 0
    finally:
        remove_temporary_environment(session)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

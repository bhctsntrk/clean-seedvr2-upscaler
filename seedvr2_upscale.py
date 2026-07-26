"""Portable SeedVR2 image upscaler with clean and reusable cache modes.

Usage:
    uv run seedvr2_upscale.py INPUT OUTPUT OUTPUT_RES

Examples:
    uv run seedvr2_upscale.py image.png image-4k.png 2560x1440
    uv run seedvr2_upscale.py input-folder output-folder 2560x1440
    uv run seedvr2_upscale.py image.png output-folder 1440

OUTPUT_RES accepts either:
    1440        Target short side; preserves the source aspect ratio.
    2560x1440   Exact size; applies a centered fill crop after SeedVR2.

By default each invocation creates and removes an isolated SeedVR2 environment.
Pass ``--cache-mode reuse`` to retain dependencies and models between runs.
Successful runs publish only validated outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
SUPPORTED_SYSTEMS = {"Windows", "Linux", "Darwin"}
DEFAULT_MIN_CPU_CORES = 4
DEFAULT_MIN_RAM_GIB = 16.0
DEFAULT_MIN_VRAM_GIB = 8.0
DEFAULT_MIN_DISK_GIB = 20.0
GIB = 1024**3


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


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    machine: str
    cpu_cores: int
    ram_bytes: int
    accelerator: str
    accelerator_memory_bytes: int


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


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
        raise argparse.ArgumentTypeError(
            "both output dimensions must be at least 64 pixels"
        )
    return Resolution(short_side=min(width, height), exact_size=(width, height))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale an image or folder with a portable SeedVR2 environment."
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input image or directory")
    parser.add_argument("output", type=Path, nargs="?", help="Output PNG or directory")
    parser.add_argument(
        "output_res",
        type=parse_resolution,
        nargs="?",
        help="Short side (1440) or exact dimensions (2560x1440)",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("clean", "reuse"),
        default="clean",
        help="Delete dependencies/models after the run (clean) or retain them (reuse)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Persistent cache root for --cache-mode reuse",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Persistent SeedVR2 model directory; never removed by this script",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Parent for the private per-run staging directory",
    )
    parser.add_argument(
        "--device",
        type=nonnegative_int,
        default=0,
        help="NVIDIA GPU index on Windows/Linux (default: 0)",
    )
    parser.add_argument(
        "--min-cpu-cores",
        type=nonnegative_int,
        default=DEFAULT_MIN_CPU_CORES,
        help=f"Minimum logical CPU cores; 0 disables (default: {DEFAULT_MIN_CPU_CORES})",
    )
    parser.add_argument(
        "--min-ram-gb",
        type=nonnegative_float,
        default=DEFAULT_MIN_RAM_GIB,
        help=f"Minimum system RAM in GiB; 0 disables (default: {DEFAULT_MIN_RAM_GIB:g})",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=nonnegative_float,
        default=DEFAULT_MIN_VRAM_GIB,
        help=f"Minimum VRAM/unified memory in GiB; 0 disables (default: {DEFAULT_MIN_VRAM_GIB:g})",
    )
    parser.add_argument(
        "--min-disk-gb",
        type=nonnegative_float,
        default=DEFAULT_MIN_DISK_GIB,
        help=f"Minimum free work/cache disk in GiB; 0 disables (default: {DEFAULT_MIN_DISK_GIB:g})",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip CPU, RAM, accelerator and free-disk refusal checks",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Report/refuse hardware suitability without downloading or upscaling",
    )
    args = parser.parse_args(argv)
    if args.preflight_only and args.skip_preflight:
        parser.error("--preflight-only cannot be combined with --skip-preflight")
    if not args.preflight_only and (
        args.input is None or args.output is None or args.output_res is None
    ):
        parser.error(
            "INPUT, OUTPUT and OUTPUT_RES are required unless --preflight-only is used"
        )
    return args


def find_uv() -> Path:
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)

    executable = "uv.exe" if os.name == "nt" else "uv"
    candidates: list[Path] = [
        Path.home() / ".local" / "bin" / executable,
        Path.home() / ".cargo" / "bin" / executable,
    ]
    if os.name == "nt":
        candidates.extend(
            [
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


def platform_cache_directory(system: str | None = None) -> Path:
    selected = system or platform.system()
    if selected == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        return (
            Path(base) if base else Path.home() / "AppData" / "Local"
        ) / "clean-seedvr2-upscaler"
    if selected == "Darwin":
        return Path.home() / "Library" / "Caches" / "clean-seedvr2-upscaler"
    if selected == "Linux":
        base = os.environ.get("XDG_CACHE_HOME")
        return (
            Path(base).expanduser() if base else Path.home() / ".cache"
        ) / "clean-seedvr2-upscaler"
    raise RuntimeError(f"Unsupported operating system: {selected}")


def nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError(f"Cannot find an existing parent for: {path}")
        candidate = parent
    return candidate


def total_ram_bytes(system: str | None = None) -> int:
    selected = system or platform.system()
    if selected == "Windows":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("Windows RAM detection failed")
        return int(status.total_physical)

    if selected == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(result.stdout.strip())

    if selected == "Linux":
        sysconf_value = vars(os).get("sysconf")
        if not callable(sysconf_value):
            raise RuntimeError("Linux RAM detection failed")
        sysconf = cast(Callable[[str], int], sysconf_value)
        page_size = sysconf("SC_PAGE_SIZE")
        page_count = sysconf("SC_PHYS_PAGES")
        return page_size * page_count

    raise RuntimeError(f"Unsupported operating system: {selected}")


def find_nvidia_smi() -> Path:
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return Path(discovered)
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidate = (
                Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            )
            if candidate.is_file():
                return candidate
    raise RuntimeError("An NVIDIA GPU and nvidia-smi are required on Windows and Linux")


def nvidia_accelerator(device: int) -> tuple[str, int]:
    result = subprocess.run(
        [
            find_nvidia_smi(),
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows: dict[int, tuple[str, int, str]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue
        index, name, memory_mib, driver = parts
        try:
            rows[int(index)] = (name, int(memory_mib), driver)
        except ValueError:
            continue
    if device not in rows:
        available = ", ".join(str(index) for index in sorted(rows)) or "none"
        raise RuntimeError(
            f"NVIDIA GPU index {device} was not found; available indices: {available}"
        )
    name, memory_mib, driver = rows[device]
    label = name if name.casefold().startswith("nvidia") else f"NVIDIA {name}"
    return f"{label} (driver {driver}, device {device})", memory_mib * 1024**2


def detect_hardware(device: int, system: str | None = None) -> HardwareProfile:
    selected = system or platform.system()
    if selected not in SUPPORTED_SYSTEMS:
        raise RuntimeError(f"Unsupported operating system: {selected}")
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("SeedVR2 requires a 64-bit operating system and Python")
    machine = platform.machine().lower()
    cpu_cores = os.cpu_count() or 0
    ram = total_ram_bytes(selected)

    if selected == "Darwin":
        if machine not in {"arm64", "aarch64"}:
            raise RuntimeError("SeedVR2 MPS support requires an Apple Silicon Mac")
        accelerator = "Apple Silicon MPS (unified memory)"
        accelerator_memory = ram
    else:
        accelerator, accelerator_memory = nvidia_accelerator(device)

    return HardwareProfile(
        system=selected,
        machine=machine,
        cpu_cores=cpu_cores,
        ram_bytes=ram,
        accelerator=accelerator,
        accelerator_memory_bytes=accelerator_memory,
    )


def gib(value: int) -> float:
    return value / GIB


def run_preflight(
    work_paths: list[Path],
    device: int,
    min_cpu_cores: int,
    min_ram_gib: float,
    min_vram_gib: float,
    min_disk_gib: float,
) -> HardwareProfile:
    profile = detect_hardware(device)
    disk_locations: dict[int, tuple[Path, int]] = {}
    for work_path in work_paths:
        existing = nearest_existing_path(work_path)
        device_id = os.stat(existing).st_dev
        disk_locations.setdefault(
            device_id, (existing, shutil.disk_usage(existing).free)
        )

    failures: list[str] = []
    if min_cpu_cores and profile.cpu_cores < min_cpu_cores:
        failures.append(
            f"CPU has {profile.cpu_cores} logical cores; {min_cpu_cores} required"
        )
    if min_ram_gib and profile.ram_bytes < min_ram_gib * GIB:
        failures.append(
            f"RAM is {gib(profile.ram_bytes):.1f} GiB; {min_ram_gib:g} GiB required"
        )
    if min_vram_gib and profile.accelerator_memory_bytes < min_vram_gib * GIB:
        label = "unified memory" if profile.system == "Darwin" else "VRAM"
        failures.append(
            f"{label} is {gib(profile.accelerator_memory_bytes):.1f} GiB; "
            f"{min_vram_gib:g} GiB required"
        )
    if min_disk_gib:
        for existing, free_disk in disk_locations.values():
            if free_disk < min_disk_gib * GIB:
                failures.append(
                    f"free disk at {existing} is {gib(free_disk):.1f} GiB; "
                    f"{min_disk_gib:g} GiB required"
                )

    print("\nPreflight:", flush=True)
    print(
        f"  OS: {profile.system} {profile.machine}\n"
        f"  CPU: {profile.cpu_cores} logical cores\n"
        f"  RAM: {gib(profile.ram_bytes):.1f} GiB\n"
        f"  Accelerator: {profile.accelerator}\n"
        f"  Accelerator memory: {gib(profile.accelerator_memory_bytes):.1f} GiB",
        flush=True,
    )
    for existing, free_disk in disk_locations.values():
        print(
            f"  Free work/cache disk: {gib(free_disk):.1f} GiB at {existing}",
            flush=True,
        )
    if failures:
        joined = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(
            "Hardware preflight refused this run:\n"
            f"{joined}\n"
            "Adjust the --min-* limits or use --skip-preflight if you accept the risk."
        )
    return profile


def run(command: list[os.PathLike[str] | str], cwd: Path | None = None) -> None:
    values = [os.fspath(part) for part in command]
    print(f"\n> {subprocess.list2cmdline(values)}", flush=True)
    subprocess.run(values, cwd=cwd, check=True)


def create_temporary_environment(
    temp_parent: Path | None = None,
    persistent_cache: Path | None = None,
) -> TemporaryEnvironment:
    """Create an owned session directory and redirect task caches into it."""
    selected_parent = temp_parent or Path(
        os.environ.get("SEEDVR2_TEMP_ROOT", tempfile.gettempdir())
    )
    parent = selected_parent.expanduser().resolve()
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

    cache_root = (
        persistent_cache.expanduser().resolve()
        if persistent_cache is not None
        else cleanup_root
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    redirected = {
        "UV_CACHE_DIR": cache_root / "uv-cache",
        "UV_PYTHON_INSTALL_DIR": cache_root / "uv-python",
        "PIP_CACHE_DIR": cache_root / "pip-cache",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TORCH_HOME": cache_root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "XDG_CACHE_HOME": cache_root / "xdg-cache",
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
                raise RuntimeError(
                    f"Cleanup entry escaped its session root: {path}"
                ) from error
            if path == marker:
                continue

            # Windows DirEntry.stat() may report st_dev/st_ino as zero even
            # when os.lstat() exposes the stable filesystem identity. Use the
            # same identity API for both manifest creation and deletion checks.
            info = os.lstat(path)
            kind = cleanup_kind(info)
            if kind == "special":
                raise RuntimeError(
                    f"Refusing to remove a special filesystem entry: {path}"
                )
            if info.st_dev != root_identity.st_dev:
                raise RuntimeError(f"Refusing to cross a filesystem boundary: {path}")
            if kind == "directory":
                if is_filesystem_boundary(path, info):
                    raise RuntimeError(
                        f"Refusing to cross a mounted filesystem: {path}"
                    )
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
            raise RuntimeError(
                f"Cleanup path identity changed after chmod: {entry.path}"
            )
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
        raise RuntimeError(
            "Both live and quarantined cleanup paths exist; refusing deletion"
        )

    expected_marker = f"{SESSION_MAGIC}\n{session.token}\n{cleanup_root}\n"
    if root_exists:
        if cleanup_kind(os.lstat(cleanup_root)) == "link":
            raise RuntimeError(
                f"Refusing to remove a linked cleanup path: {cleanup_root}"
            )
        resolved_root = cleanup_root.resolve()
        if resolved_root != cleanup_root or cleanup_root.parent != session.parent:
            raise RuntimeError(
                f"Refusing to remove an unsafe cleanup path: {cleanup_root}"
            )
        if not cleanup_root.name.startswith(SESSION_PREFIX):
            raise RuntimeError(
                f"Refusing to remove an unsafe cleanup path: {cleanup_root}"
            )

        marker = cleanup_root / SESSION_MARKER
        if marker.is_symlink() or not marker.is_file():
            raise RuntimeError(
                f"Refusing to remove an unowned cleanup path: {cleanup_root}"
            )
        if not secrets.compare_digest(
            marker.read_text(encoding="utf-8"), expected_marker
        ):
            raise RuntimeError(
                f"Cleanup ownership marker did not match: {cleanup_root}"
            )

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
    revision_marker = source / ".clean-seedvr2-source-revision"

    if source.is_dir():
        if (
            (source / "inference_cli.py").is_file()
            and revision_marker.is_file()
            and revision_marker.read_text(encoding="utf-8").strip() == SEEDVR2_REVISION
        ):
            print(
                f"Reusing SeedVR2 source revision {SEEDVR2_REVISION[:12]}",
                flush=True,
            )
            return source
        raise RuntimeError(
            f"Cached SeedVR2 source is incomplete or from another revision: {source}. "
            "Select another --cache-dir or remove this tool-owned cache manually."
        )
    if archive_path.exists() or extraction_root.exists():
        raise RuntimeError(
            f"An incomplete SeedVR2 source download exists in {cache_directory}. "
            "Select another --cache-dir or remove the incomplete tool-owned files manually."
        )

    cache_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading official SeedVR2 source at revision {SEEDVR2_REVISION[:12]}",
        flush=True,
    )
    request = urllib.request.Request(
        SEEDVR2_ARCHIVE,
        headers={"User-Agent": "clean-seedvr2-upscaler"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        archive_path.open("wb") as file,
    ):
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
    revision_marker.write_text(SEEDVR2_REVISION, encoding="utf-8")
    return source


def environment_fingerprint(source: Path, system: str) -> str:
    digest = hashlib.sha256()
    digest.update(SEEDVR2_REVISION.encode("ascii"))
    digest.update(system.encode("utf-8"))
    if system != "Darwin":
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


def ensure_environment(
    cache_directory: Path,
    source: Path,
    uv: Path,
    system: str,
) -> Path:
    fingerprint = environment_fingerprint(source, system)
    venv = cache_directory / f"venv-{fingerprint[:16]}"
    python = venv_python(venv)
    marker = venv / ".seedvr2-wrapper-ready"

    if (
        python.is_file()
        and marker.is_file()
        and marker.read_text().strip() == fingerprint
    ):
        print(f"Reusing SeedVR2 Python environment: {venv}", flush=True)
        return python
    if venv.exists():
        raise RuntimeError(
            f"Cached Python environment is incomplete: {venv}. Select another "
            "--cache-dir or remove this tool-owned environment manually."
        )

    run([uv, "venv", "--python", "3.12", venv])
    if system == "Darwin":
        print("Installing MPS PyTorch and SeedVR2 dependencies.", flush=True)
        run([uv, "pip", "install", "--python", python, "torch", "torchvision"])
    else:
        print("Installing CUDA PyTorch and SeedVR2 dependencies.", flush=True)
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
    marker.write_text(fingerprint, encoding="utf-8")
    return python


def validate_torch_accelerator(python: Path, system: str) -> None:
    if system == "Darwin":
        program = (
            "import torch; "
            "assert torch.backends.mps.is_available(), "
            "'PyTorch MPS is not available'; "
            "print('MPS ready on Apple Silicon')"
        )
    else:
        program = (
            "import torch; "
            "assert torch.cuda.is_available(), "
            "'CUDA-enabled PyTorch is not available'; "
            "print('CUDA ready:', torch.cuda.get_device_name(0))"
        )
    run(
        [
            python,
            "-c",
            program,
        ]
    )


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
        raise RuntimeError(
            f"Output appeared during processing; refusing overwrite: {conflicts[0]}"
        )

    for staged, final in zip(staged_targets, final_targets, strict=True):
        final.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(final):
            raise RuntimeError(
                f"Output appeared during processing; refusing overwrite: {final}"
            )

        partial = final.with_name(f".{final.name}.clean-seedvr2-{token}.partial")
        if os.path.lexists(partial):
            raise RuntimeError(
                f"Refusing to replace an existing publish staging file: {partial}"
            )

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
    model_directory: Path,
    system: str,
    device: int,
) -> None:
    command: list[os.PathLike[str] | str] = [
        python,
        seedvr2_source / "inference_cli.py",
        source,
        "--output",
        cli_output,
        "--output_format",
        "png",
        "--model_dir",
        model_directory,
        "--resolution",
        str(resolution.short_side),
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

    if system != "Darwin":
        command.extend(["--cuda_device", str(device)])

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
    system = platform.system()
    if system not in SUPPORTED_SYSTEMS:
        raise RuntimeError(f"Unsupported operating system: {system}")
    if args.cache_dir is not None and args.cache_mode != "reuse":
        raise RuntimeError("--cache-dir requires --cache-mode reuse")

    temp_parent = (
        args.temp_dir.expanduser().resolve()
        if args.temp_dir is not None
        else Path(os.environ.get("SEEDVR2_TEMP_ROOT", tempfile.gettempdir()))
        .expanduser()
        .resolve()
    )
    persistent_cache = None
    if args.cache_mode == "reuse":
        persistent_cache = (
            args.cache_dir.expanduser().resolve()
            if args.cache_dir is not None
            else platform_cache_directory(system).expanduser().resolve()
        )

    preflight_paths = [persistent_cache or temp_parent]
    if args.model_dir is not None:
        preflight_paths.append(args.model_dir.expanduser().resolve())
    if not args.skip_preflight:
        run_preflight(
            preflight_paths,
            args.device,
            args.min_cpu_cores,
            args.min_ram_gb,
            args.min_vram_gb,
            args.min_disk_gb,
        )
    else:
        print("\nWARNING: Hardware preflight was skipped by request.", flush=True)

    if args.preflight_only:
        print("\nPreflight passed; no files were downloaded or changed.", flush=True)
        return 0

    if args.input is None or args.output is None or args.output_res is None:
        raise RuntimeError("Internal argument validation error")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    images = input_images(source)

    if source.is_dir() and (output == source or source in output.parents):
        raise RuntimeError(
            "For directory input, output must be outside the input directory"
        )

    final_targets = plan_outputs(source, output, images)

    session = create_temporary_environment(temp_parent, persistent_cache)
    try:
        runtime_cache = (
            persistent_cache / "runtime"
            if persistent_cache is not None
            else session.cache
        )
        if args.model_dir is not None:
            model_directory = args.model_dir.expanduser().resolve()
        elif persistent_cache is not None:
            model_directory = persistent_cache / "models" / "SEEDVR2"
        else:
            model_directory = session.cache / "models" / "SEEDVR2"

        print(f"\nCache mode: {args.cache_mode}", flush=True)
        if persistent_cache is not None:
            print(f"Persistent cache: {persistent_cache}", flush=True)
        if args.model_dir is not None or persistent_cache is not None:
            print(f"Persistent models: {model_directory}", flush=True)

        uv = find_uv()
        seedvr2_source = ensure_source(runtime_cache)
        python = ensure_environment(runtime_cache, seedvr2_source, uv, system)
        validate_torch_accelerator(python, system)
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
            model_directory,
            system,
            args.device,
        )

        for target in staged_targets:
            width, height = png_size(target)
            if (
                args.output_res.exact_size
                and (width, height) != args.output_res.exact_size
            ):
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

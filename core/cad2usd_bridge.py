"""Helpers for launching the external CAD2USD converter."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

CAD2USD_ROOT_ENV = "CAD2USD_ROOT"

CAD_EXTENSIONS = (
    ".3dm",
    ".3dxml",
    ".asm",
    ".catpart",
    ".catproduct",
    ".cgr",
    ".dae",
    ".dxf",
    ".fbx",
    ".g",
    ".glb",
    ".gltf",
    ".iam",
    ".iges",
    ".igs",
    ".ipn",
    ".ipt",
    ".jt",
    ".neu",
    ".obj",
    ".par",
    ".prt",
    ".psm",
    ".sab",
    ".sat",
    ".sldasm",
    ".slddrw",
    ".sldprt",
    ".step",
    ".stl",
    ".stp",
    ".x_b",
    ".x_t",
    ".xas",
    ".xmt_bin",
    ".xmt_txt",
    ".xpr",
)

PTC_VERSIONED_EXTENSIONS = (
    ".asm",
    ".g",
    ".neu",
    ".prt",
    ".xas",
    ".xpr",
)

CAD_DIALOG_PATTERNS = tuple(f"*{ext}" for ext in CAD_EXTENSIONS) + tuple(
    f"*{ext}.*" for ext in PTC_VERSIONED_EXTENSIONS
)

CAD_FILE_FILTER = (
    "CAD and 3D files ("
    + " ".join(CAD_DIALOG_PATTERNS)
    + ");;All files (*.*)"
)


def find_cad2usd_root(app_root: Optional[Path] = None) -> Optional[Path]:
    """Return a CAD2USD checkout containing convert.bat, if one is discoverable."""
    for candidate in cad2usd_root_candidates(app_root):
        convert_bat = candidate / "convert.bat"
        if convert_bat.is_file():
            return candidate
    return None


def cad2usd_root_candidates(app_root: Optional[Path] = None) -> list[Path]:
    """Likely checkout locations, with CAD2USD_ROOT taking precedence."""
    roots: list[Path] = []
    env_root = os.environ.get(CAD2USD_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root))

    if app_root is None:
        app_root = Path(__file__).resolve().parents[1]
    app_root = Path(app_root)

    roots.extend(
        [
            app_root / "CAD2USD",
            app_root.parent / "CAD2USD",
            app_root.parent.parent / "CAD2USD",
            Path.home() / "CAD2USD",
            Path.home() / "source" / "repos" / "CAD2USD",
            Path.home() / "Documents" / "GitHub" / "CAD2USD",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def cad_file_extension(path: str | Path) -> str:
    """Return the converter extension, including Creo names like part.prt.1."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in CAD_EXTENSIONS:
        return suffix

    suffixes = [item.lower() for item in source.suffixes]
    if len(suffixes) >= 2 and suffixes[-1][1:].isdigit():
        versioned_suffix = suffixes[-2]
        if versioned_suffix in PTC_VERSIONED_EXTENSIONS:
            return versioned_suffix
    return suffix


def is_supported_cad_file(path: str | Path) -> bool:
    return cad_file_extension(path) in CAD_EXTENSIONS


def make_cad_usd_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    """Create a fresh cache output path for a converted USD."""
    source = Path(input_path)
    out_dir = Path(output_dir)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._")
    if not safe_stem:
        safe_stem = "converted_cad"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return out_dir / f"{safe_stem}_{stamp}_{unique}.usd"

# SimReady Browser

Qt desktop browser for NVIDIA Isaac SimReady assets hosted in public S3, with an interactive NVIDIA OVRTX viewport for reviewing USD assets.

## Features

- Browses SimReady assets from `s3://omniverse-content-production/Assets/Isaac/6.0/Isaac/SimReady/`
- Lazy thumbnail loading with local cache support
- Double-click an asset thumbnail to load it in the viewport
- OVRTX RTX viewport with Kit-style camera controls, WASD fly movement, and `F` frame extents
- Z-up Isaac stage setup, ground plane, shadows, dome light, and direct light controls
- OVPhysX physics playback with composed USD collision discovery for referenced and instanceable SimReady assets
- Asset loading progress overlay

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ovrtx --extra-index-url https://pypi.nvidia.com
```

Composed USD collision traversal uses an isolated helper environment so `usd-core`
does not conflict with OVRTX/OVPhysX:

```powershell
uv venv .usd_discovery_venv --python .\.venv\Scripts\python.exe
uv pip install --python .\.usd_discovery_venv\Scripts\python.exe usd-core==25.11
$env:SIMREADY_USD_PYTHON = (Resolve-Path .\.usd_discovery_venv\Scripts\python.exe)
```

You can also run `launch.bat`, which sets up the local environment and starts the app.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

## Controls

- `Alt + LMB`: tumble
- `Alt + MMB`: pan
- `Alt + RMB` or mouse wheel: dolly
- `RMB + WASD/QE`: fly
- `F`: frame asset extents
- Double-click an asset thumbnail: load in viewport

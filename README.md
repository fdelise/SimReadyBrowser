# SimReady Browser

SimReady Browser is a Qt desktop tool for browsing NVIDIA Isaac SimReady assets from the public Omniverse S3 content bucket, loading them into an interactive OVRTX viewport, and reviewing how the authored USD collision data behaves in OVPhysX.

The app is built for asset QA and simulation-readiness review. It combines a fast S3 asset browser, an RTX viewport with Kit-style navigation, collision visualization, and a lightweight physics play mode so you can inspect whether a SimReady asset renders correctly, has the expected authored colliders, and behaves properly when played, dragged, or tested against simple base scenes.

## What It Does

- Browses SimReady assets from `s3://omniverse-content-production/Assets/Isaac/6.0/Isaac/SimReady/`.
- Uses the S3-side manifest and local cache data where available so asset discovery is fast.
- Loads thumbnails lazily and caches them locally to keep category browsing responsive.
- Double-clicks an asset card to load the USD into the OVRTX viewport.
- Provides a Z-up Isaac-style review stage with ground plane, shadows, dome light, and direct light controls.
- Supports Kit-style camera navigation, WASD fly movement, zoom-to-extents with `F`, and viewport progress feedback.
- Runs OVPhysX in a subprocess so Qt, OVRTX, OpenUSD, and PhysX runtime state stay isolated.
- Traverses composed USD stages, including referenced and instanceable SimReady payloads, to find authored collision prims.
- Uses the asset's authored colliders for physics review. It does not create fake asset colliders when authored colliders are missing or unavailable.
- Adds review base scenes: plane, ramp, and obstacles.
- Shows collision debug overlays as wireframe curves so you can compare the visible render asset against the collision representation.
- Supports physics playback, restart, explicit single or multi-asset drop-from-air testing, current-pose simulation startup, optional CCD, and Shift + left mouse physics grabbing with adjustable grab force.

## Typical Workflow

1. Start the app with `launch.bat` or `python main.py`.
2. Browse SimReady categories from the left asset browser.
3. Double-click a thumbnail to load the asset into the viewport.
4. Review the asset visually using Kit-style camera controls.
5. Toggle collision visualization to inspect authored collider wireframes.
6. Use the Physics panel to select a base scene, then press Play or Drop.
7. Play starts from the asset's current viewport pose; Drop places one or more copies above the base scene and starts physics.
8. Hold Shift + left mouse button on the asset to grab it with a force-based physics handle, drag it, then release to drop or throw it.

## Project Layout

```text
SimReadyBrowser/
  main.py                         Qt application entry point
  launch.bat                      Windows launcher and environment helper
  requirements.txt                Python dependencies available from public PyPI
  core/
    s3_client.py                  S3 catalog, manifest, asset, and thumbnail loading
    ovrtx_renderer.py             OVRTX stage setup, viewport rendering, lights, overlays
    camera_controller.py          Kit-style camera and WASD movement
    physics_controller.py         Qt-side OVPhysX worker controller and scene authoring
    physics_worker.py             Isolated OVPhysX worker process
    usd_collision_discovery.py    Isolated OpenUSD composed-collider traversal helper
  ui/
    asset_browser.py              Search, category browser, and asset cards
    viewport_widget.py            Qt viewport host and input routing
    controls_panel.py             Lighting, view, collision, and physics controls
    main_window.py                Main Qt layout and signal wiring
  styles/
    nvidia_theme.py               NVIDIA-style Qt palette and widget styling
  tools/
    physics_authored_smoke.py     OVPhysX authored-collider smoke tests
    physics_controller_process_smoke.py
    physics_discovery_smoke.py
```

## Requirements

- Windows is the primary tested platform.
- Python 3.10 or newer.
- NVIDIA GPU and current driver recommended for OVRTX/OVPhysX.
- Network access to the public Omniverse S3 bucket.
- NVIDIA Python package access through `https://pypi.nvidia.com`.
- Optional but recommended: `uv` for creating the isolated USD discovery environment.

## Setup

Create the main virtual environment and install the app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe -m pip install ovrtx --extra-index-url https://pypi.nvidia.com
```

Composed USD collision traversal uses a separate helper environment so `usd-core` does not conflict with the OVRTX/OVPhysX runtime packages loaded by the main app:

```powershell
uv venv .usd_discovery_venv --python .\.venv\Scripts\python.exe
uv pip install --python .\.usd_discovery_venv\Scripts\python.exe usd-core==25.11
$env:SIMREADY_USD_PYTHON = (Resolve-Path .\.usd_discovery_venv\Scripts\python.exe)
```

You can also run:

```powershell
.\launch.bat
```

The launcher checks the local environment, sets the expected runtime paths, and starts the app.

## Running

```powershell
.\.venv\Scripts\python.exe main.py
```

The app creates local cache data under `cache/`. Cached catalog and thumbnail data are intentionally ignored by Git.

## Viewport Controls

- Double-click asset thumbnail: load asset in the viewport
- `Alt + Left Mouse`: tumble/orbit
- `Alt + Middle Mouse`: pan
- `Alt + Right Mouse`: dolly
- Mouse wheel: dolly
- Right mouse + `W/A/S/D`: fly camera
- Right mouse + `Q/E`: fly down/up
- `F`: frame asset extents
- Shift + left mouse on asset: physics grab while physics is active

## Physics Review

The physics path is designed to validate authored SimReady collision data, not to hide asset problems.

- The asset is referenced into an OVPhysX scene as `/World/Asset`; multi-drop adds sibling instances such as `/World/Asset_02`.
- The app traverses the composed USD stage, including payloads, references, and instanceable prototypes, to discover collision prims.
- Authored collider paths are passed to OVPhysX as body binding candidates.
- SDF or mesh collision authoring is preserved where possible; helper overrides are only used to make authored SimReady collision prims cook in the local OVPhysX path.
- If no usable authored collision shapes are exposed by OVPhysX, the app reports that instead of creating a fake asset collider.
- The floor is authored as a simple box collider, not a triangle mesh.
- The ramp base scene uses a closed convex wedge collider so the ramp itself has reliable collision.
- Physics Play starts from the asset's current viewport pose. It no longer automatically drops the asset from above the base scene.
- Drop places the selected number of asset copies above the base scene with tight random offsets so they collide. The UI caps this at 100 copies. Re-dropping the same count reuses the cooked OVPhysX scene; changing the count rebuilds because the number of rigid-body instances changes.
- The CCD checkbox authors scene-level `physxScene:enableCCD` on the next physics scene start. It does not mutate authored asset body or collider schemas, and toggling it does not force a collider re-cook.
- Restart resets the cooked physics scene without re-cooking when possible.
- Grab force is scaled by the estimated mass and by the user-controlled grab force multiplier, which supports 0.25x to 100x.

## Collision Debug Overlay

Collision visualization is shown as wireframe overlay geometry. For asset colliders, the app uses the USD discovery helper to compose the asset and write a wire overlay layer from authored collider mesh edges or extents. For review base scenes, the ground, ramp, and obstacle collision shapes are shown with bright wire curves.

This overlay is for inspection only. It does not replace, simplify, or fabricate asset collision for simulation.

## Development Checks

Compile the main modules:

```powershell
.\.venv\Scripts\python.exe -m py_compile main.py core\s3_client.py core\camera_controller.py core\ovrtx_renderer.py core\physics_controller.py core\physics_worker.py core\usd_collision_discovery.py ui\asset_browser.py ui\controls_panel.py ui\main_window.py ui\viewport_widget.py styles\nvidia_theme.py
```

Run physics smoke tests:

```powershell
.\.venv\Scripts\python.exe tools\physics_authored_smoke.py drop
.\.venv\Scripts\python.exe tools\physics_authored_smoke.py multi
.\.venv\Scripts\python.exe tools\physics_authored_smoke.py grab
.\.venv\Scripts\python.exe tools\physics_authored_smoke.py ramp
.\.venv\Scripts\python.exe tools\physics_controller_process_smoke.py
.\.venv\Scripts\python.exe tools\physics_discovery_smoke.py
```

Some OVPhysX warning output is expected on systems where GPU cooking falls back to software. The smoke tests should still exit successfully.

## Troubleshooting

If the app appears stuck while loading an asset, watch the bottom viewport progress bar and the right-side status panel. Asset load and collider cook progress are reported there.

If collision discovery fails with a USD package conflict, recreate the `.usd_discovery_venv` helper environment and make sure `SIMREADY_USD_PYTHON` points to that helper Python, not the main `.venv` Python.

If physics starts but an asset falls through the floor, check the status panel and collision overlay. The app should only play physics when OVPhysX exposes usable authored collision shapes for the asset. If it reports zero usable shapes, the issue is in discovery, binding, or local OVPhysX cooking rather than the review floor.

If thumbnails or catalog browsing feel slow, delete stale cache data under `cache/` and restart. The app will rebuild the catalog and thumbnail cache.

If closing the app crashes inside OVRTX/OVPhysX, make sure the latest committed subprocess-based physics path is running. The app is structured so OVPhysX shutdown happens outside the Qt/OVRTX process.

## Notes

This is a local review tool for SimReady asset inspection and physics validation. It is not a replacement for full Isaac Sim validation, but it is intended to make first-pass review fast: browse, load, inspect, visualize colliders, play physics, grab, and decide whether an asset is ready for deeper simulation testing.

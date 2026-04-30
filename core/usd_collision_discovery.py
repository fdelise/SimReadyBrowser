"""Composed OpenUSD collision discovery for SimReady assets.

This module runs in a separate Python process so importing pxr never mixes
OpenUSD Python bindings into the Qt/OVRTX process or the OVPhysX worker.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


S3_USER_AGENT = "SimReadyBrowser/1.0"
KNOWN_SIMREADY_PAYLOADS = (
    "payloads/base.usda",
    "payloads/instances.usda",
    "payloads/geometries.usd",
    "payloads/materials.usda",
)
USD_REF_RE = re.compile(r"@([^@\r\n]+?\.usd[acz]?)(?:<[^>]*>)?@")
AUTHORED_COLLIDER_WIRE_WIDTH = 0.0015


def discover(asset_ref: str, root_path: str = "/World/Asset") -> dict:
    from pxr import Usd, UsdPhysics

    stage, root = _open_inspection_stage(asset_ref, root_path)

    entries = list(_iter_composed_prims(root))
    rigid_paths = _rigid_body_paths(entries, UsdPhysics)
    articulation_paths = _articulation_root_paths(entries, UsdPhysics)
    colliders = []
    for prim, mapped_path in entries:
        if not _is_collision_prim(prim, UsdPhysics):
            continue
        schemas = list(prim.GetAppliedSchemas())
        approximation = _token_attr(prim, "physics:approximation")
        collider_type = _collider_type(prim, schemas, approximation)
        colliders.append(
            {
                "path": mapped_path,
                "body_path": _body_path_for(mapped_path, rigid_paths, root_path),
                "type": collider_type,
                "approximation": approximation,
                "schemas": schemas,
                "prim_type": prim.GetTypeName(),
            }
        )

    override_paths = [
        item["path"]
        for item in colliders
        if item["approximation"] == "sdf" or "PhysxSDFMeshCollisionAPI" in item["schemas"]
    ]
    body_paths = _dedupe([item["body_path"] for item in colliders])
    body_patterns = _body_patterns(root_path, body_paths)
    type_counts: dict[str, int] = {}
    for item in colliders:
        key = item["type"] or "collision"
        type_counts[key] = type_counts.get(key, 0) + 1

    return {
        "colliders": colliders,
        "collider_count": len(colliders),
        "override_paths": _dedupe(override_paths),
        "override_count": len(_dedupe(override_paths)),
        "body_paths": body_paths,
        "body_patterns": body_patterns,
        "articulation_paths": articulation_paths,
        "type_counts": type_counts,
    }


def write_wire_overlay(asset_ref: str, output_path: str, root_path: str = "/World/Asset") -> dict:
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _open_inspection_stage(asset_ref, root_path)
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root)
    root_inverse = root_world.GetInverse()
    entries = list(_iter_composed_prims(root))

    curve_blocks: list[str] = []
    collider_count = 0
    edge_count = 0
    for prim, mapped_path in entries:
        if not _is_collision_prim(prim, UsdPhysics):
            continue
        edges = _mesh_edges(prim)
        if not edges:
            edges = _extent_edges(prim)
        if not edges:
            continue

        collider_count += 1
        local_to_root = cache.GetLocalToWorldTransform(prim) * root_inverse
        curve_points = []
        for start, end in edges:
            transformed_start = local_to_root.Transform(Gf.Vec3d(*start))
            transformed_end = local_to_root.Transform(Gf.Vec3d(*end))
            curve_points.append(transformed_start)
            curve_points.append(transformed_end)
        edge_count += len(edges)
        curve_blocks.append(_basis_curves_block(f"ColliderWire_{collider_count:03d}", curve_points, mapped_path))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_wire_overlay_layer(curve_blocks), encoding="utf-8")
    return {"collider_count": collider_count, "edge_count": edge_count, "path": str(output)}


def _open_inspection_stage(asset_ref: str, root_path: str):
    from pxr import Usd

    work_dir = Path(tempfile.gettempdir()) / "simready_browser_usd_discovery" / _safe_hash(asset_ref)
    work_dir.mkdir(parents=True, exist_ok=True)
    local_asset = _local_asset(asset_ref, work_dir)
    scene_path = _write_inspection_scene(work_dir, local_asset)

    stage = Usd.Stage.Open(str(scene_path))
    if not stage:
        raise RuntimeError(f"OpenUSD could not open {scene_path}")
    stage.Load()

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Inspection stage has no root prim at {root_path}")
    return stage, root


def _local_asset(asset_ref: str, work_dir: Path) -> Path:
    text = str(asset_ref or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return _mirror_http_asset(text, work_dir)
    path = Path(text)
    if not path.exists():
        raise RuntimeError(f"USD asset does not exist: {asset_ref}")
    return path.resolve()


def _mirror_http_asset(url: str, work_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    asset_name = Path(urllib.parse.unquote(parsed.path)).name or "asset.usd"
    local_asset = work_dir / asset_name
    queue: list[tuple[str, Path]] = [(url, local_asset)]
    base_url = url.rsplit("/", 1)[0] + "/"
    for rel in KNOWN_SIMREADY_PAYLOADS:
        queue.append((urllib.parse.urljoin(base_url, rel), work_dir / rel))

    seen: set[str] = set()
    while queue and len(seen) < 200:
        current_url, local_path = queue.pop(0)
        if current_url in seen:
            continue
        seen.add(current_url)
        if not _download(current_url, local_path):
            continue
        text = _read_text_usd(local_path)
        if not text:
            continue
        current_base = current_url.rsplit("/", 1)[0] + "/"
        local_base = local_path.parent
        for ref in USD_REF_RE.findall(text):
            if "://" in ref or ref.startswith("#"):
                continue
            ref_url = urllib.parse.urljoin(current_base, ref)
            ref_local = (local_base / urllib.parse.unquote(ref)).resolve()
            try:
                ref_local.relative_to(work_dir.resolve())
            except ValueError:
                ref_local = work_dir / _safe_ref_name(ref)
            queue.append((ref_url, ref_local))
    return local_asset


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": S3_USER_AGENT})
        with urllib.request.urlopen(req, timeout=20.0) as response:
            dest.write_bytes(response.read())
        return True
    except Exception:
        return False


def _read_text_usd(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if data.startswith(b"PXR-USDC") or b"\x00" in data[:256]:
        return ""
    return data.decode("utf-8", "replace")


def _write_inspection_scene(work_dir: Path, local_asset: Path) -> Path:
    scene = work_dir / "inspect_scene.usda"
    asset_path = local_asset.resolve().as_posix().replace("@", "%40")
    scene.write_text(
        f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "Asset" (
        prepend references = @{asset_path}@
    )
    {{
    }}
}}
""",
        encoding="utf-8",
    )
    return scene


def _iter_composed_prims(root):
    from pxr import Usd

    for prim in Usd.PrimRange.AllPrims(root):
        path = str(prim.GetPath())
        yield prim, path
        if prim.IsInstance():
            prototype = prim.GetPrototype()
            if prototype and prototype.IsValid():
                yield from _iter_prototype_prims(prototype, path)


def _iter_prototype_prims(prototype, instance_path: str):
    from pxr import Usd

    proto_root = str(prototype.GetPath())
    for prim in Usd.PrimRange.AllPrims(prototype):
        proto_path = str(prim.GetPath())
        if proto_path == proto_root:
            continue
        mapped_path = instance_path + proto_path[len(proto_root) :]
        yield prim, mapped_path
        if prim.IsInstance():
            nested = prim.GetPrototype()
            if nested and nested.IsValid():
                yield from _iter_prototype_prims(nested, mapped_path)


def _rigid_body_paths(entries, UsdPhysics) -> list[str]:
    paths: list[str] = []
    for prim, mapped_path in entries:
        schemas = list(prim.GetAppliedSchemas())
        has_api = "PhysicsRigidBodyAPI" in schemas
        try:
            has_api = has_api or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        except Exception:
            pass
        if has_api and mapped_path not in paths:
            paths.append(mapped_path)
    return paths


def _articulation_root_paths(entries, UsdPhysics) -> list[str]:
    paths: list[str] = []
    articulation_api = getattr(UsdPhysics, "ArticulationRootAPI", None)
    for prim, mapped_path in entries:
        schemas = list(prim.GetAppliedSchemas())
        has_api = (
            "PhysicsArticulationRootAPI" in schemas
            or "PhysxArticulationAPI" in schemas
            or "PhysxArticulationRootAPI" in schemas
        )
        if articulation_api is not None:
            try:
                has_api = has_api or prim.HasAPI(articulation_api)
            except Exception:
                pass
        if has_api and mapped_path not in paths:
            paths.append(mapped_path)
    return paths


def _is_collision_prim(prim, UsdPhysics) -> bool:
    schemas = list(prim.GetAppliedSchemas())
    has_api = "PhysicsCollisionAPI" in schemas
    try:
        has_api = has_api or prim.HasAPI(UsdPhysics.CollisionAPI)
    except Exception:
        pass
    return bool(has_api)


def _token_attr(prim, name: str) -> str:
    attr = prim.GetAttribute(name)
    if not attr:
        return ""
    try:
        value = attr.Get()
    except Exception:
        return ""
    return "" if value is None else str(value)


def _collider_type(prim, schemas: list[str], approximation: str) -> str:
    if approximation:
        return approximation
    prim_type = prim.GetTypeName()
    if "PhysxSDFMeshCollisionAPI" in schemas:
        return "sdf"
    if "PhysxConvexDecompositionCollisionAPI" in schemas:
        return "convexDecomposition"
    if prim_type == "Mesh" or "PhysicsMeshCollisionAPI" in schemas:
        return "triangleMesh"
    return prim_type or "collision"


def _mesh_edges(prim) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    if prim.GetTypeName() != "Mesh":
        return []
    points_attr = prim.GetAttribute("points")
    counts_attr = prim.GetAttribute("faceVertexCounts")
    indices_attr = prim.GetAttribute("faceVertexIndices")
    if not points_attr or not counts_attr or not indices_attr:
        return []
    try:
        points = list(points_attr.Get() or [])
        counts = [int(value) for value in list(counts_attr.Get() or [])]
        indices = [int(value) for value in list(indices_attr.Get() or [])]
    except Exception:
        return []
    if not points or not counts or not indices:
        return []

    edge_indices: set[tuple[int, int]] = set()
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + max(0, count)]
        cursor += max(0, count)
        if len(face) < 2:
            continue
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            if start == end or start < 0 or end < 0 or start >= len(points) or end >= len(points):
                continue
            edge_indices.add((min(start, end), max(start, end)))

    edges = []
    for start_index, end_index in sorted(edge_indices):
        start = points[start_index]
        end = points[end_index]
        edges.append(((float(start[0]), float(start[1]), float(start[2])), (float(end[0]), float(end[1]), float(end[2]))))
    return edges


def _extent_edges(prim) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    extent_attr = prim.GetAttribute("extent")
    if not extent_attr:
        return []
    try:
        extent = list(extent_attr.Get() or [])
    except Exception:
        return []
    if len(extent) < 2:
        return []
    lo = extent[0]
    hi = extent[1]
    corners = [
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(lo[2])),
        (float(lo[0]), float(hi[1]), float(lo[2])),
        (float(lo[0]), float(lo[1]), float(hi[2])),
        (float(hi[0]), float(lo[1]), float(hi[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
        (float(lo[0]), float(hi[1]), float(hi[2])),
    ]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(corners[start], corners[end]) for start, end in pairs]


def _basis_curves_block(name: str, points, source_path: str) -> str:
    point_lines = ",\n                ".join(
        f"({_usd_float(point[0])}, {_usd_float(point[1])}, {_usd_float(point[2])})" for point in points
    )
    curve_counts = ", ".join("2" for _ in range(len(points) // 2))
    return f"""        def BasisCurves "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            custom string simready:collisionSource = "{source_path}"
            rel material:binding = </SimReadyReview/Materials/CollisionMat>
            uniform token type = "linear"
            uniform token wrap = "nonperiodic"
            int[] curveVertexCounts = [{curve_counts}]
            point3f[] points = [
                {point_lines}
            ]
            float[] widths = [{_usd_float(AUTHORED_COLLIDER_WIRE_WIDTH)}] (
                interpolation = "constant"
            )
            color3f[] primvars:displayColor = [(0.46, 0.95, 0)] (
                interpolation = "constant"
            )
            float[] primvars:displayOpacity = [1] (
                interpolation = "constant"
            )
        }}"""


def _wire_overlay_layer(curve_blocks: list[str]) -> str:
    curves = "\n\n".join(curve_blocks)
    return f"""#usda 1.0
(
    defaultPrim = "SimReadyReview"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "SimReadyReview"
{{
    def Xform "CollisionAssetOverlay"
    {{
{curves}
    }}
}}
"""


def _usd_float(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if value != value or value in (float("inf"), float("-inf")):
        value = 0.0
    return f"{value:.9g}"


def _body_path_for(collider_path: str, rigid_paths: list[str], root_path: str) -> str:
    best = ""
    for path in rigid_paths:
        if collider_path == path or collider_path.startswith(f"{path}/"):
            if len(path) > len(best):
                best = path
    if best:
        return best

    geometry_prefix = f"{root_path}/Geometry/"
    if collider_path.startswith(geometry_prefix):
        rest = collider_path[len(geometry_prefix) :]
        first = rest.split("/", 1)[0]
        return f"{geometry_prefix}{first}"
    return root_path


def _body_patterns(root_path: str, body_paths: list[str]) -> list[str]:
    patterns: list[str] = []

    def add(value: str) -> None:
        if value and value not in patterns:
            patterns.append(value)

    unique_bodies = _dedupe(body_paths)
    if any(path.startswith(f"{root_path}/Geometry/") for path in unique_bodies):
        add(f"{root_path}/Geometry/*")
    for body_path in unique_bodies:
        add(body_path)
        add(f"{body_path}/*")
    add(f"{root_path}/Geometry/*/*")
    add(f"{root_path}/*")
    add(root_path)
    return patterns


def _dedupe(values) -> list:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _safe_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()[:16]


def _safe_ref_name(value: str) -> str:
    text = urllib.parse.unquote(value).replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or _safe_hash(value)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m core.usd_collision_discovery <asset-ref> [root-path]\n"
            "       python -m core.usd_collision_discovery --wire-usd <asset-ref> <output.usda> [root-path]"
        )
    if argv[1] == "--wire-usd":
        if len(argv) < 4:
            raise SystemExit("usage: python -m core.usd_collision_discovery --wire-usd <asset-ref> <output.usda> [root-path]")
        payload = write_wire_overlay(argv[2], argv[3], argv[4] if len(argv) > 4 else "/World/Asset")
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    payload = discover(argv[1], argv[2] if len(argv) > 2 else "/World/Asset")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

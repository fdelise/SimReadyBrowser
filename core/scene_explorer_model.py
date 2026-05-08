"""Scene explorer data model for loaded SimReady assets.

The viewport renders assets under /SimReadyAsset, while the PhysX worker cooks
the same asset references under /World/Asset. This helper turns authored PhysX
discovery paths into render-stage paths so the UI can inspect and edit the
same prims the renderer shows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ASSET_ROOT = "/SimReadyAsset"
ASSET_ROOT_NAME = "SimReadyAsset"
PHYSICS_ASSET_ROOT = "/World/Asset"


def build_scene_tree(
    stage_items: list[dict],
    discoveries: list[dict] | None = None,
    bounds: dict | None = None,
) -> dict:
    """Build a serializable scene tree payload for the right-panel explorer."""
    normalized_items = _normalize_stage_items(stage_items)
    normalized_discoveries = _normalize_discoveries(discoveries, len(normalized_items))
    roots: list[dict] = []

    layout_transforms = _layout_transforms(bounds, len(normalized_items))
    for index, item in enumerate(normalized_items):
        render_root = asset_render_root(index)
        physics_root = physics_asset_root(index)
        discovery = normalized_discoveries[index] if index < len(normalized_discoveries) else {}
        root_props = _properties_from_matrix(layout_transforms[index] if index < len(layout_transforms) else None)
        root_props.update(
            {
                "source": item["source"],
                "key": item.get("key") or item["source"],
                "collider_count": int(discovery.get("collider_count", 0) or 0),
                "rigid_body_count": len(discovery.get("rigid_body_paths", []) or []),
                "body_count": len(discovery.get("body_paths", []) or []),
            }
        )
        root = {
            "name": item["name"] or _source_name(item["source"]) or render_root.rsplit("/", 1)[-1],
            "path": render_root,
            "physics_path": physics_root,
            "type": "Asset",
            "role": "asset",
            "properties": root_props,
            "children": [],
        }

        nodes = {render_root: root}
        role_by_physics_path = _roles_for_discovery(discovery, physics_root)
        for physics_path, role in role_by_physics_path.items():
            render_path = render_path_from_physics_path(physics_path, index)
            if not render_path or render_path == render_root:
                root["role"] = role if role != "xform" else root["role"]
                continue
            _ensure_node_path(nodes, root, render_path, physics_path, role)

        roots.append(root)

    status = "Load an asset to inspect the USD scene."
    if roots:
        part_count = sum(_count_nodes(root) for root in roots)
        status = f"{part_count} scene prims from {len(roots)} asset source{'s' if len(roots) != 1 else ''}."
    return {"status": status, "roots": roots}


def asset_render_root(index: int) -> str:
    return ASSET_ROOT if index <= 0 else f"/{ASSET_ROOT_NAME}_{index + 1:02d}"


def physics_asset_root(index: int) -> str:
    return PHYSICS_ASSET_ROOT if index <= 0 else f"/World/Instance_{index + 1:02d}"


def render_path_from_physics_path(path: str, asset_index: int = 0) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if not text.startswith("/"):
        text = "/" + text
    render_root = asset_render_root(asset_index)
    if text == PHYSICS_ASSET_ROOT:
        return render_root
    if text.startswith(f"{PHYSICS_ASSET_ROOT}/"):
        return render_root + text[len(PHYSICS_ASSET_ROOT) :]
    for index in range(1, 100):
        physics_root = physics_asset_root(index)
        if text == physics_root:
            return asset_render_root(index)
        if text.startswith(f"{physics_root}/"):
            return asset_render_root(index) + text[len(physics_root) :]
    if text == ASSET_ROOT or text.startswith(f"{ASSET_ROOT}/"):
        return text
    return ""


def _normalize_stage_items(items: list[dict]) -> list[dict]:
    result: list[dict] = []
    try:
        raw_items = list(items or [])
    except TypeError:
        raw_items = []
    for item in raw_items[:100]:
        if isinstance(item, dict):
            source = str(item.get("source") or item.get("usd_source") or item.get("path") or "").strip()
            name = str(item.get("name") or _source_name(source) or source).strip()
            key = str(item.get("key") or source).strip()
        else:
            source = str(item or "").strip()
            name = _source_name(source) or source
            key = source
        if source:
            result.append({"source": source, "name": name, "key": key})
    return result


def _normalize_discoveries(discoveries: list[dict] | None, count: int) -> list[dict]:
    result: list[dict] = []
    try:
        raw_items = list(discoveries or [])
    except TypeError:
        raw_items = []
    for item in raw_items[:count]:
        result.append(item if isinstance(item, dict) else {})
    while len(result) < count:
        result.append({})
    return result


def _roles_for_discovery(discovery: dict, physics_root: str) -> dict[str, str]:
    roles: dict[str, str] = {physics_root: "asset"}
    for path in discovery.get("body_paths", []) or []:
        _assign_role(roles, path, "body")
    for path in discovery.get("rigid_body_paths", []) or []:
        _assign_role(roles, path, "rigidBody")
    for path in discovery.get("articulation_paths", []) or []:
        _assign_role(roles, path, "articulation")
    return roles


def _assign_role(roles: dict[str, str], path: Any, role: str) -> None:
    text = str(path or "").strip()
    if not text:
        return
    if not text.startswith("/"):
        text = "/" + text
    priority = {"asset": 0, "xform": 1, "body": 2, "rigidBody": 3, "articulation": 4}
    current = roles.get(text)
    if current is None or priority.get(role, 0) >= priority.get(current, 0):
        roles[text] = role


def _ensure_node_path(
    nodes: dict[str, dict],
    root: dict,
    render_path: str,
    physics_path: str,
    role: str,
) -> dict:
    if render_path in nodes:
        node = nodes[render_path]
        node["role"] = role
        node["type"] = _type_from_role(role)
        node["physics_path"] = physics_path
        return node

    render_root = str(root.get("path") or ASSET_ROOT)
    suffix = render_path[len(render_root) :].strip("/")
    physics_root = str(root.get("physics_path") or PHYSICS_ASSET_ROOT)
    parent = root
    current_render = render_root
    current_physics = physics_root
    for part in [segment for segment in suffix.split("/") if segment]:
        current_render = f"{current_render}/{part}"
        current_physics = f"{current_physics}/{part}"
        node_role = role if current_render == render_path else "xform"
        node = nodes.get(current_render)
        if node is None:
            node = {
                "name": part,
                "path": current_render,
                "physics_path": current_physics,
                "type": _type_from_role(node_role),
                "role": node_role,
                "properties": _default_properties(),
                "children": [],
            }
            parent.setdefault("children", []).append(node)
            nodes[current_render] = node
        elif current_render == render_path:
            node["role"] = node_role
            node["type"] = _type_from_role(node_role)
            node["physics_path"] = physics_path
        parent = node
    return nodes[render_path]


def _type_from_role(role: str) -> str:
    return {
        "asset": "Asset",
        "rigidBody": "Rigid Body",
        "articulation": "Articulation",
        "body": "Physics Body",
    }.get(str(role or ""), "Xform")


def _default_properties() -> dict:
    return {
        "visible": True,
        "translate": [0.0, 0.0, 0.0],
        "rotate": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def _properties_from_matrix(matrix: Any) -> dict:
    props = _default_properties()
    if not matrix:
        return props
    try:
        rows = [list(row) for row in matrix][:4]
        if len(rows) != 4 or any(len(row) < 4 for row in rows):
            return props
        props["translate"] = [float(rows[3][0]), float(rows[3][1]), float(rows[3][2])]
        props["scale"] = [
            max(abs(float(rows[0][0])), 0.001),
            max(abs(float(rows[1][1])), 0.001),
            max(abs(float(rows[2][2])), 0.001),
        ]
    except Exception:
        return props
    return props


def _layout_transforms(bounds: dict | None, count: int) -> list[Any]:
    if count <= 0:
        return []
    if not isinstance(bounds, dict):
        return [None for _ in range(count)]
    try:
        raw = list(bounds.get("_asset_layout_transforms", []) or [])
    except TypeError:
        raw = []
    transforms = raw[:count]
    while len(transforms) < count:
        transforms.append(None)
    return transforms


def _source_name(source: str) -> str:
    text = str(source or "").split("?", 1)[0].rstrip("/")
    if not text:
        return ""
    return Path(text).name or text


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(child) for child in node.get("children", []) or [] if isinstance(child, dict))

"""Composed OpenUSD scene-property discovery for the Scene Explorer.

This helper runs in a separate process so the Qt/OVRTX process never imports
pxr directly. It opens the same inspection stage used by collision discovery,
walks composed prims including instance prototypes, and serializes schemas plus
USD attributes/relationships for the right-panel inspector.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from core.usd_collision_discovery import (
    _body_path_for,
    _body_patterns,
    _collider_type,
    _dedupe,
    _is_collision_prim,
    _iter_composed_prims,
    _open_inspection_stage,
    _rigid_body_paths,
    _articulation_root_paths,
    _token_attr,
)


MAX_VALUE_CHARS = 240
MAX_ARRAY_ITEMS = 8


def discover_scene(asset_ref: str, root_path: str = "/World/Asset") -> dict:
    from pxr import UsdPhysics

    _stage, root = _open_inspection_stage(asset_ref, root_path)
    entries = list(_iter_composed_prims(root))
    rigid_paths = _rigid_body_paths(entries, UsdPhysics)
    articulation_paths = _articulation_root_paths(entries, UsdPhysics)

    colliders = []
    body_paths = []
    type_counts: dict[str, int] = {}
    tree_nodes: dict[str, dict] = {}
    root_node: dict | None = None
    property_count = 0
    schema_names: set[str] = set()

    for prim, mapped_path in entries:
        node = _prim_node(prim, mapped_path, UsdPhysics, rigid_paths, articulation_paths, root_path)
        tree_nodes[mapped_path] = node
        property_count += len(node.get("usd_properties", []) or [])
        for schema in node.get("usd", {}).get("applied_schemas", []) or []:
            schema_names.add(str(schema))
        if mapped_path == root_path:
            root_node = node

        if _is_collision_prim(prim, UsdPhysics):
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
            body_paths.append(_body_path_for(mapped_path, rigid_paths, root_path))
            key = collider_type or "collision"
            type_counts[key] = type_counts.get(key, 0) + 1

    if root_node is None:
        root_node = _empty_root(root_path)

    for path, node in tree_nodes.items():
        if path == root_path:
            continue
        parent_path = path.rsplit("/", 1)[0]
        parent = tree_nodes.get(parent_path)
        if parent is None:
            parent = _ensure_parent_nodes(tree_nodes, root_node, root_path, parent_path)
        parent.setdefault("children", []).append(node)

    return {
        "prim_tree": root_node,
        "prim_count": len(tree_nodes),
        "property_count": property_count,
        "schema_count": len(schema_names),
        "colliders": colliders,
        "collider_count": len(colliders),
        "body_paths": _dedupe(body_paths) or [item["body_path"] for item in colliders if item.get("body_path")],
        "rigid_body_paths": rigid_paths,
        "articulation_paths": articulation_paths,
        "body_patterns": _body_patterns(root_path, _dedupe(body_paths)),
        "type_counts": type_counts,
    }


def _prim_node(prim, mapped_path: str, UsdPhysics, rigid_paths: list[str], articulation_paths: list[str], root_path: str) -> dict:
    name = str(prim.GetName() or mapped_path.rsplit("/", 1)[-1] or "Prim")
    type_name = str(prim.GetTypeName() or "Prim")
    schemas = [str(schema) for schema in list(prim.GetAppliedSchemas())]
    usd_properties = _usd_properties(prim)
    usd = {
        "type_name": type_name,
        "specifier": _safe_call(lambda: str(prim.GetSpecifier()), ""),
        "kind": _prim_kind(prim),
        "active": bool(_safe_call(prim.IsActive, False)),
        "defined": bool(_safe_call(prim.IsDefined, False)),
        "loaded": bool(_safe_call(prim.IsLoaded, False)),
        "abstract": bool(_safe_call(prim.IsAbstract, False)),
        "instance": bool(_safe_call(prim.IsInstance, False)),
        "instanceable": bool(_safe_call(prim.IsInstanceable, False)),
        "prototype": _prototype_path(prim),
        "applied_schemas": schemas,
        "property_count": len(usd_properties),
        "metadata": _selected_metadata(prim),
        "geometry": _geometry_summary(prim),
        "materials": _material_summary(prim),
        "physics": _physics_summary(prim, UsdPhysics, schemas, mapped_path, rigid_paths, articulation_paths, root_path),
    }
    role = _role_from_usd(usd)
    return {
        "name": name,
        "path": mapped_path,
        "physics_path": mapped_path,
        "type": _display_type(type_name, role),
        "role": role,
        "properties": _xform_edit_defaults(prim),
        "usd": usd,
        "usd_properties": usd_properties,
        "children": [],
    }


def _usd_properties(prim) -> list[dict]:
    result: list[dict] = []
    for prop in list(prim.GetProperties() or []):
        name = str(prop.GetName())
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid():
            entry = {
                "name": name,
                "kind": "attribute",
                "type": str(attr.GetTypeName()),
                "variability": _safe_call(lambda: str(attr.GetVariability()), ""),
                "custom": bool(_safe_call(attr.IsCustom, False)),
                "authored": bool(_safe_call(attr.HasAuthoredValueOpinion, False)),
                "time_samples": len(_safe_call(attr.GetTimeSamples, []) or []),
                "connections": [str(path) for path in (_safe_call(attr.GetConnections, []) or [])],
                "value": _value_preview(_safe_call(attr.Get, None)),
            }
            result.append(entry)
            continue

        rel = prim.GetRelationship(name)
        if rel and rel.IsValid():
            targets = [str(path) for path in (_safe_call(rel.GetTargets, []) or [])]
            result.append(
                {
                    "name": name,
                    "kind": "relationship",
                    "type": "relationship",
                    "custom": bool(_safe_call(rel.IsCustom, False)),
                    "authored": bool(_safe_call(rel.HasAuthoredTargets, False)),
                    "targets": targets,
                    "value": ", ".join(targets),
                }
            )
    return sorted(result, key=lambda item: (str(item.get("kind")), str(item.get("name"))))


def _geometry_summary(prim) -> dict:
    type_name = str(prim.GetTypeName() or "")
    geom: dict[str, Any] = {}
    if type_name:
        geom["schema"] = type_name

    for attr_name in ("visibility", "purpose", "extent", "orientation", "subdivisionScheme"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            geom[attr_name] = _value_preview(_safe_call(attr.Get, None))

    if type_name == "Mesh":
        points = _safe_call(lambda: prim.GetAttribute("points").Get(), None)
        face_counts = _safe_call(lambda: prim.GetAttribute("faceVertexCounts").Get(), None)
        face_indices = _safe_call(lambda: prim.GetAttribute("faceVertexIndices").Get(), None)
        normals = _safe_call(lambda: prim.GetAttribute("normals").Get(), None)
        geom["points"] = _sequence_len(points)
        geom["faces"] = _sequence_len(face_counts)
        geom["face_indices"] = _sequence_len(face_indices)
        geom["normals"] = _sequence_len(normals)
    return {key: value for key, value in geom.items() if value not in ("", None)}


def _material_summary(prim) -> dict:
    from pxr import UsdShade

    summary: dict[str, Any] = {}
    try:
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and material.GetPrim() and material.GetPrim().IsValid():
            summary["bound_material"] = str(material.GetPrim().GetPath())
        if relationship and relationship.IsValid():
            summary["binding_relationship"] = str(relationship.GetPath())
    except Exception:
        pass

    type_name = str(prim.GetTypeName() or "")
    if type_name in {"Material", "Shader"}:
        summary["schema"] = type_name
        inputs = [prop.get("name") for prop in _usd_properties(prim) if str(prop.get("name", "")).startswith("inputs:")]
        outputs = [prop.get("name") for prop in _usd_properties(prim) if str(prop.get("name", "")).startswith("outputs:")]
        if inputs:
            summary["inputs"] = ", ".join(str(item) for item in inputs)
        if outputs:
            summary["outputs"] = ", ".join(str(item) for item in outputs)
    return summary


def _physics_summary(
    prim,
    UsdPhysics,
    schemas: list[str],
    mapped_path: str,
    rigid_paths: list[str],
    articulation_paths: list[str],
    root_path: str,
) -> dict:
    physics: dict[str, Any] = {}
    lower_schemas = {schema.lower() for schema in schemas}
    is_collision = _is_collision_prim(prim, UsdPhysics)
    is_rigid = mapped_path in rigid_paths or "physicsrigidbodyapi" in lower_schemas
    is_articulation = mapped_path in articulation_paths or any("articulation" in schema for schema in lower_schemas)
    if is_collision:
        physics["collision"] = True
        physics["body_path"] = _body_path_for(mapped_path, rigid_paths, root_path)
    if is_rigid:
        physics["rigid_body"] = True
    if is_articulation:
        physics["articulation"] = True

    for prop in _usd_properties(prim):
        name = str(prop.get("name", ""))
        if _is_physics_property_name(name):
            physics[name] = prop.get("value", "")
    return physics


def _xform_edit_defaults(prim) -> dict:
    translate = [0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0]
    try:
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        if xformable:
            matrix = xformable.GetLocalTransformation()
            if isinstance(matrix, tuple):
                matrix = matrix[0]
            rows = [[float(matrix[i][j]) for j in range(4)] for i in range(4)]
            translate = [rows[3][0], rows[3][1], rows[3][2]]
            scale = [
                max(_row_length(rows[0][:3]), 0.001),
                max(_row_length(rows[1][:3]), 0.001),
                max(_row_length(rows[2][:3]), 0.001),
            ]
    except Exception:
        pass
    return {"visible": True, "translate": translate, "rotate": [0.0, 0.0, 0.0], "scale": scale}


def _selected_metadata(prim) -> dict:
    result: dict[str, str] = {}
    for key in ("kind", "documentation", "assetInfo", "references", "payload"):
        try:
            value = prim.GetMetadata(key)
        except Exception:
            continue
        if value is not None:
            result[key] = _value_preview(value)
    return result


def _prim_kind(prim) -> str:
    try:
        from pxr import Usd

        return str(Usd.ModelAPI(prim).GetKind() or "")
    except Exception:
        return ""


def _prototype_path(prim) -> str:
    try:
        prototype = prim.GetPrototype()
    except Exception:
        return ""
    if prototype and prototype.IsValid():
        return str(prototype.GetPath())
    return ""


def _display_type(type_name: str, role: str) -> str:
    if role == "rigidBody":
        return f"{type_name or 'Prim'} + Rigid Body"
    if role == "collider":
        return f"{type_name or 'Prim'} + Collider"
    if role == "material":
        return "Material"
    return type_name or "Prim"


def _role_from_usd(usd: dict) -> str:
    physics = usd.get("physics") if isinstance(usd.get("physics"), dict) else {}
    if physics.get("articulation"):
        return "articulation"
    if physics.get("rigid_body"):
        return "rigidBody"
    if physics.get("collision"):
        return "collider"
    if str(usd.get("type_name", "")) == "Material":
        return "material"
    return "xform"


def _is_physics_property_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return lowered.startswith("physics:") or lowered.startswith("physx") or lowered.startswith("physx:")


def _value_preview(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, bool)):
        return _trim(str(value))
    if isinstance(value, float):
        return _trim(f"{value:.9g}" if math.isfinite(value) else str(value))
    if isinstance(value, dict):
        items = [f"{key}: {_value_preview(item)}" for key, item in list(value.items())[:MAX_ARRAY_ITEMS]]
        suffix = ", ..." if len(value) > MAX_ARRAY_ITEMS else ""
        return _trim("{" + ", ".join(items) + suffix + "}")
    length = _sequence_len(value)
    if length >= 0 and not isinstance(value, (str, bytes)):
        try:
            sample = [value[index] for index in range(min(length, MAX_ARRAY_ITEMS))]
        except Exception:
            try:
                sample = list(value)[:MAX_ARRAY_ITEMS]
            except Exception:
                sample = []
        suffix = ", ..." if length > MAX_ARRAY_ITEMS else ""
        return _trim(f"[{length}] " + ", ".join(_value_preview(item) for item in sample) + suffix)
    return _trim(str(value))


def _sequence_len(value: Any) -> int:
    if value is None:
        return -1
    try:
        return len(value)
    except Exception:
        return -1


def _safe_call(fn, fallback):
    try:
        return fn()
    except Exception:
        return fallback


def _row_length(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _trim(text: str) -> str:
    value = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    if len(value) <= MAX_VALUE_CHARS:
        return value
    return value[: MAX_VALUE_CHARS - 3] + "..."


def _empty_root(root_path: str) -> dict:
    name = root_path.rsplit("/", 1)[-1] or "Asset"
    return {
        "name": name,
        "path": root_path,
        "physics_path": root_path,
        "type": "Asset",
        "role": "asset",
        "properties": {"visible": True, "translate": [0.0, 0.0, 0.0], "rotate": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "usd": {"type_name": "Asset", "applied_schemas": [], "property_count": 0},
        "usd_properties": [],
        "children": [],
    }


def _ensure_parent_nodes(tree_nodes: dict[str, dict], root_node: dict, root_path: str, parent_path: str) -> dict:
    if parent_path in tree_nodes:
        return tree_nodes[parent_path]
    if not parent_path.startswith(f"{root_path}/"):
        return root_node
    current = root_node
    current_path = root_path
    for part in parent_path[len(root_path) :].strip("/").split("/"):
        current_path = f"{current_path}/{part}"
        node = tree_nodes.get(current_path)
        if node is None:
            node = _empty_root(current_path)
            node["name"] = part
            node["type"] = "Prim"
            node["role"] = "xform"
            tree_nodes[current_path] = node
            current.setdefault("children", []).append(node)
        current = node
    return current


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m core.usd_scene_discovery <asset-ref> [root-path]\n"
            "   or: python -m core.usd_scene_discovery --multi <asset-ref>..."
        )

    if argv[1] == "--multi":
        refs = [item for item in argv[2:] if str(item or "").strip()]
        payload = {"discoveries": [discover_scene(ref, "/World/Asset") for ref in refs]}
    else:
        payload = discover_scene(argv[1], argv[2] if len(argv) > 2 else "/World/Asset")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

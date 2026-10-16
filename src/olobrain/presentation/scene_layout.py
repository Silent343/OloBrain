"""Stable model-space placement shared by rendering, GPU picking, and tests."""

from __future__ import annotations

import math
import numpy as np

from olobrain.domain.model import AnatomySystem, BrainPartId, BrainSnapshot, anatomy_system_for


def visible_mesh(mesh, snapshot: BrainSnapshot) -> bool:
    """Choose a paired exterior/interior surface without changing its anatomical coordinates."""
    if anatomy_system_for(mesh.part_id) not in snapshot.visible_systems:
        return False
    if mesh.view == "interior" and not snapshot.interior_view:
        return False
    if mesh.view == "exterior" and snapshot.interior_view:
        return False
    return (
        mesh.default_visible
        or snapshot.interior_view
        or snapshot.explosion > 0.035
        or snapshot.selected_part == mesh.part_id
    )


def scene_rotation(snapshot: BrainSnapshot) -> np.ndarray:
    """Return the exact same rotation matrix for every structure, including the selected one."""
    x, y, z = snapshot.rotation_x, snapshot.rotation_y, snapshot.rotation_z
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float32)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float32)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rz @ rx @ ry
    return result


def placement_matrix(
    snapshot: BrainSnapshot,
    explosion,
    part: BrainPartId,
    scale: float,
    element_key: str = "",
) -> np.ndarray:
    """Selection never changes position, scale, rotation, or explosion offsets."""
    shift = np.asarray(explosion, dtype=np.float32) * snapshot.explosion * 0.85
    for offset in snapshot.fragment_offsets:
        if offset.part_id == part and (not element_key or offset.element_key in ("", element_key)):
            shift += np.asarray(
                (offset.horizontal * 4.0, -offset.vertical * 4.0, offset.depth * 3.0),
                dtype=np.float32,
            )
            break
    shift[1] -= 0.04
    matrix = scene_rotation(snapshot)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = matrix[:3, :3] @ shift
    return matrix


def object_drag(
    snapshot: BrainSnapshot, dx: float, dy: float, depth: float
) -> tuple[float, float, float]:
    """Convert camera-space hand/mouse motion into persistent rotating model offsets."""
    delta = scene_rotation(snapshot)[:3, :3].T @ np.asarray((dx * 4.0, -dy * 4.0, depth * 3.0))
    return float(delta[0] / 4.0), float(-delta[1] / 4.0), float(delta[2] / 3.0)

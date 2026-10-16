"""Procedural 3D neuroanatomical geometry primitives independent from the desktop renderer.

Coordinates follow standard neurosurgical convention:
- Anterior (Frontal pole) = +Z
- Posterior (Occipital pole) = -Z
- Superior (Vertex) = +Y
- Inferior (Skull base / Foramen magnum) = -Y
- Anatomical Left (Dominant hemisphere) = +X
- Anatomical Right = -X
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from olobrain.domain.model import BrainPartId


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        length = max(self.length(), 1e-9)
        return self * (1.0 / length)

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@dataclass(frozen=True, slots=True)
class AnatomicalMesh:
    part_id: BrainPartId
    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]
    analysis_color: str
    realistic_color: str
    explosion_vector: Vec3
    tissue: bool = True
    default_visible: bool = True
    view: str = "all"
    motion: str = "none"
    opacity: float = 1.0
    element_key: str = ""
    element_name: str = ""
    deformation_center: Vec3 | None = None
    valve_center: Vec3 | None = None
    valve_radius: float = 0.0
    support_motion: tuple[tuple[float, ...], ...] = ()


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def shade(color: str, intensity: float) -> str:
    """Shade a material color while preserving low-light color information."""
    red, green, blue = _hex(color)
    gain = max(0.16, min(1.35, intensity))
    ambient = 5
    return (
        f"#{min(255, int(red * gain) + ambient):02x}"
        f"{min(255, int(green * gain) + ambient):02x}"
        f"{min(255, int(blue * gain) + ambient):02x}"
    )


def _curve_points(control_points: list[Vec3], steps_per_segment: int = 8) -> list[Vec3]:
    """Catmull-Rom spline interpolation for smooth axonal tracts, gyri, and cerebral vessels."""
    if len(control_points) < 2:
        return list(control_points)
    if len(control_points) == 2:
        a, b = control_points
        return [a + (b - a) * (i / max(1, steps_per_segment)) for i in range(steps_per_segment + 1)]

    extended = [control_points[0], *control_points, control_points[-1]]
    samples: list[Vec3] = []
    for idx in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[idx - 1], extended[idx], extended[idx + 1], extended[idx + 2]
        limit = steps_per_segment if idx < len(extended) - 3 else steps_per_segment + 1
        for step in range(limit):
            t = step / steps_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1.x)
                + (-p0.x + p2.x) * t
                + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
            )
            y = 0.5 * (
                (2.0 * p1.y)
                + (-p0.y + p2.y) * t
                + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
            )
            z = 0.5 * (
                (2.0 * p1.z)
                + (-p0.z + p2.z) * t
                + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
                + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3
            )
            samples.append(Vec3(x, y, z))
    return samples


def _ellipsoid(
    part: BrainPartId,
    center: Vec3,
    radii: Vec3,
    colors: tuple[str, str],
    explosion: Vec3,
    rows: int = 12,
    columns: int = 20,
    tissue: bool = True,
    default_visible: bool = True,
    folia_freq: float = 0.0,
    gyral_freq: float = 0.0,
    element_key: str = "",
    element_name: str = "",
    view: str = "all",
    opacity: float = 1.0,
) -> AnatomicalMesh:
    """Create a sculpted 3D anatomical ellipsoid with optional cerebellar folia or cortical gyri."""
    vertices: list[Vec3] = []
    for row in range(rows + 1):
        latitude = math.pi * row / rows
        sine = math.sin(latitude)
        cosine = math.cos(latitude)
        for column in range(columns):
            longitude = math.tau * column / columns
            grain = 1.0
            if tissue:
                grain += 0.018 * math.sin(longitude * 5.0 + latitude * 4.0) * sine * sine
            if folia_freq > 0.0:
                # Horizontal cerebellar folia ridges
                grain += 0.042 * math.sin(latitude * folia_freq) * sine
            if gyral_freq > 0.0:
                # Cortical sulcal-gyral convolutions
                grain += (
                    0.045
                    * (
                        math.sin(longitude * gyral_freq + math.cos(latitude * 4.0) * 1.6)
                        + 0.5 * math.cos(latitude * (gyral_freq * 0.85) - longitude * 2.0)
                    )
                    * sine
                )
            vertices.append(
                Vec3(
                    center.x + radii.x * grain * sine * math.cos(longitude),
                    center.y + radii.y * grain * cosine,
                    center.z + radii.z * grain * sine * math.sin(longitude),
                )
            )
    faces: list[tuple[int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            next_column = (column + 1) % columns
            a = row * columns + column
            b = row * columns + next_column
            c = (row + 1) * columns + column
            d = (row + 1) * columns + next_column
            faces.append((a, c, b))
            faces.append((b, c, d))
    return AnatomicalMesh(
        part_id=part,
        vertices=tuple(vertices),
        faces=tuple(faces),
        analysis_color=colors[0],
        realistic_color=colors[1],
        explosion_vector=explosion,
        tissue=tissue,
        default_visible=default_visible,
        view=view,
        opacity=opacity,
        element_key=element_key or part.value,
        element_name=element_name,
        deformation_center=center,
    )


def _tube(
    part: BrainPartId,
    path: list[Vec3],
    radius: float,
    colors: tuple[str, str],
    explosion: Vec3,
    sides: int = 14,
    tissue: bool = False,
    default_visible: bool = True,
    curve_steps: int = 5,
    taper: float = 0.0,
    gyral_bump: float = 0.0,
    element_key: str = "",
    element_name: str = "",
    view: str = "all",
    opacity: float = 1.0,
) -> AnatomicalMesh:
    """Sweep a smooth parallel-transported cross-section along a 3D spline with hemispherical endcaps."""
    samples = _curve_points(path, curve_steps) if len(path) > 2 else _curve_points(path, max(2, curve_steps))
    vertices: list[Vec3] = []
    faces: list[tuple[int, int, int]] = []

    tangents: list[Vec3] = []
    for index in range(len(samples)):
        if index == 0:
            tangent = (samples[1] - samples[0]).normalized()
        elif index == len(samples) - 1:
            tangent = (samples[-1] - samples[-2]).normalized()
        else:
            tangent = (samples[index + 1] - samples[index - 1]).normalized()
        tangents.append(tangent)

    up = Vec3(0.0, 1.0, 0.0)
    if abs(tangents[0].dot(up)) > 0.92:
        up = Vec3(1.0, 0.0, 0.0)
    normal = tangents[0].cross(up).normalized()
    binormal = tangents[0].cross(normal).normalized()

    for index, point in enumerate(samples):
        if index > 0:
            prev_t = tangents[index - 1]
            curr_t = tangents[index]
            axis = prev_t.cross(curr_t)
            sin_angle = axis.length()
            cos_angle = max(-1.0, min(1.0, prev_t.dot(curr_t)))
            if sin_angle > 1e-6:
                axis_n = axis * (1.0 / sin_angle)
                normal = (
                    normal * cos_angle
                    + axis_n.cross(normal) * sin_angle
                    + axis_n * (axis_n.dot(normal) * (1.0 - cos_angle))
                ).normalized()
                binormal = curr_t.cross(normal).normalized()

        progress = index / max(1, len(samples) - 1)
        # Smooth organic tapering at ends for gyral parcels
        envelope = 1.0 - taper * progress
        if gyral_bump > 0.0:
            envelope *= 0.78 + 0.28 * math.sin(progress * math.pi)
            envelope *= 1.0 + gyral_bump * math.sin(progress * math.pi * 5.0)
        local_r = max(0.004, radius * envelope)

        for side in range(sides):
            theta = math.tau * side / sides
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Slightly flattened oval cross-section for cortical gyri
            scale_n = 1.12 if gyral_bump > 0.0 else 1.0
            scale_b = 0.88 if gyral_bump > 0.0 else 1.0
            offset = normal * (local_r * scale_n * cos_t) + binormal * (local_r * scale_b * sin_t)
            vertices.append(point + offset)

    for ring in range(len(samples) - 1):
        base = ring * sides
        next_base = (ring + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            a = base + side
            b = base + next_side
            c = next_base + side
            d = next_base + next_side
            faces.append((a, c, b))
            faces.append((b, c, d))

    # Endcap vertices to ensure closed, watertight meshes
    start_center_idx = len(vertices)
    vertices.append(samples[0] - tangents[0] * (radius * 0.28))
    end_center_idx = len(vertices)
    vertices.append(samples[-1] + tangents[-1] * (radius * 0.28))

    last_ring_base = (len(samples) - 1) * sides
    for side in range(sides):
        next_side = (side + 1) % sides
        faces.append((start_center_idx, side, next_side))
        faces.append((end_center_idx, last_ring_base + next_side, last_ring_base + side))

    mid = samples[len(samples) // 2]
    return AnatomicalMesh(
        part_id=part,
        vertices=tuple(vertices),
        faces=tuple(faces),
        analysis_color=colors[0],
        realistic_color=colors[1],
        explosion_vector=explosion,
        tissue=tissue,
        default_visible=default_visible,
        view=view,
        opacity=opacity,
        element_key=element_key or part.value,
        element_name=element_name,
        deformation_center=mid,
    )


def _torus(
    part: BrainPartId,
    center: Vec3,
    major_radius: float,
    minor_radius: float,
    colors: tuple[str, str],
    explosion: Vec3,
    tilt: float = 0.0,
    default_visible: bool = True,
    element_key: str = "",
    element_name: str = "",
) -> AnatomicalMesh:
    rings, sides = 20, 12
    vertices: list[Vec3] = []
    cos_tilt, sin_tilt = math.cos(tilt), math.sin(tilt)
    for ring in range(rings):
        u = math.tau * ring / rings
        cu, su = math.cos(u), math.sin(u)
        for side in range(sides):
            v = math.tau * side / sides
            cv, sv = math.cos(v), math.sin(v)
            r = major_radius + minor_radius * cv
            lx = r * cu
            ly = minor_radius * sv
            lz = r * su
            ry = ly * cos_tilt - lz * sin_tilt
            rz = ly * sin_tilt + lz * cos_tilt
            vertices.append(Vec3(center.x + lx, center.y + ry, center.z + rz))
    faces: list[tuple[int, int, int]] = []
    for ring in range(rings):
        next_ring = (ring + 1) % rings
        for side in range(sides):
            next_side = (side + 1) % sides
            a = ring * sides + side
            b = ring * sides + next_side
            c = next_ring * sides + side
            d = next_ring * sides + next_side
            faces.append((a, c, b))
            faces.append((b, c, d))
    return AnatomicalMesh(
        part_id=part,
        vertices=tuple(vertices),
        faces=tuple(faces),
        analysis_color=colors[0],
        realistic_color=colors[1],
        explosion_vector=explosion,
        tissue=False,
        default_visible=default_visible,
        element_key=element_key or part.value,
        element_name=element_name,
        deformation_center=center,
    )


def build_brain_geometry() -> tuple[AnatomicalMesh, ...]:
    """Build the complete 139-part 3D neuroanatomical atlas."""
    from olobrain.infrastructure.brain_atlas import build_brain_atlas

    return build_brain_atlas()


# Compatibility alias
build_heart_geometry = build_brain_geometry

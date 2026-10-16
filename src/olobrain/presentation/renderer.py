"""High-resolution z-buffered 3D software renderer v2.0 with Sulcal Shading, RBCs, and Synaptic Impulses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from olobrain.domain.model import AnatomySystem, BrainPartId, BrainSnapshot, anatomy_system_for
from olobrain.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from olobrain.infrastructure.geometry import AnatomicalMesh
from olobrain.presentation.scene_layout import placement_matrix, visible_mesh


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


class BrainSoftwareRenderer:
    """High-resolution 3D software renderer with realistic cortical sulcal creases, RBCs, and neural spikes."""

    def __init__(self, meshes: tuple[AnatomicalMesh, ...]) -> None:
        self._meshes = meshes
        self._routes = build_flow_routes()

    def render_frame(
        self,
        snapshot: BrainSnapshot,
        width: int = 1280,
        height: int = 800,
        hovered_part: BrainPartId | None = None,
    ) -> Image.Image:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width * 0.50, height * 0.50
        dist = np.sqrt(((xx - cx) / (width * 0.58)) ** 2 + ((yy - cy) / (height * 0.58)) ** 2)
        glow = np.clip(1.0 - dist, 0.0, 1.0)
        img[:, :, 0] = (5 + glow * 14).astype(np.uint8)
        img[:, :, 1] = (12 + glow * 26).astype(np.uint8)
        img[:, :, 2] = (22 + glow * 38).astype(np.uint8)

        scale = 1.18 * snapshot.zoom / (1.0 + snapshot.explosion * 0.42)
        fov_factor = min(width, height) * 0.68
        cam_z = 3.9

        light_key = np.array([-0.38, 0.65, 0.88], dtype=np.float32)
        light_key /= np.linalg.norm(light_key)
        light_fill = np.array([0.62, -0.22, 0.58], dtype=np.float32)
        light_fill /= np.linalg.norm(light_fill)

        opaque_tris: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        trans_tris: list[tuple[float, np.ndarray, tuple[int, int, int], float]] = []

        for mesh in self._meshes:
            if not visible_mesh(mesh, snapshot):
                continue
            system = anatomy_system_for(mesh.part_id)
            opacity = mesh.opacity
            if hovered_part == mesh.part_id and snapshot.explosion < 0.05 and system in {
                AnatomySystem.CEREBRAL_CORTEX_LOBES,
                AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
            }:
                opacity = 0.35
            elif circulation_visible(snapshot) and system in {
                AnatomySystem.CEREBRAL_CORTEX_LOBES,
                AnatomySystem.MENINGES_AND_VENTRICLES,
            }:
                opacity = min(opacity, 0.28)

            verts = np.asarray([(v.x, v.y, v.z) for v in mesh.vertices], dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            model = placement_matrix(
                snapshot,
                (mesh.explosion_vector.x, mesh.explosion_vector.y, mesh.explosion_vector.z),
                mesh.part_id,
                scale,
                mesh.element_key,
            )
            rot = model[:3, :3]
            trans = model[:3, 3]
            world = verts @ rot.T + trans

            v0_obj = verts[faces[:, 0]]
            v1_obj = verts[faces[:, 1]]
            v2_obj = verts[faces[:, 2]]
            obj_centers = (v0_obj + v1_obj + v2_obj) / 3.0

            v0 = world[faces[:, 0]]
            v1 = world[faces[:, 1]]
            v2 = world[faces[:, 2]]
            area_normals = np.cross(v1 - v0, v2 - v0)
            v_normals = np.zeros_like(world, dtype=np.float32)
            np.add.at(v_normals, faces[:, 0], area_normals)
            np.add.at(v_normals, faces[:, 1], area_normals)
            np.add.at(v_normals, faces[:, 2], area_normals)
            normals = (v_normals[faces[:, 0]] + v_normals[faces[:, 1]] + v_normals[faces[:, 2]]) / 3.0
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(lengths, 1e-8)
            flip_mask = normals[:, 2] < 0
            normals[flip_mask] *= -1.0

            centers = (v0 + v1 + v2) / 3.0
            z_cam = cam_z - centers[:, 2]
            valid = z_cam > 0.25
            if not np.any(valid):
                continue

            hex_col = (mesh.realistic_color if snapshot.realistic else mesh.analysis_color).lstrip("#")
            base_rgb = np.array(
                [int(hex_col[0:2], 16), int(hex_col[2:4], 16), int(hex_col[4:6], 16)],
                dtype=np.float32,
            )
            if snapshot.selected_part == mesh.part_id or hovered_part == mesh.part_id:
                base_rgb = np.minimum(255.0, base_rgb * 1.18 + np.array([12.0, 42.0, 52.0]))

            # Smooth vertex-averaged sulcal & cerebellar folia shading (zero per-triangle aliasing)
            vx_o, vy_o, vz_o = verts[:, 0], verts[:, 1], verts[:, 2]
            if mesh.tissue and "cerebell" in mesh.element_key:
                folia_w = np.sin((vy_o + 0.34) * 68.0)
                v_sulcus = np.exp(-((folia_w / 0.30) ** 2))
            elif mesh.tissue and system in {
                AnatomySystem.CEREBRAL_CORTEX_LOBES,
                AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
            }:
                wx = vx_o + 0.12 * np.sin(vy_o * 3.6 + vz_o * 3.2)
                wy = vy_o + 0.12 * np.cos(vz_o * 3.5 - vx_o * 3.2)
                wz = vz_o + 0.12 * np.sin(vx_o * 3.8 + vy_o * 3.4)
                w1 = np.sin(wx * 10.5 + wy * 9.8 + 1.35 * np.cos(wz * 9.5 - wy * 8.2))
                w2 = np.cos(wz * 10.8 - wy * 9.4 + 1.25 * np.sin(wx * 10.2 + wz * 8.8))
                s1 = np.exp(-((w1 / 0.25) ** 2))
                s2 = np.exp(-((w2 / 0.22) ** 2)) * (1.0 - 0.5 * s1)
                sylvian = np.exp(-(((vy_o - (-0.08 - 0.18 * vz_o)) / 0.055) ** 2)) * (np.abs(vx_o) > 0.22) * (vz_o > -0.30) * (vz_o < 0.44)
                v_sulcus = np.minimum(1.0, np.maximum(np.maximum(s1, s2 * 0.82), sylvian))
            else:
                v_sulcus = np.zeros_like(vx_o)
            sulcus_crease = (v_sulcus[faces[:, 0]] + v_sulcus[faces[:, 1]] + v_sulcus[faces[:, 2]]) / 3.0

            diff_key = np.maximum(0.0, normals @ light_key)
            diff_fill = np.maximum(0.0, normals @ light_fill)
            spec = np.power(np.maximum(0.0, normals[:, 2]), 28.0) * (36.0 if mesh.tissue else 20.0)
            intensity = (0.24 + 0.72 * diff_key + 0.26 * diff_fill) * (1.0 - 0.34 * sulcus_crease)

            vz = np.maximum(0.25, cam_z - world[:, 2])
            sx = cx + (world[:, 0] / vz) * fov_factor
            sy = cy - (world[:, 1] / vz) * fov_factor
            screen_pts = np.stack([sx, sy], axis=1).astype(np.int32)

            for idx in np.where(valid)[0]:
                f = faces[idx]
                pts = screen_pts[f]
                col_arr = np.clip(base_rgb * intensity[idx] + spec[idx], 0, 255).astype(int)
                if snapshot.realistic and mesh.tissue and sulcus_crease[idx] > 0.05:
                    sb = float(sulcus_crease[idx]) ** 1.35
                    col_arr[0] = int(col_arr[0] * (1.0 - 0.22 * sb) + 98.0 * sb * 0.22)
                    col_arr[1] = int(col_arr[1] * (1.0 - 0.44 * sb) + 32.0 * sb * 0.44)
                    col_arr[2] = int(col_arr[2] * (1.0 - 0.46 * sb) + 28.0 * sb * 0.46)
                col_tuple = (int(col_arr[0]), int(col_arr[1]), int(col_arr[2]))
                if opacity >= 0.92:
                    opaque_tris.append((float(centers[idx, 2]), pts, col_tuple))
                else:
                    trans_tris.append((float(centers[idx, 2]), pts, col_tuple, opacity))

        opaque_tris.sort(key=lambda item: item[0])
        for _, pts, col in opaque_tris:
            cv2.fillConvexPoly(img, pts, col, lineType=cv2.LINE_AA)

        # Render 3D Red Blood Cells (Erythrocytes) & Synaptic Action Potentials before translucent cortex
        if circulation_visible(snapshot) or hovered_part is not None:
            model0 = placement_matrix(snapshot, (0.0, 0.0, 0.0), BrainPartId.THALAMUS_L, scale)
            rot0, trans0 = model0[:3, :3], model0[:3, 3]
            for route in self._routes:
                if hovered_part is not None and not snapshot.flow_enabled:
                    if hovered_part not in route.parts and anatomy_system_for(hovered_part) not in {
                        anatomy_system_for(p) for p in route.parts
                    }:
                        continue
                pts_w = route.points @ rot0.T + trans0
                vz = np.maximum(0.25, cam_z - pts_w[:, 2])
                sx = (cx + (pts_w[:, 0] / vz) * fov_factor).astype(np.int32)
                sy = (cy - (pts_w[:, 1] / vz) * fov_factor).astype(np.int32)
                poly = np.stack([sx, sy], axis=1).reshape(-1, 1, 2)
                if route.kind == "vascular":
                    cv2.polylines(img, [poly], False, (180, 38, 42), 2, lineType=cv2.LINE_AA)
                    for p_idx in range(6, 96, 10):
                        px, py = int(sx[p_idx]), int(sy[p_idx])
                        # Biconcave erythrocyte disc
                        cv2.ellipse(img, (px, py), (6, 4), 25, 0, 360, (238, 42, 46), -1, lineType=cv2.LINE_AA)
                        cv2.ellipse(img, (px, py), (3, 2), 25, 0, 360, (135, 18, 22), -1, lineType=cv2.LINE_AA)
                else:
                    cv2.polylines(img, [poly], False, (35, 195, 235), 1, lineType=cv2.LINE_AA)
                    for p_idx in range(8, 95, 14):
                        px, py = int(sx[p_idx]), int(sy[p_idx])
                        cv2.circle(img, (px, py), 6, (40, 220, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(img, (px, py), 3, (255, 245, 130), -1, lineType=cv2.LINE_AA)

        if trans_tris:
            overlay = img.copy()
            trans_tris.sort(key=lambda item: item[0])
            for _, pts, col, _ in trans_tris:
                cv2.fillConvexPoly(overlay, pts, col, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.24, img, 0.76, 0, img)

        return Image.fromarray(img)


HeartRenderer = BrainSoftwareRenderer

"""Continuous volumetric 3D brain surface sculpting with seamless gyral/sulcal relief."""

from __future__ import annotations

import math
from dataclasses import replace

from olobrain.domain.model import BrainPartId as I
from olobrain.infrastructure.geometry import AnatomicalMesh, Vec3, _curve_points, _ellipsoid, _tube


def cortical_sulcus_field(x: float, y: float, z: float, side: float) -> tuple[float, float]:
    """Evaluate smooth, organic 3D gyral crowns and sulcal valleys at (x, y, z)."""
    wx = x + 0.12 * math.sin(y * 3.6 + z * 3.2 + side * 0.4)
    wy = y + 0.12 * math.cos(z * 3.5 - x * 3.2)
    wz = z + 0.12 * math.sin(x * 3.8 + y * 3.4)

    # 1. Deep Sylvian Fissure (Lateral Sulcus)
    sylvian_y = -0.08 - 0.18 * z
    sylvian_fissure = (
        math.exp(-(((y - sylvian_y) / 0.055) ** 2))
        * (1.0 / (1.0 + math.exp(-(z + 0.30) * 12.0)))
        * (1.0 / (1.0 + math.exp((z - 0.44) * 12.0)))
        * max(0.0, min(1.0, (abs(x) - 0.22) * 2.8))
    )

    # 2. Central Sulcus of Rolando, Precentral & Postcentral Gyri
    rolandic_coord = wz + 0.04 - 0.18 * wy
    central_weight = math.exp(-(((z + 0.04) / 0.32) ** 2)) * max(0.0, min(1.0, (y + 0.05) * 1.5))
    rolandic_wave = math.sin(rolandic_coord * 9.5)

    # 3. Frontal Lobe longitudinal meandering gyri (Superior, Middle, Inferior Frontal)
    frontal_weight = 1.0 / (1.0 + math.exp(-(z - 0.12) * 9.0))
    frontal_wave = math.sin(wy * 8.5 + 1.10 * math.sin(wz * 5.8 + wx * 3.8))

    # 4. Temporal Lobe parallel oblique gyri (Superior, Middle, Inferior Temporal)
    temporal_weight = (1.0 / (1.0 + math.exp((y + 0.04) * 11.0))) * (1.0 / (1.0 + math.exp(-(z + 0.48) * 9.0)))
    temporal_wave = math.sin((wy + 0.20 * wz) * 9.5 + 0.70 * math.cos(wz * 6.0))

    # 5. Multi-scale Labyrinthine Gyral Convolutions across all lobes
    labyrinth_a = math.sin(wx * 10.5 + wy * 9.8 + 1.35 * math.cos(wz * 9.5 - wy * 8.2))
    labyrinth_b = math.cos(wz * 10.8 - wy * 9.4 + 1.25 * math.sin(wx * 10.2 + wz * 8.8))
    labyrinth_c = math.sin((wx - wz) * 9.6 + wy * 8.8 + 1.15 * math.sin(wx * 7.5 + wy * 8.0))

    lobe_guided = (
        central_weight * rolandic_wave
        + frontal_weight * frontal_wave * 0.55
        + temporal_weight * temporal_wave * 0.65
    )
    w = 0.52 * labyrinth_a + 0.38 * lobe_guided + 0.30 * labyrinth_c

    primary_sulcus = math.exp(-((w / 0.25) ** 2))
    branch_sulcus = math.exp(-((labyrinth_b / 0.22) ** 2)) * (1.0 - 0.5 * primary_sulcus)
    sulcus_depth = min(1.0, max(primary_sulcus, branch_sulcus * 0.82, sylvian_fissure))

    crown = (1.0 - sulcus_depth) ** 0.65
    disp = 0.032 * crown - 0.036 * sulcus_depth
    return disp, sulcus_depth


def _encephalic_hemisphere_point(theta: float, phi: float, side: float, inset: float = 0.0) -> Vec3:
    """Map spherical coordinates (theta in [0, pi] sup->inf, phi in [0, 2*pi)) to a realistic human cerebral hemisphere."""
    st = math.sin(theta)
    ct = math.cos(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)

    u_lat = st * cp
    u_ap = st * sp
    u_dv = ct

    z = u_ap * 0.99

    if u_dv >= 0.0:
        ap_arch = 1.0 - 0.14 * ((u_ap + 0.06) ** 2)
        y = 0.08 + 0.78 * (u_dv ** 0.84) * ap_arch
    else:
        depth = (-u_dv) ** 0.82
        lat_factor = max(0.0, u_lat) ** 0.55
        temporal_lobe_drop = (
            0.36
            * (1.0 / (1.0 + math.exp(-(z + 0.46) * 11.0)))
            * (1.0 / (1.0 + math.exp((z - 0.36) * 14.0)))
            * (0.35 + 0.65 * lat_factor)
        )
        base_floor = 0.22 + temporal_lobe_drop - 0.06 * max(0.0, -z - 0.40)
        y = 0.08 - depth * base_floor

    if u_lat >= 0.0:
        ap_taper = (1.0 - 0.26 * (u_ap ** 2)) ** 0.52
        temporal_bulge = (
            0.15
            * math.exp(-((z + 0.04) / 0.42) ** 2)
            * math.exp(-((y + 0.22) / 0.24) ** 2)
        )
        sylvian_y = -0.08 - 0.18 * z
        sylvian_cleft = (
            -0.11
            * math.exp(-((y - sylvian_y) / 0.055) ** 2)
            * (1.0 / (1.0 + math.exp(-(z + 0.32) * 12.0)))
            * (1.0 / (1.0 + math.exp((z - 0.42) * 12.0)))
            * (u_lat ** 0.65)
        )
        lat_width = 0.68 * (u_lat ** 0.74) * ap_taper + temporal_bulge + sylvian_cleft
        x_mag = 0.018 + max(0.02, lat_width - inset)
    else:
        medial_arch = 0.022 * ((-u_lat) ** 0.75)
        callosal_recess = 0.052 * math.exp(-((y - 0.04) / 0.24) ** 2) * math.exp(-((z + 0.02) / 0.42) ** 2)
        x_mag = 0.016 + medial_arch + callosal_recess + inset * 0.35

    if inset == 0.0:
        disp, _ = cortical_sulcus_field(x_mag * side, y, z, side)
        if u_lat >= 0.0:
            x_mag = max(0.018, x_mag + disp)
            y += disp * 0.62 * u_dv
            z += disp * 0.52 * u_ap
        else:
            y += disp * 0.28 * u_dv
            z += disp * 0.28 * u_ap

    return Vec3(side * x_mag, y, z)


def _cortical_seeds(side: float) -> list[tuple[I, float, float, float, tuple[str, str], Vec3]]:
    is_left = side > 0
    FRONTAL = ("#289eb4", "#e0a596")
    PARIETAL = ("#3292b8", "#dd9f90")
    TEMPORAL = ("#26a69a", "#dfa192")
    OCCIPITAL = ("#3f88b8", "#d99a8b")
    ELOQUENT_MOTOR = ("#00bcd4", "#e3998a")
    ELOQUENT_SPEECH = ("#00e5ff", "#e59586")
    ELOQUENT_SENS = ("#26c6da", "#de9c8d")
    ELOQUENT_VIS = ("#29b6f6", "#da9688")

    return [
        (I.SUPERIOR_FRONTAL_GYRUS_L if is_left else I.SUPERIOR_FRONTAL_GYRUS_R,
         0.42, 0.72, 0.52, FRONTAL, Vec3(side * 0.45, 0.88, 0.62)),
        (I.MIDDLE_FRONTAL_GYRUS_L if is_left else I.MIDDLE_FRONTAL_GYRUS_R,
         0.78, 0.46, 0.54, FRONTAL, Vec3(side * 0.78, 0.54, 0.62)),
        (I.INFERIOR_FRONTAL_PARS_ORBITALIS_L if is_left else I.INFERIOR_FRONTAL_PARS_ORBITALIS_R,
         0.72, -0.02, 0.68, FRONTAL, Vec3(side * 0.74, -0.04, 0.78)),
        (I.ORBITOFRONTAL_CORTEX_L if is_left else I.ORBITOFRONTAL_CORTEX_R,
         0.45, -0.18, 0.58, FRONTAL, Vec3(side * 0.48, -0.36, 0.72)),
        (I.FRONTAL_POLE_L if is_left else I.FRONTAL_POLE_R,
         0.48, 0.14, 0.90, FRONTAL, Vec3(side * 0.48, 0.16, 1.06)),
        (I.PARACENTRAL_LOBULE_L if is_left else I.PARACENTRAL_LOBULE_R,
         -0.55, 0.66, -0.08, FRONTAL, Vec3(side * 0.25, 0.90, -0.08)),
        (I.PRIMARY_MOTOR_CORTEX_M1_L if is_left else I.PRIMARY_MOTOR_CORTEX_M1_R,
         0.76, 0.50, -0.02, ELOQUENT_MOTOR, Vec3(side * 0.82, 0.64, -0.02)),
        (I.PREMOTOR_CORTEX_PMA_L if is_left else I.PREMOTOR_CORTEX_PMA_R,
         0.78, 0.48, 0.20, ELOQUENT_MOTOR, Vec3(side * 0.82, 0.60, 0.24)),
        (I.SUPPLEMENTARY_MOTOR_AREA_SMA_L if is_left else I.SUPPLEMENTARY_MOTOR_AREA_SMA_R,
         -0.45, 0.70, 0.24, ELOQUENT_MOTOR, Vec3(side * 0.28, 0.92, 0.28)),
        (I.FRONTAL_EYE_FIELD_FEF_L if is_left else I.FRONTAL_EYE_FIELD_FEF_R,
         0.68, 0.58, 0.34, ELOQUENT_MOTOR, Vec3(side * 0.72, 0.70, 0.38)),
        *([
            (I.BROCA_PARS_OPERCULARIS_L, 0.86, 0.14, 0.26, ELOQUENT_SPEECH, Vec3(0.92, 0.18, 0.30)),
            (I.BROCA_PARS_TRIANGULARIS_L, 0.84, 0.16, 0.48, ELOQUENT_SPEECH, Vec3(0.90, 0.20, 0.54)),
        ] if is_left else [
            (I.RIGHT_HOMOLOGUE_BROCA_R, 0.85, 0.15, 0.36, ELOQUENT_SPEECH, Vec3(-0.92, 0.18, 0.40)),
        ]),
        (I.PRIMARY_SOMATOSENSORY_S1_L if is_left else I.PRIMARY_SOMATOSENSORY_S1_R,
         0.76, 0.48, -0.18, ELOQUENT_SENS, Vec3(side * 0.82, 0.62, -0.20)),
        (I.SUPERIOR_PARIETAL_LOBULE_L if is_left else I.SUPERIOR_PARIETAL_LOBULE_R,
         0.52, 0.68, -0.46, PARIETAL, Vec3(side * 0.56, 0.84, -0.52)),
        (I.SUPRAMARGINAL_GYRUS_L if is_left else I.SUPRAMARGINAL_GYRUS_R,
         0.86, 0.24, -0.34, PARIETAL, Vec3(side * 0.92, 0.32, -0.38)),
        (I.ANGULAR_GYRUS_L if is_left else I.ANGULAR_GYRUS_R,
         0.80, 0.20, -0.56, PARIETAL, Vec3(side * 0.86, 0.26, -0.62)),
        (I.PRECUNEUS_L if is_left else I.PRECUNEUS_R,
         -0.55, 0.50, -0.48, PARIETAL, Vec3(side * 0.26, 0.74, -0.56)),
        *([
            (I.WERNICKE_AREA_L, 0.90, -0.06, -0.34, ELOQUENT_SPEECH, Vec3(0.96, -0.06, -0.38)),
            (I.GESCHWIND_TERRITORY_L, 0.88, 0.10, -0.44, ELOQUENT_SPEECH, Vec3(0.94, 0.12, -0.48)),
            (I.PRIMARY_AUDITORY_HESCHL_BI, 0.82, -0.02, -0.12, ELOQUENT_SENS, Vec3(0.88, -0.02, -0.12)),
        ] if is_left else []),
        (I.SUPERIOR_TEMPORAL_GYRUS_L if is_left else I.SUPERIOR_TEMPORAL_GYRUS_R,
         0.88, -0.16, 0.08, TEMPORAL, Vec3(side * 0.94, -0.16, 0.10)),
        (I.MIDDLE_TEMPORAL_GYRUS_L if is_left else I.MIDDLE_TEMPORAL_GYRUS_R,
         0.88, -0.28, -0.06, TEMPORAL, Vec3(side * 0.94, -0.30, -0.06)),
        (I.INFERIOR_TEMPORAL_GYRUS_L if is_left else I.INFERIOR_TEMPORAL_GYRUS_R,
         0.76, -0.40, -0.10, TEMPORAL, Vec3(side * 0.82, -0.46, -0.12)),
        (I.FUSIFORM_GYRUS_L if is_left else I.FUSIFORM_GYRUS_R,
         0.38, -0.38, -0.26, TEMPORAL, Vec3(side * 0.44, -0.52, -0.30)),
        (I.LATERAL_OCCIPITAL_CORTEX_L if is_left else I.LATERAL_OCCIPITAL_CORTEX_R,
         0.62, 0.02, -0.82, OCCIPITAL, Vec3(side * 0.68, 0.02, -0.96)),
        (I.CUNEUS_L if is_left else I.CUNEUS_R,
         -0.50, 0.24, -0.82, OCCIPITAL, Vec3(side * 0.26, 0.32, -0.98)),
        (I.LINGUAL_GYRUS_L if is_left else I.LINGUAL_GYRUS_R,
         -0.42, -0.18, -0.74, OCCIPITAL, Vec3(side * 0.26, -0.30, -0.92)),
        (I.PRIMARY_VISUAL_CALCARINE_V1_L if is_left else I.PRIMARY_VISUAL_CALCARINE_V1_R,
         -0.55, 0.02, -0.88, ELOQUENT_VIS, Vec3(side * 0.24, 0.02, -1.06)),
    ]


def build_seamless_cortical_hemisphere(
    side: float,
    names: dict[I, str],
    rows: int = 84,
    cols: int = 112,
) -> list[AnatomicalMesh]:
    """Generate a high-resolution, contiguous, gap-free parcellated cerebral hemisphere."""
    seeds = _cortical_seeds(side)
    outer_grid: list[list[Vec3]] = []
    meta_grid: list[list[tuple[float, float, float]]] = []

    for r in range(rows + 1):
        theta = math.pi * (0.012 + 0.976 * r / rows)
        st = math.sin(theta)
        row_out: list[Vec3] = []
        row_meta: list[tuple[float, float, float]] = []
        for c in range(cols):
            phi = math.tau * c / cols
            p_out = _encephalic_hemisphere_point(theta, phi, side, inset=0.0)
            u_lat = st * math.cos(phi)
            row_out.append(p_out)
            row_meta.append((u_lat, p_out.y, p_out.z))
        outer_grid.append(row_out)
        meta_grid.append(row_meta)

    cell_owner: list[list[int]] = []
    for r in range(rows):
        row_own: list[int] = []
        for c in range(cols):
            nc = (c + 1) % cols
            u_lat = 0.25 * (
                meta_grid[r][c][0] + meta_grid[r][nc][0] + meta_grid[r + 1][c][0] + meta_grid[r + 1][nc][0]
            )
            cy = 0.25 * (
                meta_grid[r][c][1] + meta_grid[r][nc][1] + meta_grid[r + 1][c][1] + meta_grid[r + 1][nc][1]
            )
            cz = 0.25 * (
                meta_grid[r][c][2] + meta_grid[r][nc][2] + meta_grid[r + 1][c][2] + meta_grid[r + 1][nc][2]
            )
            best_idx = 0
            best_dist = 1e9
            for idx, (_, su, sy, sz, _, _) in enumerate(seeds):
                side_penalty = 0.0 if (u_lat >= -0.05) == (su >= 0.0) else 1.25
                d = (u_lat - su) ** 2 * 0.8 + (cy - sy) ** 2 * 1.4 + (cz - sz) ** 2 * 1.2 + side_penalty
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            row_own.append(best_idx)
        cell_owner.append(row_own)

    meshes: list[AnatomicalMesh] = []
    for seed_idx, (part_id, su, sy, sz, palette, exp_vec) in enumerate(seeds):
        cells = [(r, c) for r in range(rows) for c in range(cols) if cell_owner[r][c] == seed_idx]
        if not cells:
            continue
        vert_map: dict[tuple[int, int], int] = {}
        verts: list[Vec3] = []
        faces: list[tuple[int, int, int]] = []

        def get_v(r_i: int, c_i: int) -> int:
            key = (r_i, c_i % cols)
            if key not in vert_map:
                vert_map[key] = len(verts)
                verts.append(outer_grid[key[0]][key[1]])
            return vert_map[key]

        for r, c in cells:
            nc = (c + 1) % cols
            o00, o01, o10, o11 = get_v(r, c), get_v(r, nc), get_v(r + 1, c), get_v(r + 1, nc)
            if side > 0:
                faces.append((o00, o10, o01))
                faces.append((o01, o10, o11))
            else:
                faces.append((o00, o01, o10))
                faces.append((o01, o11, o10))

        center = Vec3(
            sum(v.x for v in verts) / len(verts),
            sum(v.y for v in verts) / len(verts),
            sum(v.z for v in verts) / len(verts),
        )

        is_lateral_cover = su > 0.25
        if is_lateral_cover:
            meshes.append(
                AnatomicalMesh(
                    part_id=part_id,
                    vertices=tuple(verts),
                    faces=tuple(faces),
                    analysis_color=palette[0],
                    realistic_color=palette[1],
                    explosion_vector=exp_vec,
                    tissue=True,
                    default_visible=True,
                    view="exterior",
                    opacity=1.0,
                    element_key=part_id.value,
                    element_name=names[part_id],
                    deformation_center=center,
                )
            )
            meshes.append(
                AnatomicalMesh(
                    part_id=part_id,
                    vertices=tuple(verts),
                    faces=tuple(faces),
                    analysis_color=palette[0],
                    realistic_color=palette[1],
                    explosion_vector=exp_vec,
                    tissue=True,
                    default_visible=True,
                    view="interior",
                    opacity=0.14 if side > 0 else 0.08,
                    element_key=part_id.value,
                    element_name=names[part_id],
                    deformation_center=center,
                )
            )
        else:
            meshes.append(
                AnatomicalMesh(
                    part_id=part_id,
                    vertices=tuple(verts),
                    faces=tuple(faces),
                    analysis_color=palette[0],
                    realistic_color=palette[1],
                    explosion_vector=exp_vec,
                    tissue=True,
                    default_visible=True,
                    view="all",
                    opacity=1.0,
                    element_key=part_id.value,
                    element_name=names[part_id],
                    deformation_center=center,
                )
            )

    return meshes


def build_foliated_cerebellum_hemisphere(
    part: I,
    side: float,
    colors: tuple[str, str],
    explosion: Vec3,
    element_name: str,
) -> AnatomicalMesh:
    """Sculpt a realistic bilobed cerebellar hemisphere tucked snugly under the occipital lobe and behind the pons."""
    rows, cols = 52, 52
    vertices: list[Vec3] = []
    center = Vec3(side * 0.26, -0.34, -0.44)
    for r in range(rows + 1):
        lat = math.pi * r / rows
        sl, cl = math.sin(lat), math.cos(lat)
        for c in range(cols):
            lon = math.tau * c / cols
            cp, sp = math.cos(lon), math.sin(lon)
            rx, ry, rz = 0.29, 0.21, 0.27
            pontine_notch = -0.05 * max(0.0, sp) * math.exp(-((cl) / 0.55) ** 2)
            medial_hug = -0.04 * max(0.0, -side * cp)
            folia_wave = math.sin(lat * 20.0)
            folia_relief = 0.012 * (abs(folia_wave) ** 0.60) - 0.010 * math.exp(-((folia_wave / 0.24) ** 2))
            horiz_fissure = -0.016 * math.exp(-((lat - math.pi * 0.52) / 0.08) ** 2)

            rad_scale = 1.0 + folia_relief + horiz_fissure
            vx = center.x + (rx * rad_scale + medial_hug) * sl * cp
            vy = center.y + ry * rad_scale * cl
            vz = center.z + (rz * rad_scale + pontine_notch) * sl * sp
            vertices.append(Vec3(vx, vy, vz))

    faces: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            nc = (c + 1) % cols
            a = r * cols + c
            b = r * cols + nc
            d = (r + 1) * cols + c
            e = (r + 1) * cols + nc
            faces.append((a, d, b))
            faces.append((b, d, e))

    return AnatomicalMesh(
        part_id=part,
        vertices=tuple(vertices),
        faces=tuple(faces),
        analysis_color=colors[0],
        realistic_color=colors[1],
        explosion_vector=explosion,
        tissue=True,
        default_visible=True,
        view="all",
        element_key=part.value,
        element_name=element_name,
        deformation_center=center,
    )


def _shell(
    part: I,
    center: Vec3,
    radii: Vec3,
    thickness: float,
    colors: tuple[str, str],
    explosion: Vec3,
    section: bool = False,
    default_visible: bool = True,
    opacity: float = 1.0,
    element_key: str = "",
    element_name: str = "",
) -> AnatomicalMesh:
    """Watertight double-walled meningeal shell with closed cut-edge thickness."""
    rows = 16
    cols = 22 if section else 26
    max_phi = math.pi * 1.35 if section else math.tau

    outer_verts: list[Vec3] = []
    inner_verts: list[Vec3] = []
    for r in range(rows + 1):
        theta = math.pi * (0.05 + 0.90 * r / rows)
        st, ct = math.sin(theta), math.cos(theta)
        for c in range(cols + (1 if section else 0)):
            phi = max_phi * c / cols
            cp, sp = math.cos(phi), math.sin(phi)
            ox = center.x + radii.x * st * cp
            oy = center.y + radii.y * ct
            oz = center.z + radii.z * st * sp
            ix = center.x + max(0.04, radii.x - thickness) * st * cp
            iy = center.y + max(0.04, radii.y - thickness) * ct
            iz = center.z + max(0.04, radii.z - thickness) * st * sp
            outer_verts.append(Vec3(ox, oy, oz))
            inner_verts.append(Vec3(ix, iy, iz))

    vertices = outer_verts + inner_verts
    n_verts = len(outer_verts)
    stride = cols + 1 if section else cols
    faces: list[tuple[int, int, int]] = []

    for r in range(rows):
        for c in range(cols):
            nc = c + 1 if section else (c + 1) % cols
            a = r * stride + c
            b = r * stride + nc
            d = (r + 1) * stride + c
            e = (r + 1) * stride + nc
            faces.append((a, d, b))
            faces.append((b, d, e))
            ia, ib, id_, ie = n_verts + a, n_verts + b, n_verts + d, n_verts + e
            faces.append((ia, ib, id_))
            faces.append((ib, ie, id_))

    for r_rim, flip in ((0, False), (rows, True)):
        for c in range(cols):
            nc = c + 1 if section else (c + 1) % cols
            o1 = r_rim * stride + c
            o2 = r_rim * stride + nc
            i1 = n_verts + o1
            i2 = n_verts + o2
            if not flip:
                faces.append((o1, o2, i1))
                faces.append((o2, i2, i1))
            else:
                faces.append((o1, i1, o2))
                faces.append((o2, i1, i2))

    if section:
        for c_cut, flip in ((0, True), (cols, False)):
            for r in range(rows):
                o1 = r * stride + c_cut
                o2 = (r + 1) * stride + c_cut
                i1 = n_verts + o1
                i2 = n_verts + o2
                if not flip:
                    faces.append((o1, o2, i1))
                    faces.append((o2, i2, i1))
                else:
                    faces.append((o1, i1, o2))
                    faces.append((o2, i1, i2))

    return AnatomicalMesh(
        part_id=part,
        vertices=tuple(vertices),
        faces=tuple(faces),
        analysis_color=colors[0],
        realistic_color=colors[1],
        explosion_vector=explosion,
        tissue=True,
        default_visible=default_visible,
        view="interior" if section else "exterior",
        opacity=opacity,
        element_key=element_key or part.value,
        element_name=element_name,
        deformation_center=center,
    )

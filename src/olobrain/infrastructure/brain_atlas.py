"""Parametric 3D neuroanatomical atlas v2.0 of 139 distinct, gap-free brain structures.

Features:
- Continuous, watertight Voronoi-parcellated Left & Right Cerebral Hemispheres (zero gaps between gyri).
- High-resolution foliated Cerebellum tucked beneath the occipital lobes and behind the Pons.
- Contiguous sculpted Brainstem (Midbrain tectum/peduncles, bulging Pons Varolii, Medulla pyramids).
- Multi-fascicle white-matter connectome tracts (Corona radiata / Internal capsule fan, Arcuate bundle).
- Delicate microsurgical Circle of Willis and branching cortical pial vessels.
"""

from __future__ import annotations

import math
from dataclasses import replace

from olobrain.domain.model import BrainPartId as I, build_anatomy_catalog
from olobrain.infrastructure.geometry import AnatomicalMesh, Vec3, _ellipsoid, _tube
from olobrain.infrastructure.sculpted_geometry import (
    _shell,
    build_foliated_cerebellum_hemisphere,
    build_seamless_cortical_hemisphere,
)

NAMES: dict[I, str] = {part.identifier: part.display_name for part in build_anatomy_catalog()}

# Lifelike biological palettes (analysis_color, realistic_color)
MENINGES_COL = ("#56a8b8", "#dfd0c2")
VENTRICLE_COL = ("#2ccce4", "#6ccad8")
CHOROID_COL = ("#e06878", "#c95862")
INSULA_CORTEX = ("#42a5f5", "#d4887c")

WHITE_MATTER = ("#80deea", "#f2e7da")
ASSOCIATION_TRACT = ("#4dd0e1", "#ece0d0")
PROJECTION_TRACT = ("#00acc1", "#e9dac6")

STRIATUM_COL = ("#5c9ce6", "#c87f75")
PALLIDUM_COL = ("#64b5f6", "#d1a496")
THALAMUS_COL = ("#4fc3f7", "#cc887c")
ENDOCRINE_COL = ("#7986cb", "#d67c74")

LIMBIC_COL = ("#26a69a", "#d68e7b")
HIPPO_COL = ("#00bfa5", "#db957f")

BRAINSTEM_COL = ("#4db6ac", "#ead8ca")
CEREBELLUM_COL = ("#2698a6", "#c97e72")
CRANIAL_NERVE_COL = ("#ffd54f", "#f0d88c")

ARTERIAL_COL = ("#ff5252", "#d92b27")
VENOUS_COL = ("#448aff", "#3868b4")

# Delicate, microsurgically accurate cerebrovascular trunks (consumed by circulation_geometry.py)
VESSELS = (
    (I.INTERNAL_CAROTID_ARTERY_L, "internal_carotid_artery_l", NAMES[I.INTERNAL_CAROTID_ARTERY_L], 0.017, ARTERIAL_COL,
     Vec3(0.42, -0.52, 0.42),
     ((0.13, -0.58, 0.08), (0.14, -0.42, 0.14), (0.13, -0.26, 0.18), (0.15, -0.15, 0.16))),
    (I.INTERNAL_CAROTID_ARTERY_R, "internal_carotid_artery_r", NAMES[I.INTERNAL_CAROTID_ARTERY_R], 0.017, ARTERIAL_COL,
     Vec3(-0.42, -0.52, 0.42),
     ((-0.13, -0.58, 0.08), (-0.14, -0.42, 0.14), (-0.13, -0.26, 0.18), (-0.15, -0.15, 0.16))),
    (I.ANTERIOR_CEREBRAL_ARTERY_ACA_L, "anterior_cerebral_artery_aca_l", NAMES[I.ANTERIOR_CEREBRAL_ARTERY_ACA_L], 0.013, ARTERIAL_COL,
     Vec3(0.24, 0.52, 0.68),
     ((0.15, -0.15, 0.16), (0.03, -0.12, 0.28), (0.025, 0.12, 0.44), (0.025, 0.34, 0.28), (0.025, 0.38, -0.18))),
    (I.ANTERIOR_CEREBRAL_ARTERY_ACA_R, "anterior_cerebral_artery_aca_r", NAMES[I.ANTERIOR_CEREBRAL_ARTERY_ACA_R], 0.013, ARTERIAL_COL,
     Vec3(-0.24, 0.52, 0.68),
     ((-0.15, -0.15, 0.16), (-0.03, -0.12, 0.28), (-0.025, 0.12, 0.44), (-0.025, 0.34, 0.28), (-0.025, 0.38, -0.18))),
    (I.ANTERIOR_COMMUNICATING_ARTERY_ACOM, "anterior_communicating_artery_acom", NAMES[I.ANTERIOR_COMMUNICATING_ARTERY_ACOM], 0.011, ARTERIAL_COL,
     Vec3(0.0, -0.20, 0.76),
     ((-0.035, -0.12, 0.28), (0.0, -0.115, 0.29), (0.035, -0.12, 0.28))),
    (I.MIDDLE_CEREBRAL_ARTERY_MCA_L, "middle_cerebral_artery_mca_l", NAMES[I.MIDDLE_CEREBRAL_ARTERY_MCA_L], 0.015, ARTERIAL_COL,
     Vec3(0.88, 0.08, 0.28),
     ((0.15, -0.15, 0.16), (0.30, -0.12, 0.14), (0.44, -0.02, 0.08), (0.58, 0.14, -0.06), (0.56, 0.34, -0.26))),
    (I.MIDDLE_CEREBRAL_ARTERY_MCA_R, "middle_cerebral_artery_mca_r", NAMES[I.MIDDLE_CEREBRAL_ARTERY_MCA_R], 0.015, ARTERIAL_COL,
     Vec3(-0.88, 0.08, 0.28),
     ((-0.15, -0.15, 0.16), (-0.30, -0.12, 0.14), (-0.44, -0.02, 0.08), (-0.58, 0.14, -0.06), (-0.56, 0.34, -0.26))),
    (I.BASILAR_AND_VERTEBRAL_ARTERIES, "basilar_and_vertebral_arteries", NAMES[I.BASILAR_AND_VERTEBRAL_ARTERIES], 0.017, ARTERIAL_COL,
     Vec3(0.0, -0.76, 0.48),
     ((0.0, -0.88, -0.16), (0.0, -0.64, -0.06), (0.0, -0.40, 0.02), (0.0, -0.22, 0.04))),
    (I.POSTERIOR_CEREBRAL_ARTERY_PCA_L, "posterior_cerebral_artery_pca_l", NAMES[I.POSTERIOR_CEREBRAL_ARTERY_PCA_L], 0.014, ARTERIAL_COL,
     Vec3(0.48, -0.32, -0.74),
     ((0.0, -0.22, 0.04), (0.14, -0.21, -0.04), (0.20, -0.18, -0.24), (0.16, -0.08, -0.56), (0.10, 0.02, -0.84))),
    (I.POSTERIOR_CEREBRAL_ARTERY_PCA_R, "posterior_cerebral_artery_pca_r", NAMES[I.POSTERIOR_CEREBRAL_ARTERY_PCA_R], 0.014, ARTERIAL_COL,
     Vec3(-0.48, -0.32, -0.74),
     ((0.0, -0.22, 0.04), (-0.14, -0.21, -0.04), (-0.20, -0.18, -0.24), (-0.16, -0.08, -0.56), (-0.10, 0.02, -0.84))),
    (I.SUPERIOR_SAGITTAL_AND_STRAIGHT_SINUS, "superior_sagittal_and_straight_sinus", NAMES[I.SUPERIOR_SAGITTAL_AND_STRAIGHT_SINUS], 0.016, VENOUS_COL,
     Vec3(0.0, 1.08, -0.28),
     ((0.0, 0.24, 0.88), (0.0, 0.78, 0.50), (0.0, 0.90, 0.0), (0.0, 0.74, -0.58), (0.0, 0.14, -0.96), (0.0, -0.16, -0.88))),
    (I.VEIN_OF_GALEN_AND_TRANSVERSE_SINUSES, "vein_of_galen_and_transverse_sinuses", NAMES[I.VEIN_OF_GALEN_AND_TRANSVERSE_SINUSES], 0.015, VENOUS_COL,
     Vec3(0.0, -0.22, -1.04),
     ((-0.48, -0.22, -0.62), (-0.26, -0.18, -0.84), (0.0, -0.16, -0.88), (0.0, 0.04, -0.38), (0.0, -0.16, -0.88), (0.26, -0.18, -0.84), (0.48, -0.22, -0.62))),
)


def _owned(mesh: AnatomicalMesh, part: I, view: str | None = None) -> AnatomicalMesh:
    return replace(
        mesh,
        part_id=part,
        element_key=part.value,
        element_name=NAMES[part],
        view=view if view is not None else mesh.view,
    )


def build_brain_atlas() -> tuple[AnatomicalMesh, ...]:
    """Construct the complete 139-structure 3D brain atlas with seamless cortical hemispheres."""
    meshes: list[AnatomicalMesh] = []

    # =========================================================================
    # 1. MENINGES, CISTERNS & VENTRICULAR SYSTEM (14 structures)
    # =========================================================================
    for section in (False, True):
        dura = _shell(
            I.DURA_MATER_CONVEXITY,
            Vec3(0.0, 0.14, -0.02),
            Vec3(0.84, 0.82, 1.08),
            0.018,
            MENINGES_COL,
            Vec3(0.0, 1.18, 0.15),
            section=section,
            default_visible=True,
            opacity=0.42 if section else 0.20,
            element_key=I.DURA_MATER_CONVEXITY.value,
            element_name=NAMES[I.DURA_MATER_CONVEXITY],
        )
        meshes.append(dura)

    meshes.append(_owned(
        _ellipsoid(I.FALX_CEREBRI, Vec3(0.0, 0.38, 0.02), Vec3(0.010, 0.38, 0.86),
                   MENINGES_COL, Vec3(0.0, 1.06, 0.0), 12, 20, True, True, opacity=0.48),
        I.FALX_CEREBRI,
    ))
    meshes.append(_owned(
        _ellipsoid(I.TENTORIUM_CEREBELLI, Vec3(0.0, -0.24, -0.54), Vec3(0.56, 0.016, 0.38),
                   MENINGES_COL, Vec3(0.0, -0.38, -0.98), 12, 20, True, True, opacity=0.52),
        I.TENTORIUM_CEREBELLI,
    ))
    meshes.append(_owned(
        _ellipsoid(I.ARACHNOID_PIA_MATER, Vec3(0.0, 0.14, -0.02), Vec3(0.80, 0.78, 1.04),
                   ("#68c2ce", "#e8c4b8"), Vec3(0.0, 1.12, -0.22), 14, 22, True, True, opacity=0.18),
        I.ARACHNOID_PIA_MATER,
    ))

    # C-shaped Ventricular System nested inside the hemispheres
    meshes.append(_owned(
        _tube(I.LEFT_LATERAL_VENTRICLE,
              [Vec3(0.09, 0.14, 0.32), Vec3(0.11, 0.22, 0.06), Vec3(0.15, 0.14, -0.24),
               Vec3(0.14, 0.04, -0.42), Vec3(0.21, -0.12, -0.16), Vec3(0.23, -0.18, 0.08)],
              0.048, VENTRICLE_COL, Vec3(0.36, 0.40, 0.12), 16, False, True),
        I.LEFT_LATERAL_VENTRICLE,
    ))
    meshes.append(_owned(
        _tube(I.RIGHT_LATERAL_VENTRICLE,
              [Vec3(-0.09, 0.14, 0.32), Vec3(-0.11, 0.22, 0.06), Vec3(-0.15, 0.14, -0.24),
               Vec3(-0.14, 0.04, -0.42), Vec3(-0.21, -0.12, -0.16), Vec3(-0.23, -0.18, 0.08)],
              0.048, VENTRICLE_COL, Vec3(-0.36, 0.40, 0.12), 16, False, True),
        I.RIGHT_LATERAL_VENTRICLE,
    ))
    meshes.append(_owned(
        _tube(I.FORAMEN_OF_MONRO_LEFT, [Vec3(0.08, 0.12, 0.12), Vec3(0.015, 0.06, 0.08)],
              0.018, VENTRICLE_COL, Vec3(0.20, 0.28, 0.26), 10, False, True),
        I.FORAMEN_OF_MONRO_LEFT,
    ))
    meshes.append(_owned(
        _tube(I.FORAMEN_OF_MONRO_RIGHT, [Vec3(-0.08, 0.12, 0.12), Vec3(-0.015, 0.06, 0.08)],
              0.018, VENTRICLE_COL, Vec3(-0.20, 0.28, 0.26), 10, False, True),
        I.FORAMEN_OF_MONRO_RIGHT,
    ))
    meshes.append(_owned(
        _ellipsoid(I.THIRD_VENTRICLE, Vec3(0.0, 0.02, 0.0), Vec3(0.022, 0.10, 0.15),
                   VENTRICLE_COL, Vec3(0.0, 0.26, 0.34), 14, 18, False, True),
        I.THIRD_VENTRICLE,
    ))
    meshes.append(_owned(
        _tube(I.CEREBRAL_AQUEDUCT_SYLVIUS,
              [Vec3(0.0, -0.06, -0.10), Vec3(0.0, -0.22, -0.18), Vec3(0.0, -0.38, -0.26)],
              0.013, VENTRICLE_COL, Vec3(0.0, -0.16, -0.44), 10, False, True),
        I.CEREBRAL_AQUEDUCT_SYLVIUS,
    ))
    meshes.append(_owned(
        _ellipsoid(I.FOURTH_VENTRICLE, Vec3(0.0, -0.48, -0.30), Vec3(0.068, 0.11, 0.068),
                   VENTRICLE_COL, Vec3(0.0, -0.46, -0.54), 14, 18, False, True),
        I.FOURTH_VENTRICLE,
    ))
    meshes.append(_owned(
        _tube(I.CHOROID_PLEXUS_LATERAL,
              [Vec3(-0.11, 0.16, 0.06), Vec3(0.0, 0.10, 0.02), Vec3(0.11, 0.16, 0.06), Vec3(0.16, 0.10, -0.20)],
              0.016, CHOROID_COL, Vec3(0.0, 0.42, -0.12), 12, True, True, gyral_bump=0.14),
        I.CHOROID_PLEXUS_LATERAL,
    ))
    meshes.append(_owned(
        _tube(I.FORAMINA_LUSCHKA_MAGENDIE,
              [Vec3(-0.10, -0.54, -0.28), Vec3(0.0, -0.58, -0.34), Vec3(0.10, -0.54, -0.28)],
              0.014, VENTRICLE_COL, Vec3(0.0, -0.68, -0.58), 10, False, True),
        I.FORAMINA_LUSCHKA_MAGENDIE,
    ))
    meshes.append(_owned(
        _ellipsoid(I.SEPTUM_PELLUCIDUM, Vec3(0.0, 0.20, 0.18), Vec3(0.010, 0.09, 0.15),
                   MENINGES_COL, Vec3(0.0, 0.50, 0.40), 12, 16, True, True),
        I.SEPTUM_PELLUCIDUM,
    ))

    # =========================================================================
    # 2 & 3. SEAMLESS CORTICAL HEMISPHERES (52 surface parcels + 2 deep Insulae = 54)
    # =========================================================================
    meshes.extend(build_seamless_cortical_hemisphere(1.0, NAMES))
    meshes.extend(build_seamless_cortical_hemisphere(-1.0, NAMES))

    # Deep Insular Cortex (Island of Reil) tucked inside the Sylvian cleft bilaterally
    meshes.append(_owned(
        _ellipsoid(I.INSULAR_CORTEX_L, Vec3(0.36, 0.04, 0.08), Vec3(0.042, 0.14, 0.22),
                   INSULA_CORTEX, Vec3(0.66, 0.06, 0.14), 18, 24, True, True, gyral_freq=7.0),
        I.INSULAR_CORTEX_L,
    ))
    meshes.append(_owned(
        _ellipsoid(I.INSULAR_CORTEX_R, Vec3(-0.36, 0.04, 0.08), Vec3(0.042, 0.14, 0.22),
                   INSULA_CORTEX, Vec3(-0.66, 0.06, 0.14), 18, 24, True, True, gyral_freq=7.0),
        I.INSULAR_CORTEX_R,
    ))

    # =========================================================================
    # 4. WHITE MATTER TRACTS & BIOELECTRICAL CONNECTOME (16 structures)
    # =========================================================================
    # Broad, continuous C-shaped Corpus Callosum (Genu, Body, Splenium)
    callosal_specs = (
        (I.CORPUS_CALLOSUM_GENU, Vec3(0.0, 0.54, 0.64),
         [([Vec3(-0.22, 0.18, 0.48), Vec3(-0.10, 0.21, 0.38), Vec3(0.0, 0.22, 0.35),
            Vec3(0.10, 0.21, 0.38), Vec3(0.22, 0.18, 0.48)], 0.044),
          ([Vec3(-0.18, 0.14, 0.42), Vec3(0.0, 0.16, 0.31), Vec3(0.18, 0.14, 0.42)], 0.036)]),
        (I.CORPUS_CALLOSUM_BODY, Vec3(0.0, 0.64, 0.04),
         [([Vec3(-0.22, 0.26, z_off), Vec3(0.0, 0.30, z_off), Vec3(0.22, 0.26, z_off)], 0.042)
          for z_off in (0.22, 0.10, -0.02, -0.14)]),
        (I.CORPUS_CALLOSUM_SPLENIUM, Vec3(0.0, 0.52, -0.64),
         [([Vec3(-0.24, 0.16, -0.48), Vec3(-0.10, 0.23, -0.32), Vec3(0.0, 0.25, -0.28),
            Vec3(0.10, 0.23, -0.32), Vec3(0.24, 0.16, -0.48)], 0.048),
          ([Vec3(-0.20, 0.12, -0.40), Vec3(0.0, 0.20, -0.24), Vec3(0.20, 0.12, -0.40)], 0.038)]),
    )
    for part_id, exp, bundles in callosal_specs:
        for path, rad in bundles:
            meshes.append(_owned(_tube(part_id, path, rad, WHITE_MATTER, exp, 14, True, True), part_id))

    tract_specs = (
        (I.ANTERIOR_COMMISSURE, WHITE_MATTER, 0.018, Vec3(0.0, 0.06, 0.54),
         [Vec3(-0.24, -0.05, 0.10), Vec3(-0.10, 0.01, 0.15), Vec3(0.0, 0.02, 0.16),
          Vec3(0.10, 0.01, 0.15), Vec3(0.24, -0.05, 0.10)]),
        (I.ARCUATE_FASCICULUS_L, ASSOCIATION_TRACT, 0.032, Vec3(0.72, 0.32, 0.10),
         [Vec3(0.42, -0.06, -0.28), Vec3(0.44, 0.16, -0.30), Vec3(0.42, 0.28, -0.08),
          Vec3(0.40, 0.24, 0.16), Vec3(0.38, 0.12, 0.30)]),
        (I.ARCUATE_FASCICULUS_R, ASSOCIATION_TRACT, 0.030, Vec3(-0.72, 0.32, 0.10),
         [Vec3(-0.42, -0.06, -0.28), Vec3(-0.44, 0.16, -0.30), Vec3(-0.42, 0.28, -0.08),
          Vec3(-0.40, 0.24, 0.16), Vec3(-0.38, 0.12, 0.30)]),
        (I.SUPERIOR_LONGITUDINAL_FASCICULUS_L, ASSOCIATION_TRACT, 0.028, Vec3(0.68, 0.52, 0.04),
         [Vec3(0.36, 0.40, -0.38), Vec3(0.38, 0.44, -0.10), Vec3(0.36, 0.40, 0.18), Vec3(0.32, 0.32, 0.40)]),
        (I.SUPERIOR_LONGITUDINAL_FASCICULUS_R, ASSOCIATION_TRACT, 0.030, Vec3(-0.68, 0.52, 0.04),
         [Vec3(-0.36, 0.40, -0.38), Vec3(-0.38, 0.44, -0.10), Vec3(-0.36, 0.40, 0.18), Vec3(-0.32, 0.32, 0.40)]),
        (I.UNCINATE_FASCICULUS_L, ASSOCIATION_TRACT, 0.024, Vec3(0.60, -0.22, 0.54),
         [Vec3(0.32, -0.24, 0.28), Vec3(0.28, -0.12, 0.20), Vec3(0.24, -0.04, 0.32), Vec3(0.20, -0.08, 0.52)]),
        (I.UNCINATE_FASCICULUS_R, ASSOCIATION_TRACT, 0.024, Vec3(-0.60, -0.22, 0.54),
         [Vec3(-0.32, -0.24, 0.28), Vec3(-0.28, -0.12, 0.20), Vec3(-0.24, -0.04, 0.32), Vec3(-0.20, -0.08, 0.52)]),
        (I.INFERIOR_FRONTO_OCCIPITAL_FASCICULUS_L, ASSOCIATION_TRACT, 0.026, Vec3(0.64, -0.08, -0.16),
         [Vec3(0.24, -0.06, -0.66), Vec3(0.30, -0.08, -0.28), Vec3(0.28, -0.03, 0.06),
          Vec3(0.26, 0.05, 0.36), Vec3(0.22, 0.10, 0.56)]),
        (I.INFERIOR_FRONTO_OCCIPITAL_FASCICULUS_R, ASSOCIATION_TRACT, 0.026, Vec3(-0.64, -0.08, -0.16),
         [Vec3(-0.24, -0.06, -0.66), Vec3(-0.30, -0.08, -0.28), Vec3(-0.28, -0.03, 0.06),
          Vec3(-0.26, 0.05, 0.36), Vec3(-0.22, 0.10, 0.56)]),
        (I.INTERNAL_CAPSULE_CORTICOSPINAL_L, PROJECTION_TRACT, 0.032, Vec3(0.44, 0.12, -0.06),
         [Vec3(0.28, 0.58, -0.04), Vec3(0.22, 0.30, -0.04), Vec3(0.17, 0.04, -0.04),
          Vec3(0.10, -0.22, -0.08), Vec3(0.04, -0.52, -0.12), Vec3(0.02, -0.78, -0.15)]),
        (I.INTERNAL_CAPSULE_CORTICOSPINAL_R, PROJECTION_TRACT, 0.032, Vec3(-0.44, 0.12, -0.06),
         [Vec3(-0.28, 0.58, -0.04), Vec3(-0.22, 0.30, -0.04), Vec3(-0.17, 0.04, -0.04),
          Vec3(-0.10, -0.22, -0.08), Vec3(-0.04, -0.52, -0.12), Vec3(-0.02, -0.78, -0.15)]),
        (I.OPTIC_RADIATION_MEYER_LOOP_L, PROJECTION_TRACT, 0.024, Vec3(0.54, -0.16, -0.50),
         [Vec3(0.16, -0.05, -0.14), Vec3(0.28, -0.15, 0.04), Vec3(0.32, -0.14, -0.28),
          Vec3(0.22, -0.06, -0.56), Vec3(0.12, 0.0, -0.76)]),
        (I.OPTIC_RADIATION_MEYER_LOOP_R, PROJECTION_TRACT, 0.024, Vec3(-0.54, -0.16, -0.50),
         [Vec3(-0.16, -0.05, -0.14), Vec3(-0.28, -0.15, 0.04), Vec3(-0.32, -0.14, -0.28),
          Vec3(-0.22, -0.06, -0.56), Vec3(-0.12, 0.0, -0.76)]),
    )
    for part_id, col, rad, exp, path in tract_specs:
        meshes.append(_owned(_tube(part_id, path, rad, col, exp, 14, True, True), part_id))

    # =========================================================================
    # 5. BASAL GANGLIA, THALAMUS & DIENCEPHALON (18 structures)
    # =========================================================================
    meshes.append(_owned(
        _tube(I.CAUDATE_NUCLEUS_L,
              [Vec3(0.11, 0.12, 0.28), Vec3(0.13, 0.22, 0.10), Vec3(0.15, 0.16, -0.16),
               Vec3(0.20, 0.02, -0.24), Vec3(0.23, -0.14, -0.05)],
              0.040, STRIATUM_COL, Vec3(0.40, 0.30, 0.22), 16, True, True, taper=0.45),
        I.CAUDATE_NUCLEUS_L,
    ))
    meshes.append(_owned(
        _tube(I.CAUDATE_NUCLEUS_R,
              [Vec3(-0.11, 0.12, 0.28), Vec3(-0.13, 0.22, 0.10), Vec3(-0.15, 0.16, -0.16),
               Vec3(-0.20, 0.02, -0.24), Vec3(-0.23, -0.14, -0.05)],
              0.040, STRIATUM_COL, Vec3(-0.40, 0.30, 0.22), 16, True, True, taper=0.45),
        I.CAUDATE_NUCLEUS_R,
    ))

    nuclei_ellipsoids = (
        (I.PUTAMEN_L, Vec3(0.24, 0.03, 0.06), Vec3(0.058, 0.10, 0.15), STRIATUM_COL, Vec3(0.52, 0.08, 0.16)),
        (I.PUTAMEN_R, Vec3(-0.24, 0.03, 0.06), Vec3(0.058, 0.10, 0.15), STRIATUM_COL, Vec3(-0.52, 0.08, 0.16)),
        (I.GLOBUS_PALLIDUS_EXTERNUS_L, Vec3(0.19, 0.02, 0.04), Vec3(0.042, 0.078, 0.11), PALLIDUM_COL, Vec3(0.44, 0.04, 0.10)),
        (I.GLOBUS_PALLIDUS_EXTERNUS_R, Vec3(-0.19, 0.02, 0.04), Vec3(0.042, 0.078, 0.11), PALLIDUM_COL, Vec3(-0.44, 0.04, 0.10)),
        (I.GLOBUS_PALLIDUS_INTERNUS_L, Vec3(0.15, -0.01, 0.02), Vec3(0.034, 0.062, 0.088), PALLIDUM_COL, Vec3(0.38, -0.02, 0.06)),
        (I.GLOBUS_PALLIDUS_INTERNUS_R, Vec3(-0.15, -0.01, 0.02), Vec3(0.034, 0.062, 0.088), PALLIDUM_COL, Vec3(-0.38, -0.02, 0.06)),
        (I.SUBTHALAMIC_NUCLEUS_STN_L, Vec3(0.10, -0.08, -0.04), Vec3(0.030, 0.022, 0.042), PALLIDUM_COL, Vec3(0.30, -0.14, 0.02)),
        (I.SUBTHALAMIC_NUCLEUS_STN_R, Vec3(-0.10, -0.08, -0.04), Vec3(0.030, 0.022, 0.042), PALLIDUM_COL, Vec3(-0.30, -0.14, 0.02)),
        (I.CLAUSTRUM_L, Vec3(0.31, 0.03, 0.05), Vec3(0.014, 0.11, 0.17), STRIATUM_COL, Vec3(0.60, 0.06, 0.12)),
        (I.CLAUSTRUM_R, Vec3(-0.31, 0.03, 0.05), Vec3(0.014, 0.11, 0.17), STRIATUM_COL, Vec3(-0.60, 0.06, 0.12)),
        (I.THALAMUS_L, Vec3(0.09, 0.05, -0.08), Vec3(0.080, 0.10, 0.15), THALAMUS_COL, Vec3(0.28, 0.12, -0.16)),
        (I.THALAMUS_R, Vec3(-0.09, 0.05, -0.08), Vec3(0.080, 0.10, 0.15), THALAMUS_COL, Vec3(-0.28, 0.12, -0.16)),
        (I.GENICULATE_BODIES_LGN_MGN, Vec3(0.0, -0.03, -0.20), Vec3(0.16, 0.038, 0.048), THALAMUS_COL, Vec3(0.0, -0.06, -0.42)),
        (I.HYPOTHALAMUS_AND_PITUITARY_STALK, Vec3(0.0, -0.10, 0.08), Vec3(0.052, 0.068, 0.078), ENDOCRINE_COL, Vec3(0.0, -0.26, 0.32)),
        (I.EPITHALAMUS_PINEAL_GLAND, Vec3(0.0, 0.03, -0.24), Vec3(0.034, 0.034, 0.044), ENDOCRINE_COL, Vec3(0.0, 0.10, -0.50)),
        (I.PITUITARY_GLAND_HYPOPHYSIS, Vec3(0.0, -0.20, 0.14), Vec3(0.044, 0.038, 0.044), ENDOCRINE_COL, Vec3(0.0, -0.44, 0.42)),
    )
    for part_id, center, radii, col, exp in nuclei_ellipsoids:
        meshes.append(_owned(_ellipsoid(part_id, center, radii, col, exp, 16, 22, True, True), part_id))

    # =========================================================================
    # 6. LIMBIC SYSTEM & MEMORY CIRCUIT (12 structures)
    # =========================================================================
    limbic_tubes = (
        (I.HIPPOCAMPUS_L, HIPPO_COL, 0.042, Vec3(0.50, -0.26, -0.14),
         [Vec3(0.23, -0.20, 0.06), Vec3(0.25, -0.18, -0.08), Vec3(0.23, -0.12, -0.22), Vec3(0.18, -0.02, -0.28)]),
        (I.HIPPOCAMPUS_R, HIPPO_COL, 0.042, Vec3(-0.50, -0.26, -0.14),
         [Vec3(-0.23, -0.20, 0.06), Vec3(-0.25, -0.18, -0.08), Vec3(-0.23, -0.12, -0.22), Vec3(-0.18, -0.02, -0.28)]),
        (I.FORNIX_L, LIMBIC_COL, 0.020, Vec3(0.22, 0.32, -0.04),
         [Vec3(0.18, -0.02, -0.28), Vec3(0.12, 0.14, -0.20), Vec3(0.03, 0.18, -0.02),
          Vec3(0.025, 0.08, 0.12), Vec3(0.025, -0.10, 0.02)]),
        (I.FORNIX_R, LIMBIC_COL, 0.020, Vec3(-0.22, 0.32, -0.04),
         [Vec3(-0.18, -0.02, -0.28), Vec3(-0.12, 0.14, -0.20), Vec3(-0.03, 0.18, -0.02),
          Vec3(-0.025, 0.08, 0.12), Vec3(-0.025, -0.10, 0.02)]),
        (I.CINGULATE_GYRUS_L, LIMBIC_COL, 0.048, Vec3(0.20, 0.58, 0.06),
         [Vec3(0.055, 0.06, 0.42), Vec3(0.055, 0.32, 0.48), Vec3(0.055, 0.40, 0.10),
          Vec3(0.055, 0.36, -0.28), Vec3(0.065, 0.14, -0.42)]),
        (I.CINGULATE_GYRUS_R, LIMBIC_COL, 0.048, Vec3(-0.20, 0.58, 0.06),
         [Vec3(-0.055, 0.06, 0.42), Vec3(-0.055, 0.32, 0.48), Vec3(-0.055, 0.40, 0.10),
          Vec3(-0.055, 0.36, -0.28), Vec3(-0.065, 0.14, -0.42)]),
        (I.PARAHIPPOCAMPAL_ENTORHINAL_L, LIMBIC_COL, 0.044, Vec3(0.44, -0.38, -0.12),
         [Vec3(0.20, -0.26, 0.12), Vec3(0.22, -0.26, -0.08), Vec3(0.18, -0.22, -0.30)]),
        (I.PARAHIPPOCAMPAL_ENTORHINAL_R, LIMBIC_COL, 0.044, Vec3(-0.44, -0.38, -0.12),
         [Vec3(-0.20, -0.26, 0.12), Vec3(-0.22, -0.26, -0.08), Vec3(-0.18, -0.22, -0.30)]),
    )
    for part_id, col, rad, exp, path in limbic_tubes:
        meshes.append(_owned(_tube(part_id, path, rad, col, exp, 14, True, True, gyral_bump=0.06), part_id))

    limbic_ellipsoids = (
        (I.AMYGDALA_L, Vec3(0.22, -0.16, 0.14), Vec3(0.044, 0.044, 0.050), HIPPO_COL, Vec3(0.46, -0.24, 0.30)),
        (I.AMYGDALA_R, Vec3(-0.22, -0.16, 0.14), Vec3(0.044, 0.044, 0.050), HIPPO_COL, Vec3(-0.46, -0.24, 0.30)),
        (I.MAMMILLARY_BODIES, Vec3(0.0, -0.13, 0.02), Vec3(0.038, 0.028, 0.030), LIMBIC_COL, Vec3(0.0, -0.28, 0.16)),
        (I.NUCLEUS_ACCUMBENS_SEPTAL, Vec3(0.0, 0.02, 0.22), Vec3(0.072, 0.042, 0.046), LIMBIC_COL, Vec3(0.0, 0.06, 0.48)),
    )
    for part_id, center, radii, col, exp in limbic_ellipsoids:
        meshes.append(_owned(_ellipsoid(part_id, center, radii, col, exp, 14, 18, True, True), part_id))

    # =========================================================================
    # 7. CONTIGUOUS BRAINSTEM, FOLIATED CEREBELLUM & CRANIAL NERVES (13)
    # =========================================================================
    meshes.append(_owned(
        _ellipsoid(I.MIDBRAIN_TECTUM_TEGMENTUM, Vec3(0.0, -0.18, -0.14), Vec3(0.13, 0.11, 0.11),
                   BRAINSTEM_COL, Vec3(0.0, -0.24, -0.36), 16, 22, True, True),
        I.MIDBRAIN_TECTUM_TEGMENTUM,
    ))
    meshes.append(_owned(
        _ellipsoid(I.CEREBRAL_PEDUNCLES_SUBSTANTIA_NIGRA, Vec3(0.0, -0.18, -0.05), Vec3(0.15, 0.11, 0.085),
                   BRAINSTEM_COL, Vec3(0.0, -0.24, 0.16), 16, 22, True, True),
        I.CEREBRAL_PEDUNCLES_SUBSTANTIA_NIGRA,
    ))
    meshes.append(_owned(
        _ellipsoid(I.PONS_VAROLII, Vec3(0.0, -0.34, -0.08), Vec3(0.18, 0.15, 0.17),
                   BRAINSTEM_COL, Vec3(0.0, -0.52, 0.14), 22, 26, True, True, folia_freq=10.0),
        I.PONS_VAROLII,
    ))
    meshes.append(_owned(
        _tube(I.MEDULLA_OBLONGATA_PYRAMIDS,
              [Vec3(0.0, -0.45, -0.10), Vec3(0.0, -0.66, -0.14), Vec3(0.0, -0.88, -0.18)],
              0.105, BRAINSTEM_COL, Vec3(0.0, -0.98, -0.10), 18, True, True, taper=0.32),
        I.MEDULLA_OBLONGATA_PYRAMIDS,
    ))
    meshes.append(_owned(
        _tube(I.RETICULAR_ACTIVATING_SYSTEM_ARAS,
              [Vec3(0.0, -0.76, -0.17), Vec3(0.0, -0.42, -0.15), Vec3(0.0, -0.16, -0.11)],
              0.036, ("#ffb74d", "#e09c68"), Vec3(0.0, -0.48, -0.40), 12, True, True),
        I.RETICULAR_ACTIVATING_SYSTEM_ARAS,
        view="interior",
    ))

    # Realistic Foliated Cerebellum tucked under occipital lobes
    meshes.append(build_foliated_cerebellum_hemisphere(
        I.CEREBELLAR_HEMISPHERE_L, 1.0, CEREBELLUM_COL, Vec3(0.62, -0.62, -0.82), NAMES[I.CEREBELLAR_HEMISPHERE_L]
    ))
    meshes.append(build_foliated_cerebellum_hemisphere(
        I.CEREBELLAR_HEMISPHERE_R, -1.0, CEREBELLUM_COL, Vec3(-0.62, -0.62, -0.82), NAMES[I.CEREBELLAR_HEMISPHERE_R]
    ))
    meshes.append(_owned(
        _ellipsoid(I.CEREBELLAR_VERMIS_FLOCCULONODULAR, Vec3(0.0, -0.34, -0.42), Vec3(0.10, 0.20, 0.23),
                   CEREBELLUM_COL, Vec3(0.0, -0.64, -0.88), 28, 24, True, True, folia_freq=22.0),
        I.CEREBELLAR_VERMIS_FLOCCULONODULAR,
    ))
    meshes.append(_owned(
        _ellipsoid(I.DENTATE_AND_DEEP_CEREBELLAR_NUCLEI, Vec3(0.0, -0.34, -0.36), Vec3(0.18, 0.068, 0.075),
                   ("#81c784", "#d69e8e"), Vec3(0.0, -0.48, -0.68), 14, 20, True, True, gyral_freq=8.0),
        I.DENTATE_AND_DEEP_CEREBELLAR_NUCLEI,
    ))

    # Delicate Cranial Nerves I - XII emerging from true brainstem/cisternal exit zones
    cn_specs = (
        (I.CRANIAL_NERVES_I_II_SENSORY, Vec3(0.0, -0.30, 0.66),
         [Vec3(-0.11, -0.15, 0.56), Vec3(-0.06, -0.14, 0.24), Vec3(0.0, -0.12, 0.16),
          Vec3(0.06, -0.14, 0.24), Vec3(0.11, -0.15, 0.56)]),
        (I.CRANIAL_NERVES_III_IV_VI_OCULOMOTOR, Vec3(0.0, -0.42, 0.50),
         [Vec3(-0.11, -0.26, 0.26), Vec3(-0.05, -0.22, -0.02), Vec3(0.05, -0.22, -0.02), Vec3(0.11, -0.26, 0.26)]),
        (I.CRANIAL_NERVES_V_VII_VIII_CPA, Vec3(0.0, -0.58, 0.32),
         [Vec3(-0.24, -0.44, 0.04), Vec3(-0.14, -0.42, -0.10), Vec3(0.14, -0.42, -0.10), Vec3(0.24, -0.44, 0.04)]),
        (I.CRANIAL_NERVES_IX_X_XI_XII_LOWER, Vec3(0.0, -0.84, 0.24),
         [Vec3(-0.18, -0.70, -0.06), Vec3(-0.08, -0.68, -0.15), Vec3(0.08, -0.68, -0.15), Vec3(0.18, -0.70, -0.06)]),
    )
    for part_id, exp, path in cn_specs:
        meshes.append(_owned(_tube(part_id, path, 0.011, CRANIAL_NERVE_COL, exp, 10, False, True), part_id))

    # =========================================================================
    # 8. CEREBROVASCULAR CIRCULATION: CIRCLE OF WILLIS & VENOUS SINUSES (12)
    # =========================================================================
    for part_id, key, name, radius, col, exp, path in VESSELS:
        pts = [Vec3(*p) for p in path]
        # Venous sinuses are hidden by default in assembled exterior view unless selected or in interior/exploded view
        # so they never obstruct the natural cortical gyri!
        is_sinus = part_id in {
            I.SUPERIOR_SAGITTAL_AND_STRAIGHT_SINUS,
            I.VEIN_OF_GALEN_AND_TRANSVERSE_SINUSES,
        }
        meshes.append(
            _owned(
                _tube(part_id, pts, radius, col, exp, 14, False, not is_sinus, curve_steps=6),
                part_id,
            )
        )

    return tuple(meshes)


build_cardiac_atlas = build_brain_atlas

"""Bioelectrical axonal action potential routes and cerebral hemodynamic pathways.

Models three synchronized dynamic flow regimes:
- Gate 0: Continuous Cerebral Blood Flow (Circle of Willis & Dural Venous Sinuses)
- Gate 1: Afferent Sensory, Language Comprehension (Wernicke -> Arcuate -> Broca) & Limbic Memory Spikes
- Gate 2: Efferent Pyramidal Motor Execution (Corticospinal, Corticobulbar & Cerebellar Loops)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from olobrain.domain.model import BrainPartId as I, BrainSnapshot
from olobrain.infrastructure.brain_atlas import VESSELS
from olobrain.infrastructure.geometry import Vec3, _curve_points


@dataclass(frozen=True, slots=True)
class FlowRoute:
    key: str
    oxygenated: bool  # True = Excitatory Action Potential / Arterial; False = Inhibitory / Venous
    gate: int         # 0 = Continuous vascular, 1 = Afferent/Language/Memory, 2 = Efferent Motor/Speech
    parts: tuple[I, ...]
    points: np.ndarray
    kind: str = "neural"  # "neural" or "vascular"

    def active(self, state: BrainSnapshot) -> bool:
        """Gate 1 and Gate 2 fire in synchrony with the neurophysiological cycle."""
        return (
            self.gate == 0
            or (self.gate == 1 and state.cycle.sensory_afferent_gate_open)
            or (self.gate == 2 and state.cycle.motor_efferent_gate_open)
        )


def _route(key: str, oxygenated: bool, gate: int, parts: tuple[I, ...], path, kind: str = "neural") -> FlowRoute:
    samples = _curve_points([p if isinstance(p, Vec3) else Vec3(*p) for p in path], 10)
    positions = np.asarray([(p.x, p.y, p.z) for p in samples], dtype=np.float32)
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    distances = np.linspace(0.0, float(lengths[-1]), 100)
    points = np.column_stack([np.interp(distances, lengths, positions[:, axis]) for axis in range(3)])
    return FlowRoute(key, oxygenated, gate, parts, points.astype(np.float32), kind=kind)


def build_flow_routes() -> tuple[FlowRoute, ...]:
    """Build 24 bioelectrical axonal and cerebrovascular flow routes."""
    routes: list[FlowRoute] = []

    # 12 Cerebrovascular routes from VESSELS (Gate 0: Continuous hemodynamic perfusion)
    for part, key, _, _, _, _, path in VESSELS:
        is_arterial = part not in {
            I.SUPERIOR_SAGITTAL_AND_STRAIGHT_SINUS,
            I.VEIN_OF_GALEN_AND_TRANSVERSE_SINUSES,
        }
        routes.append(_route(key, is_arterial, 0, (part,), path, kind="vascular"))

    # 6 Gate-1 Bioelectrical Afferent, Language & Memory Routes
    routes.extend((
        _route(
            "dorsal_speech_stream_left",
            True,
            1,
            (I.WERNICKE_AREA_L, I.ARCUATE_FASCICULUS_L, I.BROCA_PARS_OPERCULARIS_L),
            [(0.52, -0.08, -0.34), (0.54, 0.18, -0.36), (0.52, 0.34, -0.12), (0.50, 0.28, 0.18), (0.48, 0.14, 0.36)],
            kind="neural",
        ),
        _route(
            "ventral_semantic_stream_left",
            True,
            1,
            (I.FUSIFORM_GYRUS_L, I.INFERIOR_FRONTO_OCCIPITAL_FASCICULUS_L, I.BROCA_PARS_TRIANGULARIS_L),
            [(0.28, -0.08, -0.78), (0.36, -0.10, -0.34), (0.34, -0.04, 0.08), (0.32, 0.06, 0.42), (0.44, 0.14, 0.46)],
            kind="neural",
        ),
        _route(
            "thalamocortical_sensory_left",
            True,
            1,
            (I.THALAMUS_L, I.PRIMARY_SOMATOSENSORY_S1_L),
            [(0.11, 0.06, -0.10), (0.22, 0.28, -0.14), (0.34, 0.56, -0.18), (0.44, 0.78, -0.18)],
            kind="neural",
        ),
        _route(
            "thalamocortical_sensory_right",
            True,
            1,
            (I.THALAMUS_R, I.PRIMARY_SOMATOSENSORY_S1_R),
            [(-0.11, 0.06, -0.10), (-0.22, 0.28, -0.14), (-0.34, 0.56, -0.18), (-0.44, 0.78, -0.18)],
            kind="neural",
        ),
        _route(
            "optic_radiation_left",
            True,
            1,
            (I.GENICULATE_BODIES_LGN_MGN, I.OPTIC_RADIATION_MEYER_LOOP_L, I.PRIMARY_VISUAL_CALCARINE_V1_L),
            [(0.18, -0.06, -0.20), (0.34, -0.18, 0.04), (0.38, -0.16, -0.32), (0.26, -0.08, -0.66), (0.12, 0.0, -0.88)],
            kind="neural",
        ),
        _route(
            "papez_memory_circuit_left",
            True,
            1,
            (I.HIPPOCAMPUS_L, I.FORNIX_L, I.MAMMILLARY_BODIES, I.THALAMUS_L, I.CINGULATE_GYRUS_L),
            [(0.28, -0.22, -0.08), (0.22, -0.02, -0.32), (0.04, 0.22, -0.02), (0.02, -0.14, 0.02), (0.08, 0.08, -0.04), (0.07, 0.44, 0.12)],
            kind="neural",
        ),
    ))

    # 6 Gate-2 Bioelectrical Efferent Motor, Speech Execution & Interhemispheric Routes
    routes.extend((
        _route(
            "corticospinal_pyramidal_left",
            True,
            2,
            (I.PRIMARY_MOTOR_CORTEX_M1_L, I.INTERNAL_CAPSULE_CORTICOSPINAL_L, I.MEDULLA_OBLONGATA_PYRAMIDS),
            [(0.34, 0.74, -0.04), (0.26, 0.36, -0.04), (0.20, 0.06, -0.04), (0.12, -0.32, -0.10), (0.04, -0.82, -0.18), (-0.04, -1.08, -0.24)],
            kind="neural",
        ),
        _route(
            "corticospinal_pyramidal_right",
            True,
            2,
            (I.PRIMARY_MOTOR_CORTEX_M1_R, I.INTERNAL_CAPSULE_CORTICOSPINAL_R, I.MEDULLA_OBLONGATA_PYRAMIDS),
            [(-0.34, 0.74, -0.04), (-0.26, 0.36, -0.04), (-0.20, 0.06, -0.04), (-0.12, -0.32, -0.10), (-0.04, -0.82, -0.18), (0.04, -1.08, -0.24)],
            kind="neural",
        ),
        _route(
            "corticobulbar_speech_execution",
            True,
            2,
            (I.BROCA_PARS_OPERCULARIS_L, I.PRIMARY_MOTOR_CORTEX_M1_L, I.CRANIAL_NERVES_IX_X_XI_XII_LOWER),
            [(0.48, 0.14, 0.34), (0.46, 0.24, -0.02), (0.20, 0.04, -0.02), (0.08, -0.52, -0.14), (0.12, -0.84, -0.16)],
            kind="neural",
        ),
        _route(
            "cerebellar_dentatothalamocortical_loop",
            False,
            2,
            (I.CEREBELLAR_HEMISPHERE_R, I.DENTATE_AND_DEEP_CEREBELLAR_NUCLEI, I.THALAMUS_L, I.PRIMARY_MOTOR_CORTEX_M1_L),
            [(-0.34, -0.58, -0.62), (-0.12, -0.54, -0.48), (0.0, -0.26, -0.18), (0.11, 0.06, -0.10), (0.34, 0.68, -0.04)],
            kind="neural",
        ),
        _route(
            "basal_ganglia_direct_motor_loop",
            False,
            2,
            (I.PUTAMEN_L, I.GLOBUS_PALLIDUS_INTERNUS_L, I.THALAMUS_L, I.SUPPLEMENTARY_MOTOR_AREA_SMA_L),
            [(0.28, 0.04, 0.08), (0.17, -0.01, 0.02), (0.11, 0.06, -0.08), (0.08, 0.68, 0.22)],
            kind="neural",
        ),
        _route(
            "transcallosal_motor_sync",
            True,
            2,
            (I.PRIMARY_MOTOR_CORTEX_M1_L, I.CORPUS_CALLOSUM_BODY, I.PRIMARY_MOTOR_CORTEX_M1_R),
            [(0.42, 0.64, -0.04), (0.18, 0.38, 0.0), (0.0, 0.38, 0.0), (-0.18, 0.38, 0.0), (-0.42, 0.64, -0.04)],
            kind="neural",
        ),
    ))

    return tuple(routes)


def circulation_visible(state: BrainSnapshot) -> bool:
    """Displaced pieces no longer form a connected bioelectrical or vascular network."""
    return state.flow_enabled and state.explosion < 0.035 and not state.fragment_offsets

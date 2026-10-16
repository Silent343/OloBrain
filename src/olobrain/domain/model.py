"""Pure domain model for neuroanatomy, electrophysiology, and the interactive brain aggregate.

Follows Domain-Driven Design (DDD), Clean Architecture, and SOLID principles:
- Entities and Value Objects are immutable dataclasses with zero framework dependencies.
- BrainExperience acts as the Aggregate Root enforcing all domain invariants.
- Grounded in Q1/Q2 neuroscientific and neurosurgical literature:
  * Glasser et al. (2016), Nature 536:171-178 (Human Connectome Project MMP1.0)
  * Rhoton AL Jr. (2002/2023), Neurosurgery / Cranial Anatomy and Surgical Approaches
  * Hickok & Poeppel (2007/2012), Nature Reviews Neuroscience (Dual-Stream Speech Model)
  * Buzsáki & Draguhn (2004), Science 304:1926-1929 (Neuronal Oscillations in Cortical Networks)
  * StatPearls / NCBI Neuroanatomy Monographs (2023-2025)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class BrainPartId(StrEnum):
    """139 distinct neuroanatomical structures for full 3D parcellation and exploded view."""

    # 1. Meninges, Cisterns & Ventricular System (14)
    DURA_MATER_CONVEXITY = "dura_mater_convexity"
    FALX_CEREBRI = "falx_cerebri"
    TENTORIUM_CEREBELLI = "tentorium_cerebelli"
    ARACHNOID_PIA_MATER = "arachnoid_pia_mater"
    LEFT_LATERAL_VENTRICLE = "left_lateral_ventricle"
    RIGHT_LATERAL_VENTRICLE = "right_lateral_ventricle"
    FORAMEN_OF_MONRO_LEFT = "foramen_of_monro_left"
    FORAMEN_OF_MONRO_RIGHT = "foramen_of_monro_right"
    THIRD_VENTRICLE = "third_ventricle"
    CEREBRAL_AQUEDUCT_SYLVIUS = "cerebral_aqueduct_sylvius"
    FOURTH_VENTRICLE = "fourth_ventricle"
    CHOROID_PLEXUS_LATERAL = "choroid_plexus_lateral"
    FORAMINA_LUSCHKA_MAGENDIE = "foramina_luschka_magendie"
    SEPTUM_PELLUCIDUM = "septum_pellucidum"

    # 2. Cerebral Cortex: Gyri & Lobes (36 = 18 Left + 18 Right)
    SUPERIOR_FRONTAL_GYRUS_L = "superior_frontal_gyrus_l"
    SUPERIOR_FRONTAL_GYRUS_R = "superior_frontal_gyrus_r"
    MIDDLE_FRONTAL_GYRUS_L = "middle_frontal_gyrus_l"
    MIDDLE_FRONTAL_GYRUS_R = "middle_frontal_gyrus_r"
    INFERIOR_FRONTAL_PARS_ORBITALIS_L = "inferior_frontal_pars_orbitalis_l"
    INFERIOR_FRONTAL_PARS_ORBITALIS_R = "inferior_frontal_pars_orbitalis_r"
    ORBITOFRONTAL_CORTEX_L = "orbitofrontal_cortex_l"
    ORBITOFRONTAL_CORTEX_R = "orbitofrontal_cortex_r"
    PARACENTRAL_LOBULE_L = "paracentral_lobule_l"
    PARACENTRAL_LOBULE_R = "paracentral_lobule_r"
    FRONTAL_POLE_L = "frontal_pole_l"
    FRONTAL_POLE_R = "frontal_pole_r"
    SUPERIOR_PARIETAL_LOBULE_L = "superior_parietal_lobule_l"
    SUPERIOR_PARIETAL_LOBULE_R = "superior_parietal_lobule_r"
    SUPRAMARGINAL_GYRUS_L = "supramarginal_gyrus_l"
    SUPRAMARGINAL_GYRUS_R = "supramarginal_gyrus_r"
    ANGULAR_GYRUS_L = "angular_gyrus_l"
    ANGULAR_GYRUS_R = "angular_gyrus_r"
    PRECUNEUS_L = "precuneus_l"
    PRECUNEUS_R = "precuneus_r"
    SUPERIOR_TEMPORAL_GYRUS_L = "superior_temporal_gyrus_l"
    SUPERIOR_TEMPORAL_GYRUS_R = "superior_temporal_gyrus_r"
    MIDDLE_TEMPORAL_GYRUS_L = "middle_temporal_gyrus_l"
    MIDDLE_TEMPORAL_GYRUS_R = "middle_temporal_gyrus_r"
    INFERIOR_TEMPORAL_GYRUS_L = "inferior_temporal_gyrus_l"
    INFERIOR_TEMPORAL_GYRUS_R = "inferior_temporal_gyrus_r"
    FUSIFORM_GYRUS_L = "fusiform_gyrus_l"
    FUSIFORM_GYRUS_R = "fusiform_gyrus_r"
    CUNEUS_L = "cuneus_l"
    CUNEUS_R = "cuneus_r"
    LINGUAL_GYRUS_L = "lingual_gyrus_l"
    LINGUAL_GYRUS_R = "lingual_gyrus_r"
    LATERAL_OCCIPITAL_CORTEX_L = "lateral_occipital_cortex_l"
    LATERAL_OCCIPITAL_CORTEX_R = "lateral_occipital_cortex_r"
    INSULAR_CORTEX_L = "insular_cortex_l"
    INSULAR_CORTEX_R = "insular_cortex_r"

    # 3. Eloquent Speech, Motor & Primary Sensory Areas (18)
    PRIMARY_MOTOR_CORTEX_M1_L = "primary_motor_cortex_m1_l"
    PRIMARY_MOTOR_CORTEX_M1_R = "primary_motor_cortex_m1_r"
    PREMOTOR_CORTEX_PMA_L = "premotor_cortex_pma_l"
    PREMOTOR_CORTEX_PMA_R = "premotor_cortex_pma_r"
    SUPPLEMENTARY_MOTOR_AREA_SMA_L = "supplementary_motor_area_sma_l"
    SUPPLEMENTARY_MOTOR_AREA_SMA_R = "supplementary_motor_area_sma_r"
    FRONTAL_EYE_FIELD_FEF_L = "frontal_eye_field_fef_l"
    FRONTAL_EYE_FIELD_FEF_R = "frontal_eye_field_fef_r"
    BROCA_PARS_OPERCULARIS_L = "broca_pars_opercularis_l"
    BROCA_PARS_TRIANGULARIS_L = "broca_pars_triangularis_l"
    RIGHT_HOMOLOGUE_BROCA_R = "right_homologue_broca_r"
    WERNICKE_AREA_L = "wernicke_area_l"
    GESCHWIND_TERRITORY_L = "geschwind_territory_l"
    PRIMARY_SOMATOSENSORY_S1_L = "primary_somatosensory_s1_l"
    PRIMARY_SOMATOSENSORY_S1_R = "primary_somatosensory_s1_r"
    PRIMARY_AUDITORY_HESCHL_BI = "primary_auditory_heschl_bi"
    PRIMARY_VISUAL_CALCARINE_V1_L = "primary_visual_calcarine_v1_l"
    PRIMARY_VISUAL_CALCARINE_V1_R = "primary_visual_calcarine_v1_r"

    # 4. White Matter Tracts & Bioelectrical Connectome (16)
    CORPUS_CALLOSUM_GENU = "corpus_callosum_genu"
    CORPUS_CALLOSUM_BODY = "corpus_callosum_body"
    CORPUS_CALLOSUM_SPLENIUM = "corpus_callosum_splenium"
    ANTERIOR_COMMISSURE = "anterior_commissure"
    ARCUATE_FASCICULUS_L = "arcuate_fasciculus_l"
    ARCUATE_FASCICULUS_R = "arcuate_fasciculus_r"
    SUPERIOR_LONGITUDINAL_FASCICULUS_L = "superior_longitudinal_fasciculus_l"
    SUPERIOR_LONGITUDINAL_FASCICULUS_R = "superior_longitudinal_fasciculus_r"
    UNCINATE_FASCICULUS_L = "uncinate_fasciculus_l"
    UNCINATE_FASCICULUS_R = "uncinate_fasciculus_r"
    INFERIOR_FRONTO_OCCIPITAL_FASCICULUS_L = "inferior_fronto_occipital_fasciculus_l"
    INFERIOR_FRONTO_OCCIPITAL_FASCICULUS_R = "inferior_fronto_occipital_fasciculus_r"
    INTERNAL_CAPSULE_CORTICOSPINAL_L = "internal_capsule_corticospinal_l"
    INTERNAL_CAPSULE_CORTICOSPINAL_R = "internal_capsule_corticospinal_r"
    OPTIC_RADIATION_MEYER_LOOP_L = "optic_radiation_meyer_loop_l"
    OPTIC_RADIATION_MEYER_LOOP_R = "optic_radiation_meyer_loop_r"

    # 5. Basal Ganglia, Thalamus & Diencephalon (18)
    CAUDATE_NUCLEUS_L = "caudate_nucleus_l"
    CAUDATE_NUCLEUS_R = "caudate_nucleus_r"
    PUTAMEN_L = "putamen_l"
    PUTAMEN_R = "putamen_r"
    GLOBUS_PALLIDUS_EXTERNUS_L = "globus_pallidus_externus_l"
    GLOBUS_PALLIDUS_EXTERNUS_R = "globus_pallidus_externus_r"
    GLOBUS_PALLIDUS_INTERNUS_L = "globus_pallidus_internus_l"
    GLOBUS_PALLIDUS_INTERNUS_R = "globus_pallidus_internus_r"
    SUBTHALAMIC_NUCLEUS_STN_L = "subthalamic_nucleus_stn_l"
    SUBTHALAMIC_NUCLEUS_STN_R = "subthalamic_nucleus_stn_r"
    CLAUSTRUM_L = "claustrum_l"
    CLAUSTRUM_R = "claustrum_r"
    THALAMUS_L = "thalamus_l"
    THALAMUS_R = "thalamus_r"
    GENICULATE_BODIES_LGN_MGN = "geniculate_bodies_lgn_mgn"
    HYPOTHALAMUS_AND_PITUITARY_STALK = "hypothalamus_and_pituitary_stalk"
    EPITHALAMUS_PINEAL_GLAND = "epithalamus_pineal_gland"
    PITUITARY_GLAND_HYPOPHYSIS = "pituitary_gland_hypophysis"

    # 6. Limbic System & Memory Circuit (12)
    HIPPOCAMPUS_L = "hippocampus_l"
    HIPPOCAMPUS_R = "hippocampus_r"
    AMYGDALA_L = "amygdala_l"
    AMYGDALA_R = "amygdala_r"
    FORNIX_L = "fornix_l"
    FORNIX_R = "fornix_r"
    CINGULATE_GYRUS_L = "cingulate_gyrus_l"
    CINGULATE_GYRUS_R = "cingulate_gyrus_r"
    PARAHIPPOCAMPAL_ENTORHINAL_L = "parahippocampal_entorhinal_l"
    PARAHIPPOCAMPAL_ENTORHINAL_R = "parahippocampal_entorhinal_r"
    MAMMILLARY_BODIES = "mammillary_bodies"
    NUCLEUS_ACCUMBENS_SEPTAL = "nucleus_accumbens_septal"

    # 7. Brainstem, Cerebellum & Cranial Nerves (13)
    MIDBRAIN_TECTUM_TEGMENTUM = "midbrain_tectum_tegmentum"
    CEREBRAL_PEDUNCLES_SUBSTANTIA_NIGRA = "cerebral_peduncles_substantia_nigra"
    PONS_VAROLII = "pons_varolii"
    MEDULLA_OBLONGATA_PYRAMIDS = "medulla_oblongata_pyramids"
    RETICULAR_ACTIVATING_SYSTEM_ARAS = "reticular_activating_system_aras"
    CEREBELLAR_HEMISPHERE_L = "cerebellar_hemisphere_l"
    CEREBELLAR_HEMISPHERE_R = "cerebellar_hemisphere_r"
    CEREBELLAR_VERMIS_FLOCCULONODULAR = "cerebellar_vermis_flocculonodular"
    DENTATE_AND_DEEP_CEREBELLAR_NUCLEI = "dentate_and_deep_cerebellar_nuclei"
    CRANIAL_NERVES_I_II_SENSORY = "cranial_nerves_i_ii_sensory"
    CRANIAL_NERVES_III_IV_VI_OCULOMOTOR = "cranial_nerves_iii_iv_vi_oculomotor"
    CRANIAL_NERVES_V_VII_VIII_CPA = "cranial_nerves_v_vii_viii_cpa"
    CRANIAL_NERVES_IX_X_XI_XII_LOWER = "cranial_nerves_ix_x_xi_xii_lower"

    # 8. Cerebrovascular Circulation: Circle of Willis & Venous Sinuses (12)
    INTERNAL_CAROTID_ARTERY_L = "internal_carotid_artery_l"
    INTERNAL_CAROTID_ARTERY_R = "internal_carotid_artery_r"
    ANTERIOR_CEREBRAL_ARTERY_ACA_L = "anterior_cerebral_artery_aca_l"
    ANTERIOR_CEREBRAL_ARTERY_ACA_R = "anterior_cerebral_artery_aca_r"
    ANTERIOR_COMMUNICATING_ARTERY_ACOM = "anterior_communicating_artery_acom"
    MIDDLE_CEREBRAL_ARTERY_MCA_L = "middle_cerebral_artery_mca_l"
    MIDDLE_CEREBRAL_ARTERY_MCA_R = "middle_cerebral_artery_mca_r"
    BASILAR_AND_VERTEBRAL_ARTERIES = "basilar_and_vertebral_arteries"
    POSTERIOR_CEREBRAL_ARTERY_PCA_L = "posterior_cerebral_artery_pca_l"
    POSTERIOR_CEREBRAL_ARTERY_PCA_R = "posterior_cerebral_artery_pca_r"
    SUPERIOR_SAGITTAL_AND_STRAIGHT_SINUS = "superior_sagittal_and_straight_sinus"
    VEIN_OF_GALEN_AND_TRANSVERSE_SINUSES = "vein_of_galen_and_transverse_sinuses"


# Alias for seamless compatibility with shared architectural tests
HeartPartId = BrainPartId


class AnatomyCategory(StrEnum):
    MENINGEAL_VENTRICULAR = "MENINGES Y SISTEMA VENTRICULAR"
    CORTICAL_GYRUS = "CORTEZA CEREBRAL Y GIROS"
    ELOQUENT_CORTEX = "ÁREA ELOCUENTE (HABLA / MOTOR / SENSORIAL)"
    WHITE_MATTER_TRACT = "TRACTO DE SUSTANCIA BLANCA (CONECTOMA)"
    DEEP_NUCLEUS_DIENCEPHALON = "GANGLIOS BASALES Y DIENCÉFALO"
    LIMBIC_STRUCTURE = "SISTEMA LÍMBICO Y MEMORIA"
    BRAINSTEM_CEREBELLUM = "TRONCO ENCEFÁLICO, CEREBELO Y PARES CRANEALES"
    CEREBROVASCULAR = "CIRCULACIÓN ARTERIAL Y VENOSA CEREBRAL"


class AnatomySystem(StrEnum):
    """Eight neurosurgically meaningful visibility groups used by the workstation."""

    MENINGES_AND_VENTRICLES = "Meninges y ventrículos"
    CEREBRAL_CORTEX_LOBES = "Corteza y giros cerebrales"
    ELOQUENT_SPEECH_AND_MOTOR_AREAS = "Áreas elocuentes (Habla y Motor)"
    WHITE_MATTER_CONNECTOME = "Tractos y conectoma eléctrico"
    BASAL_GANGLIA_AND_DIENCEPHALON = "Ganglios basales y tálamo"
    LIMBIC_AND_MEMORY_SYSTEM = "Sistema límbico y memoria"
    BRAINSTEM_CEREBELLUM_CRANIAL_NERVES = "Tronco, cerebelo y pares craneales"
    CEREBROVASCULAR_CIRCULATION = "Polígono de Willis y senos venosos"


_CATEGORY_TO_SYSTEM: dict[AnatomyCategory, AnatomySystem] = {
    AnatomyCategory.MENINGEAL_VENTRICULAR: AnatomySystem.MENINGES_AND_VENTRICLES,
    AnatomyCategory.CORTICAL_GYRUS: AnatomySystem.CEREBRAL_CORTEX_LOBES,
    AnatomyCategory.ELOQUENT_CORTEX: AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
    AnatomyCategory.WHITE_MATTER_TRACT: AnatomySystem.WHITE_MATTER_CONNECTOME,
    AnatomyCategory.DEEP_NUCLEUS_DIENCEPHALON: AnatomySystem.BASAL_GANGLIA_AND_DIENCEPHALON,
    AnatomyCategory.LIMBIC_STRUCTURE: AnatomySystem.LIMBIC_AND_MEMORY_SYSTEM,
    AnatomyCategory.BRAINSTEM_CEREBELLUM: AnatomySystem.BRAINSTEM_CEREBELLUM_CRANIAL_NERVES,
    AnatomyCategory.CEREBROVASCULAR: AnatomySystem.CEREBROVASCULAR_CIRCULATION,
}


class BrainActivityPhase(StrEnum):
    """Six sequential neurophysiological phases of sensorimotor, cognitive & speech processing."""

    SENSORY_TRANSDUCTION = "Recepción y transducción tálamo-cortical"
    WERNICKE_COMPREHENSION = "Decodificación semántica (Wernicke BA22)"
    LIMBIC_HIPPOCAMPAL_INTEGRATION = "Integración límbica y memoria (Papez)"
    ARCUATE_BROCA_SYNTHESIS = "Conducción dorsal y síntesis del habla (Broca)"
    MOTOR_PLANNING_BASAL_CEREBELLAR = "Modulación ganglios basales y cerebelo"
    CORTICOSPINAL_EXECUTION = "Ejecución motora piramidal y corticobulbar"


CardiacPhase = BrainActivityPhase


class GestureKind(StrEnum):
    NONE = "None"
    OPEN_PALM = "Open_Palm"
    CLOSED_FIST = "Closed_Fist"
    POINT = "Pointing_Up"
    THUMB_UP = "Thumb_Up"
    THUMB_DOWN = "Thumb_Down"
    VICTORY = "Victory"
    THREE_FINGERS = "Three_Fingers"
    PINKY = "Pinky"
    PINCH = "Pinch"
    FLOW = "Index_Pinky"


@dataclass(frozen=True, slots=True)
class ClinicalMetric:
    label: str
    value: str
    reference: str


@dataclass(frozen=True, slots=True)
class BrainPart:
    identifier: BrainPartId
    display_name: str
    latin_name: str
    category: AnatomyCategory
    summary: str
    function: str
    location: str
    metrics: tuple[ClinicalMetric, ...]

    @property
    def system(self) -> AnatomySystem:
        return _CATEGORY_TO_SYSTEM[self.category]


HeartPart = BrainPart


@dataclass(frozen=True, slots=True)
class HandObservation:
    """Camera-derived spatial data that deliberately excludes image pixels."""

    gesture: GestureKind
    confidence: float
    palm_x: float
    palm_y: float
    pointer_x: float
    pointer_y: float
    span: float
    hands: int
    landmarks: tuple[tuple[float, float], ...]
    captured_at: float

    @classmethod
    def empty(cls, captured_at: float = 0.0) -> "HandObservation":
        return cls(GestureKind.NONE, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0, (), captured_at)


@dataclass(frozen=True, slots=True)
class AnatomicalElement:
    """Identity of one manipulable structure in the 139-part parcellation."""

    part_id: BrainPartId
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class FragmentOffset:
    """Persistent model-space displacement of one explicitly acquired element."""

    part_id: BrainPartId
    horizontal: float
    vertical: float
    depth: float
    element_key: str = ""


@dataclass(frozen=True, slots=True)
class NeuralCycleState:
    """Neurophysiological cycle state modeling synaptic firing, EEG rhythms, and CSF/arterial pulse."""

    phase: BrainActivityPhase
    progress: float
    cortical_depolarization: float
    subcortical_modulation: float
    sensory_afferent_gate_open: bool
    motor_efferent_gate_open: bool
    cerebral_perfusion: float
    electrical_progress: float
    dominant_eeg_hz: float = 40.0

    @property
    def atrial_contraction(self) -> float:
        """Compatibility property representing cortical surface bioelectric micro-pulsation."""
        return self.cortical_depolarization

    @property
    def ventricular_contraction(self) -> float:
        """Compatibility property representing deep nuclear / brainstem pulse amplitude."""
        return self.subcortical_modulation

    @property
    def atrioventricular_valves_open(self) -> bool:
        """True when afferent thalamo-cortical & language association relays are firing."""
        return self.sensory_afferent_gate_open

    @property
    def semilunar_valves_open(self) -> bool:
        """True when efferent corticospinal, corticobulbar & motor execution tracts are firing."""
        return self.motor_efferent_gate_open

    @property
    def coronary_perfusion(self) -> float:
        return self.cerebral_perfusion

    @property
    def atrioventricular_opening(self) -> float:
        """Continuous synaptic gate factor for afferent/association neural pathways."""
        if not self.sensory_afferent_gate_open:
            return 0.0
        return 0.85 + 0.15 * math.sin(self.progress * math.tau)

    @property
    def semilunar_opening(self) -> float:
        """Continuous synaptic gate factor for efferent pyramidal/corticobulbar execution."""
        if not self.motor_efferent_gate_open:
            return 0.0
        return 0.85 + 0.15 * math.cos(self.progress * math.tau)

    @property
    def ventricular_emptying(self) -> float:
        """Smooth physiological brain pulsation (Monro-Kellie systolic-CSF wave)."""
        if self.phase == BrainActivityPhase.CORTICOSPINAL_EXECUTION:
            t = (self.progress - 0.80) / 0.20
        elif self.phase == BrainActivityPhase.ARCUATE_BROCA_SYNTHESIS:
            t = (self.progress - 0.50) / 0.16
        else:
            return 0.35 * math.sin(self.progress * math.tau) ** 2
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


CardiacCycleState = NeuralCycleState


@dataclass(frozen=True, slots=True)
class BrainSnapshot:
    rotation_x: float
    rotation_y: float
    rotation_z: float
    zoom: float
    explosion: float
    selected_part: BrainPartId | None
    realistic: bool
    beating: bool
    beat_scale: float
    bpm: int
    spatial_mode: bool
    fragment_offsets: tuple[FragmentOffset, ...]
    grabbed_fragment: BrainPartId | None
    cycle: NeuralCycleState
    visible_systems: frozenset[AnatomySystem]
    interior_view: bool = False
    selected_element: AnatomicalElement | None = None
    grabbed_element: AnatomicalElement | None = None
    flow_enabled: bool = False
    flow_times: tuple[float, float, float] = (0.0, 0.0, 0.0)


HeartSnapshot = BrainSnapshot


class BrainExperience:
    """Aggregate root that owns all 3D transformations and neurophysiological state."""

    MIN_ZOOM = 0.55
    MAX_ZOOM = 2.35

    def __init__(self) -> None:
        self._rotation = [-0.14, 0.42, -0.03]
        self._target_rotation = self._rotation.copy()
        self._zoom = self._target_zoom = 1.0
        self._explosion = self._target_explosion = 0.0
        self._selected_part: BrainPartId | None = None
        self._selected_element: AnatomicalElement | None = None
        self._realistic = self._beating = False
        self._spatial_mode = False
        self._interior_view = False
        self._fragment_offsets: dict[tuple[BrainPartId, str], tuple[float, float, float]] = {}
        self._grabbed_fragment: BrainPartId | None = None
        self._grabbed_element: AnatomicalElement | None = None
        self._flow_enabled = False
        self._flow_times = [0.0, 0.0, 0.0]
        self._bpm = 40  # 40 Hz Gamma cognitive binding / 60 cycles per min reference
        self._beat_clock = 0.0
        self._visible_systems = set(AnatomySystem)
        # Meninges start hidden so the pial cortical surface and gyri are immediately visible,
        # just as pericardium starts hidden in OloHeart.
        self._visible_systems.discard(AnatomySystem.MENINGES_AND_VENTRICLES)

    def select(self, part: BrainPartId, element: AnatomicalElement | None = None) -> None:
        self._selected_part = part
        self._selected_element = element
        system = anatomy_system_for(part)
        self._visible_systems.add(system)
        if system in {
            AnatomySystem.WHITE_MATTER_CONNECTOME,
            AnatomySystem.BASAL_GANGLIA_AND_DIENCEPHALON,
            AnatomySystem.LIMBIC_AND_MEMORY_SYSTEM,
        } or part in {
            BrainPartId.INSULAR_CORTEX_L,
            BrainPartId.INSULAR_CORTEX_R,
            BrainPartId.PRIMARY_AUDITORY_HESCHL_BI,
            BrainPartId.THIRD_VENTRICLE,
            BrainPartId.FOURTH_VENTRICLE,
            BrainPartId.CEREBRAL_AQUEDUCT_SYLVIUS,
            BrainPartId.CHOROID_PLEXUS_LATERAL,
            BrainPartId.SEPTUM_PELLUCIDUM,
            BrainPartId.DENTATE_AND_DEEP_CEREBELLAR_NUCLEI,
            BrainPartId.RETICULAR_ACTIVATING_SYSTEM_ARAS,
        }:
            self._interior_view = True

    def focus_whole_brain(self) -> None:
        self._selected_part = None
        self._selected_element = None

    def focus_whole_heart(self) -> None:
        self.focus_whole_brain()

    def rotate_by(self, horizontal: float, vertical: float) -> None:
        self._target_rotation[1] += horizontal
        self._target_rotation[0] = self._clamp(self._target_rotation[0] + vertical, -1.42, 1.42)

    def zoom_by(self, factor: float) -> None:
        self._target_zoom = self._clamp(self._target_zoom * factor, self.MIN_ZOOM, self.MAX_ZOOM)

    def toggle_explosion(self) -> None:
        self._target_explosion = 0.0 if self._target_explosion > 0.5 else 1.0
        if self._target_explosion == 0.0:
            self._fragment_offsets.clear()
            self._grabbed_fragment = None
            self._grabbed_element = None

    def begin_fragment_drag(
        self, part: BrainPartId, element: AnatomicalElement | None = None
    ) -> bool:
        if self._target_explosion < 0.5 and self._explosion < 0.45:
            return False
        self._grabbed_fragment = part
        self._grabbed_element = element
        self._selected_part = part
        self._selected_element = element
        return True

    def move_grabbed_fragment(self, horizontal: float, vertical: float, depth: float = 0.0) -> bool:
        if self._grabbed_fragment is None:
            return False
        key = (
            self._grabbed_fragment,
            self._grabbed_element.key if self._grabbed_element else "",
        )
        current_x, current_y, current_z = self._fragment_offsets.get(key, (0.0, 0.0, 0.0))
        self._fragment_offsets[key] = (
            self._clamp(current_x + horizontal, -0.95, 0.95),
            self._clamp(current_y + vertical, -0.85, 0.85),
            self._clamp(current_z + depth, -0.85, 0.85),
        )
        return True

    def release_fragment(self) -> None:
        self._grabbed_fragment = None
        self._grabbed_element = None

    def toggle_realistic_activity(self) -> None:
        self._realistic = not self._realistic
        self._beating = self._realistic

    def toggle_realistic_heartbeat(self) -> None:
        self.toggle_realistic_activity()

    def toggle_spatial_mode(self) -> None:
        self._spatial_mode = not self._spatial_mode

    def toggle_interior_view(self) -> None:
        self._interior_view = not self._interior_view

    def toggle_neural_flow(self) -> None:
        self._flow_enabled = not self._flow_enabled
        if self._flow_enabled and not self._beating:
            self._realistic = True
            self._beating = True

    def toggle_blood_flow(self) -> None:
        self.toggle_neural_flow()

    def set_system_visibility(self, system: AnatomySystem, visible: bool) -> None:
        if visible:
            self._visible_systems.add(system)
        elif len(self._visible_systems) > 1:
            self._visible_systems.discard(system)
            if self._selected_part and anatomy_system_for(self._selected_part) == system:
                self._selected_part = None
                self._selected_element = None
            if self._grabbed_fragment and anatomy_system_for(self._grabbed_fragment) == system:
                self._grabbed_fragment = None
                self._grabbed_element = None

    def toggle_system(self, system: AnatomySystem) -> None:
        self.set_system_visibility(system, system not in self._visible_systems)

    def reset_view(self) -> None:
        self._target_rotation = [-0.14, 0.42, -0.03]
        self._target_zoom = 1.0
        self._target_explosion = 0.0
        self._selected_part = None
        self._selected_element = None
        self._interior_view = False
        self._fragment_offsets.clear()
        self._grabbed_fragment = None
        self._grabbed_element = None

    def tick(self, delta_seconds: float) -> BrainSnapshot:
        delta = max(0.0, min(0.25, delta_seconds))
        alpha = min(1.0, delta * 11.0)
        for index in range(3):
            self._rotation[index] += (self._target_rotation[index] - self._rotation[index]) * alpha
        self._zoom += (self._target_zoom - self._zoom) * alpha
        self._explosion += (self._target_explosion - self._explosion) * min(1.0, delta * 7.5)

        if self._beating:
            self._beat_clock = (self._beat_clock + delta * 0.42) % 1.0
        cycle = self._compute_cycle(self._beat_clock if self._beating else 0.0)
        if self._flow_enabled:
            # Gate 0: Continuous vascular & basal oscillations
            self._flow_times[0] = (self._flow_times[0] + delta * 0.65) % 100.0
            if self._beating and cycle.sensory_afferent_gate_open:
                self._flow_times[1] = (self._flow_times[1] + delta * 0.85) % 100.0
            if self._beating and cycle.motor_efferent_gate_open:
                self._flow_times[2] = (self._flow_times[2] + delta * 0.95) % 100.0

        beat_scale = (
            1.0 + 0.012 * cycle.cortical_depolarization + 0.018 * cycle.subcortical_modulation
            if self._beating
            else 1.0
        )
        offsets = tuple(
            FragmentOffset(part_id, x, y, z, element_key)
            for (part_id, element_key), (x, y, z) in self._fragment_offsets.items()
        )
        return BrainSnapshot(
            rotation_x=self._rotation[0],
            rotation_y=self._rotation[1],
            rotation_z=self._rotation[2],
            zoom=self._zoom,
            explosion=self._explosion,
            selected_part=self._selected_part,
            realistic=self._realistic,
            beating=self._beating,
            beat_scale=beat_scale,
            bpm=self._bpm,
            spatial_mode=self._spatial_mode,
            fragment_offsets=offsets,
            grabbed_fragment=self._grabbed_fragment,
            cycle=cycle,
            visible_systems=frozenset(self._visible_systems),
            interior_view=self._interior_view,
            selected_element=self._selected_element,
            grabbed_element=self._grabbed_element,
            flow_enabled=self._flow_enabled,
            flow_times=(self._flow_times[0], self._flow_times[1], self._flow_times[2]),
        )

    def snapshot(self) -> BrainSnapshot:
        return self.tick(0.0)

    @classmethod
    def _compute_cycle(cls, p: float) -> NeuralCycleState:
        """Compute the 6-phase neurophysiological cycle (sensory -> language -> memory -> motor)."""
        p = p % 1.0
        if p < 0.16:
            phase = BrainActivityPhase.SENSORY_TRANSDUCTION
            cortical = cls._smoothstep(p / 0.16) * 0.75
            subcortical = 0.65
            afferent_open = True
            efferent_open = False
            perfusion = 0.78
            eeg_hz = 12.0  # Alpha-Beta transition
        elif p < 0.33:
            phase = BrainActivityPhase.WERNICKE_COMPREHENSION
            t = (p - 0.16) / 0.17
            cortical = 0.75 + 0.25 * math.sin(t * math.pi)
            subcortical = 0.45
            afferent_open = True
            efferent_open = False
            perfusion = 0.84
            eeg_hz = 38.0  # Gamma semantic binding
        elif p < 0.50:
            phase = BrainActivityPhase.LIMBIC_HIPPOCAMPAL_INTEGRATION
            t = (p - 0.33) / 0.17
            cortical = 0.65
            subcortical = 0.85 * math.sin(t * math.pi) + 0.15
            afferent_open = False
            efferent_open = False
            perfusion = 0.88
            eeg_hz = 6.5  # Theta hippocampal-prefrontal coupling
        elif p < 0.68:
            phase = BrainActivityPhase.ARCUATE_BROCA_SYNTHESIS
            t = (p - 0.50) / 0.18
            cortical = 0.92 * cls._smoothstep(t)
            subcortical = 0.55
            afferent_open = True
            efferent_open = True
            perfusion = 0.94
            eeg_hz = 42.0  # High Gamma dorsal stream transfer
        elif p < 0.84:
            phase = BrainActivityPhase.MOTOR_PLANNING_BASAL_CEREBELLAR
            t = (p - 0.68) / 0.16
            cortical = 0.80
            subcortical = 0.95 * math.sin(t * math.pi)
            afferent_open = False
            efferent_open = True
            perfusion = 0.90
            eeg_hz = 22.0  # Beta sensorimotor loop
        else:
            phase = BrainActivityPhase.CORTICOSPINAL_EXECUTION
            t = (p - 0.84) / 0.16
            cortical = 1.0 - 0.40 * t
            subcortical = 0.90 * (1.0 - 0.5 * t)
            afferent_open = False
            efferent_open = True
            perfusion = 0.96
            eeg_hz = 45.0  # Pyramidal motor burst

        return NeuralCycleState(
            phase=phase,
            progress=p,
            cortical_depolarization=cortical,
            subcortical_modulation=subcortical,
            sensory_afferent_gate_open=afferent_open,
            motor_efferent_gate_open=efferent_open,
            cerebral_perfusion=perfusion,
            electrical_progress=p,
            dominant_eeg_hz=eeg_hz,
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        clamped = max(0.0, min(1.0, value))
        return clamped * clamped * (3.0 - 2.0 * clamped)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


HeartExperience = BrainExperience


def anatomy_system_for(identifier: BrainPartId) -> AnatomySystem:
    """Resolve any of the 139 selectable brain structures to its visibility system."""
    from olobrain.domain.catalog import PART_SYSTEM_MAP

    return PART_SYSTEM_MAP[identifier]


def build_anatomy_catalog() -> tuple[BrainPart, ...]:
    """Return the complete neurosurgical catalog of 139 brain structures."""
    from olobrain.domain.catalog import build_brain_catalog

    return build_brain_catalog()

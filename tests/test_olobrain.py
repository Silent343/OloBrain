"""Unit tests for OloBrain domain aggregate, 139-part neurosurgical catalog, and gestures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from olobrain.application.controller import BrainController, GestureCoordinator, InteractionAction
from olobrain.domain.model import (
    AnatomySystem,
    BrainActivityPhase,
    BrainExperience,
    BrainPartId,
    GestureKind,
    HandObservation,
)
from olobrain.infrastructure.geometry import build_brain_geometry
from olobrain.infrastructure.vision import MediaPipeHandTracker


def observation(
    gesture: GestureKind,
    captured_at: float,
    hands: int = 1,
    palm: tuple[float, float] = (0.5, 0.5),
    span: float = 1.0,
    pointer: tuple[float, float] = (0.48, 0.46),
) -> HandObservation:
    return HandObservation(gesture, 0.95, *palm, *pointer, span, hands, (), captured_at)


class BrainExperienceTests(unittest.TestCase):
    def test_zoom_is_bounded_and_state_is_smoothed(self) -> None:
        brain = BrainExperience()
        brain.zoom_by(100.0)
        for _ in range(100):
            brain.tick(0.05)
        self.assertLessEqual(brain.snapshot().zoom, BrainExperience.MAX_ZOOM)
        brain.zoom_by(0.001)
        for _ in range(100):
            brain.tick(0.05)
        self.assertGreaterEqual(brain.snapshot().zoom, BrainExperience.MIN_ZOOM)

    def test_realistic_mode_drives_bioelectric_activity_and_pulsation(self) -> None:
        brain = BrainExperience()
        self.assertFalse(brain.snapshot().beating)
        brain.toggle_realistic_activity()
        brain.tick(0.07)
        self.assertTrue(brain.snapshot().realistic)
        self.assertTrue(brain.snapshot().beating)
        self.assertGreater(brain.snapshot().beat_scale, 1.0)

    def test_neurophysiological_cycle_progresses_through_all_six_phases(self) -> None:
        brain = BrainExperience()
        brain.toggle_realistic_activity()
        seen_phases = set()
        for _ in range(120):
            state = brain.tick(0.05)
            seen_phases.add(state.cycle.phase)
            self.assertGreater(state.cycle.dominant_eeg_hz, 0.0)
            self.assertGreater(state.cycle.cerebral_perfusion, 0.7)
        self.assertEqual(set(BrainActivityPhase), seen_phases)

    def test_eight_neurosurgical_systems_are_independently_visible(self) -> None:
        brain = BrainExperience()
        self.assertEqual(8, len(AnatomySystem))
        self.assertNotIn(AnatomySystem.MENINGES_AND_VENTRICLES, brain.snapshot().visible_systems)
        brain.select(BrainPartId.DURA_MATER_CONVEXITY)
        self.assertIn(AnatomySystem.MENINGES_AND_VENTRICLES, brain.snapshot().visible_systems)
        brain.toggle_system(AnatomySystem.CEREBRAL_CORTEX_LOBES)
        self.assertNotIn(AnatomySystem.CEREBRAL_CORTEX_LOBES, brain.snapshot().visible_systems)

    def test_fragments_move_only_in_exploded_view_and_reset_on_reconstruction(self) -> None:
        brain = BrainExperience()
        self.assertFalse(brain.begin_fragment_drag(BrainPartId.BROCA_PARS_OPERCULARIS_L))
        brain.toggle_explosion()
        self.assertTrue(brain.begin_fragment_drag(BrainPartId.BROCA_PARS_OPERCULARIS_L))
        self.assertTrue(brain.move_grabbed_fragment(0.14, -0.09, 0.06))
        snapshot = brain.snapshot()
        self.assertEqual(BrainPartId.BROCA_PARS_OPERCULARIS_L, snapshot.grabbed_fragment)
        self.assertEqual(1, len(snapshot.fragment_offsets))
        self.assertAlmostEqual(0.14, snapshot.fragment_offsets[0].horizontal)
        brain.release_fragment()
        brain.toggle_explosion()
        self.assertFalse(brain.snapshot().fragment_offsets)
        self.assertIsNone(brain.snapshot().grabbed_fragment)


class CatalogAndGeometryTests(unittest.TestCase):
    def test_exact_139_catalog_parts_have_renderable_3d_geometry(self) -> None:
        controller = BrainController()
        meshes = build_brain_geometry()
        self.assertEqual(139, len(controller.catalog))
        self.assertEqual(set(AnatomySystem), {part.system for part in controller.catalog})
        self.assertEqual(
            {part.identifier for part in controller.catalog},
            {mesh.part_id for mesh in meshes},
        )
        for mesh in meshes:
            self.assertGreater(len(mesh.vertices), 5)
            self.assertGreater(len(mesh.faces), 3)
            self.assertTrue(all(max(face) < len(mesh.vertices) for face in mesh.faces))

    def test_all_139_parts_have_nonzero_explosion_vectors_and_clinical_metadata(self) -> None:
        controller = BrainController()
        meshes = build_brain_geometry()
        for part in controller.catalog:
            self.assertTrue(part.display_name)
            self.assertTrue(part.latin_name)
            self.assertTrue(part.summary)
            self.assertTrue(part.function)
            self.assertTrue(part.location)
            self.assertGreaterEqual(len(part.metrics), 3)
        for mesh in meshes:
            self.assertGreater(mesh.explosion_vector.length(), 0.15, mesh.element_key)


class GestureCoordinatorTests(unittest.TestCase):
    def test_open_palm_rotates_relative_to_previous_frame(self) -> None:
        controller = BrainController()
        coordinator = GestureCoordinator(controller)
        initial = controller.snapshot().rotation_y
        coordinator.process(observation(GestureKind.OPEN_PALM, 1.0, palm=(0.40, 0.50)))
        coordinator.process(observation(GestureKind.OPEN_PALM, 1.1, palm=(0.46, 0.50)))
        for _ in range(20):
            controller.tick(0.04)
        self.assertGreater(controller.snapshot().rotation_y, initial)

    def test_stable_victory_toggles_139_part_exploded_view_once(self) -> None:
        controller = BrainController()
        coordinator = GestureCoordinator(controller)
        coordinator.process(observation(GestureKind.VICTORY, 1.0))
        result = coordinator.process(observation(GestureKind.VICTORY, 1.6))
        self.assertEqual(InteractionAction.EXPLOSION_CHANGED, result.action)
        self.assertEqual(
            InteractionAction.NONE,
            coordinator.process(observation(GestureKind.VICTORY, 1.8)).action,
        )
        for _ in range(30):
            controller.tick(0.04)
        self.assertGreater(controller.snapshot().explosion, 0.8)


class LandmarkGestureTests(unittest.TestCase):
    @staticmethod
    def _hand(extended: set[str]) -> list[SimpleNamespace]:
        points = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
        points[5] = SimpleNamespace(x=0.0, y=-0.22, z=0.0)
        for name, joint, tip, x in (
            ("index", 6, 8, -0.18),
            ("middle", 10, 12, -0.05),
            ("ring", 14, 16, 0.09),
            ("pinky", 18, 20, 0.20),
        ):
            points[joint] = SimpleNamespace(x=x, y=-0.32, z=0.0)
            tip_y = -0.68 if name in extended else -0.16
            points[tip] = SimpleNamespace(x=x, y=tip_y, z=0.0)
        points[2] = SimpleNamespace(x=0.12, y=-0.08, z=0.0)
        points[3] = SimpleNamespace(x=0.11, y=-0.12, z=0.0)
        points[4] = SimpleNamespace(x=0.08, y=-0.15, z=0.0)
        return points

    def test_landmarks_classify_primary_control_gestures(self) -> None:
        classify = MediaPipeHandTracker._classify_gesture
        self.assertEqual(
            GestureKind.OPEN_PALM,
            classify(self._hand({"index", "middle", "ring", "pinky"}))[0],
        )
        self.assertEqual(GestureKind.VICTORY, classify(self._hand({"index", "middle"}))[0])
        self.assertEqual(
            GestureKind.THREE_FINGERS,
            classify(self._hand({"index", "middle", "ring"}))[0],
        )
        self.assertEqual(GestureKind.PINKY, classify(self._hand({"pinky"}))[0])
        self.assertEqual(GestureKind.POINT, classify(self._hand({"index"}))[0])
        self.assertEqual(GestureKind.CLOSED_FIST, classify(self._hand(set()))[0])


if __name__ == "__main__":
    unittest.main()

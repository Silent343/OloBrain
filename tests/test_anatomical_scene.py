"""Regressions for 3D neuroanatomical sections, watertight dura shell, and non-teleporting selections."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from olobrain.application.controller import BrainController, GestureCoordinator, InteractionAction
from olobrain.domain.model import AnatomySystem, BrainPartId as I, GestureKind, HandObservation
from olobrain.infrastructure.geometry import build_brain_geometry
from olobrain.presentation.scene_layout import object_drag, placement_matrix, scene_rotation, visible_mesh


class AnatomicalSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.meshes = build_brain_geometry()

    def test_selection_never_changes_any_fragment_transform(self) -> None:
        controller = BrainController()
        controller.toggle_explosion()
        for _ in range(20):
            controller.tick(0.08)
        controller.begin_fragment_drag(I.WERNICKE_AREA_L)
        controller.move_grabbed_fragment(0.12, -0.07, 0.03)
        before = controller.snapshot()
        controller.select_part(I.WERNICKE_AREA_L)
        after = controller.snapshot()
        for mesh in self.meshes:
            e = mesh.explosion_vector
            np.testing.assert_array_equal(
                placement_matrix(before, (e.x, e.y, e.z), mesh.part_id, 1.0),
                placement_matrix(after, (e.x, e.y, e.z), mesh.part_id, 1.0),
            )

    def test_camera_drag_round_trips_through_rotated_model(self) -> None:
        state = replace(BrainController().snapshot(), rotation_x=0.6, rotation_y=1.2, rotation_z=-0.3)
        dx, dy, dz = object_drag(state, 0.10, -0.05, 0.02)
        result = scene_rotation(state)[:3, :3] @ np.asarray((dx * 4, -dy * 4, dz * 3))
        np.testing.assert_allclose(result, (0.40, 0.20, 0.06), atol=1e-6)

    def test_dura_mater_dissection_section_has_watertight_closed_thickness(self) -> None:
        section = next(m for m in self.meshes if m.part_id == I.DURA_MATER_CONVEXITY and m.view == "interior")
        edges = Counter()
        for a, b, c in section.faces:
            edges.update(tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
        self.assertTrue(all(count == 2 for count in edges.values()))

    def test_exterior_and_interior_cortical_views_are_mutually_exclusive(self) -> None:
        state = BrainController().snapshot()
        surfaces = [m for m in self.meshes if m.part_id == I.SUPERIOR_FRONTAL_GYRUS_L and m.view != "all"]
        self.assertEqual({"exterior"}, {m.view for m in surfaces if visible_mesh(m, state)})
        state = replace(state, interior_view=True)
        self.assertEqual({"interior"}, {m.view for m in surfaces if visible_mesh(m, state)})

    def test_thumb_down_opens_deep_dissection_once_without_moving_selection(self) -> None:
        controller = BrainController()
        controller.select_part(I.PRIMARY_MOTOR_CORTEX_M1_L)
        coordinator = GestureCoordinator(controller)

        def hand(t: float) -> HandObservation:
            return HandObservation(GestureKind.THUMB_DOWN, 0.95, 0.5, 0.5, 0.5, 0.5, 1.0, 1, (), t)

        coordinator.process(hand(1.0))
        self.assertEqual(InteractionAction.INTERIOR_VIEW_CHANGED, coordinator.process(hand(1.6)).action)
        self.assertEqual(InteractionAction.NONE, coordinator.process(hand(1.8)).action)
        self.assertTrue(controller.snapshot().interior_view)
        self.assertEqual(I.PRIMARY_MOTOR_CORTEX_M1_L, controller.snapshot().selected_part)


if __name__ == "__main__":
    unittest.main()

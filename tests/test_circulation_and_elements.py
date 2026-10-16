"""Regressions for independent picking across 139 brain parts and phase-gated bioelectrical flow."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from olobrain.application.controller import BrainController, GestureCoordinator, InteractionAction
from olobrain.domain.model import AnatomicalElement, AnatomySystem, BrainPartId as I, GestureKind, HandObservation
from olobrain.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from olobrain.infrastructure.geometry import build_brain_geometry
from olobrain.presentation.scene_layout import placement_matrix


class IndependentElementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.meshes = build_brain_geometry()

    def test_mesh_identities_cover_all_139_catalog_parts_without_collision(self) -> None:
        ownership: dict[str, set[I]] = {}
        for mesh in self.meshes:
            self.assertTrue(mesh.element_key)
            self.assertTrue(mesh.element_name)
            ownership.setdefault(mesh.element_key, set()).add(mesh.part_id)
        self.assertEqual(139, len(ownership))
        self.assertTrue(all(len(parts) == 1 for parts in ownership.values()))
        self.assertEqual(set(I), {m.part_id for m in self.meshes})

    def test_dragging_one_brain_structure_does_not_move_any_other_of_the_139(self) -> None:
        controller = BrainController()
        controller.toggle_explosion()
        for _ in range(30):
            controller.tick(0.05)
        target = next(m for m in self.meshes if m.part_id == I.ARCUATE_FASCICULUS_L)
        before = controller.snapshot()
        element = AnatomicalElement(target.part_id, target.element_key, target.element_name)
        self.assertTrue(controller.begin_fragment_drag(element))
        controller.move_grabbed_fragment(0.15, 0.08, 0.05)
        after = controller.snapshot()
        for mesh in self.meshes:
            e = mesh.explosion_vector
            args = ((e.x, e.y, e.z), mesh.part_id, 1.0, mesh.element_key)
            a, b = placement_matrix(before, *args), placement_matrix(after, *args)
            if mesh.element_key == target.element_key:
                self.assertFalse(np.allclose(a, b))
            else:
                np.testing.assert_array_equal(a, b)


class BioelectricAndVascularFlowTests(unittest.TestCase):
    def test_neural_and_vascular_routes_gate_correctly_across_phases(self) -> None:
        controller = BrainController()
        controller.toggle_neural_flow()
        routes = build_flow_routes()
        self.assertEqual(24, len(routes))
        seen = set()
        for _ in range(250):
            state = controller.tick(0.015)
            seen.add(state.cycle.phase)
            for route in routes:
                if route.gate == 1:
                    self.assertEqual(state.cycle.sensory_afferent_gate_open, route.active(state))
                elif route.gate == 2:
                    self.assertEqual(state.cycle.motor_efferent_gate_open, route.active(state))
        self.assertEqual(6, len(seen))

    def test_exploded_brain_pauses_connected_axonal_flow(self) -> None:
        controller = BrainController()
        controller.toggle_neural_flow()
        self.assertTrue(circulation_visible(controller.snapshot()))
        controller.toggle_explosion()
        controller.tick(0.08)
        self.assertFalse(circulation_visible(controller.snapshot()))

    def test_index_and_pinky_hold_toggles_neural_flow_once(self) -> None:
        controller = BrainController()
        coordinator = GestureCoordinator(controller)
        hand = HandObservation(GestureKind.FLOW, 0.95, 0.5, 0.5, 0.5, 0.5, 1.0, 1, (), 0.0)
        coordinator.process(hand)
        self.assertEqual(
            InteractionAction.FLOW_CHANGED,
            coordinator.process(replace(hand, captured_at=0.6)).action,
        )
        self.assertTrue(controller.snapshot().flow_enabled)


if __name__ == "__main__":
    unittest.main()

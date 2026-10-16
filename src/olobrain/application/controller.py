"""Application use cases and gesture orchestration for OloBrain.

Adheres to SOLID principles:
- SRP: BrainController handles domain use-case orchestration; GestureCoordinator translates spatial input.
- OCP & DIP: Depends on abstract HandTracker protocol rather than concrete camera hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from olobrain.domain.model import (
    AnatomicalElement,
    AnatomySystem,
    BrainExperience,
    BrainPart,
    BrainPartId,
    BrainSnapshot,
    GestureKind,
    HandObservation,
    build_anatomy_catalog,
)


class HandTracker(Protocol):
    """Infrastructure boundary that never exposes raw camera images."""

    def start(
        self,
        on_observation: Callable[[HandObservation], None],
        on_status: Callable[[str], None],
    ) -> None: ...

    def stop(self) -> None: ...


class InteractionAction(StrEnum):
    NONE = "none"
    ACTIVATE_POINTER = "activate_pointer"
    FOCUS_WHOLE = "focus_whole"
    EXPLOSION_CHANGED = "explosion_changed"
    REALISM_CHANGED = "realism_changed"
    SPATIAL_MODE_CHANGED = "spatial_mode_changed"
    INTERIOR_VIEW_CHANGED = "interior_view_changed"
    FLOW_CHANGED = "flow_changed"
    FRAGMENT_GRAB_STARTED = "fragment_grab_started"
    FRAGMENT_GRAB_MOVED = "fragment_grab_moved"
    FRAGMENT_GRAB_ENDED = "fragment_grab_ended"


@dataclass(frozen=True, slots=True)
class InteractionResult:
    action: InteractionAction
    pointer_x: float = 0.5
    pointer_y: float = 0.5
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_depth: float = 0.0


class BrainController:
    """Facade exposing intention-oriented neurosurgical use cases to the desktop shell."""

    def __init__(self, brain: BrainExperience | None = None) -> None:
        self._brain = brain or BrainExperience()
        self._catalog = build_anatomy_catalog()
        self._parts_by_id = {part.identifier: part for part in self._catalog}

    @property
    def catalog(self) -> tuple[BrainPart, ...]:
        return self._catalog

    def part(self, identifier: BrainPartId | AnatomicalElement | None) -> BrainPart | None:
        if isinstance(identifier, AnatomicalElement):
            part = self._parts_by_id.get(identifier.part_id)
            return replace(part, display_name=identifier.name) if part and identifier.name else part
        return self._parts_by_id.get(identifier) if identifier else None

    def select_part(self, identifier: BrainPartId | AnatomicalElement) -> None:
        element = identifier if isinstance(identifier, AnatomicalElement) else None
        part = element.part_id if element else identifier
        if part not in self._parts_by_id:
            raise ValueError(f"Unknown brain structure: {identifier}")
        self._brain.select(part, element)

    def focus_whole_brain(self) -> None:
        self._brain.focus_whole_brain()

    def focus_whole_heart(self) -> None:
        self._brain.focus_whole_brain()

    def rotate(self, horizontal: float, vertical: float) -> None:
        self._brain.rotate_by(horizontal, vertical)

    def zoom(self, factor: float) -> None:
        self._brain.zoom_by(factor)

    def toggle_explosion(self) -> None:
        self._brain.toggle_explosion()

    def begin_fragment_drag(self, identifier: BrainPartId | AnatomicalElement) -> bool:
        element = identifier if isinstance(identifier, AnatomicalElement) else None
        part = element.part_id if element else identifier
        if part not in self._parts_by_id:
            raise ValueError(f"Unknown brain structure: {identifier}")
        return self._brain.begin_fragment_drag(part, element)

    def move_grabbed_fragment(self, horizontal: float, vertical: float, depth: float = 0.0) -> bool:
        return self._brain.move_grabbed_fragment(horizontal, vertical, depth)

    def release_fragment(self) -> None:
        self._brain.release_fragment()

    def toggle_realistic_activity(self) -> None:
        self._brain.toggle_realistic_activity()

    def toggle_realistic_heartbeat(self) -> None:
        self._brain.toggle_realistic_activity()

    def toggle_spatial_mode(self) -> None:
        self._brain.toggle_spatial_mode()

    def toggle_interior_view(self) -> None:
        self._brain.toggle_interior_view()

    def toggle_neural_flow(self) -> None:
        self._brain.toggle_neural_flow()

    def toggle_blood_flow(self) -> None:
        self._brain.toggle_neural_flow()

    def set_system_visibility(self, system: AnatomySystem, visible: bool) -> None:
        self._brain.set_system_visibility(system, visible)

    def toggle_system(self, system: AnatomySystem) -> None:
        self._brain.toggle_system(system)

    def reset_view(self) -> None:
        self._brain.reset_view()

    def tick(self, delta_seconds: float) -> BrainSnapshot:
        return self._brain.tick(delta_seconds)

    def snapshot(self) -> BrainSnapshot:
        return self._brain.snapshot()


HeartController = BrainController


class GestureCoordinator:
    """Translate noisy hand landmarks into debounced, deliberate spatial commands."""

    def __init__(self, controller: BrainController) -> None:
        self._controller = controller
        self._previous_palm: tuple[float, float] | None = None
        self._previous_span: float | None = None
        self._pinch_position: tuple[float, float, float] | None = None
        self._gesture = GestureKind.NONE
        self._gesture_started_at = 0.0
        self._gesture_fired = False

    def process(self, observation: HandObservation) -> InteractionResult:
        if observation.hands == 0 or observation.confidence < 0.55:
            self._previous_palm = None
            self._previous_span = None
            if self._pinch_position is not None:
                self._pinch_position = None
                self._controller.release_fragment()
                return InteractionResult(InteractionAction.FRAGMENT_GRAB_ENDED)
            self._gesture = GestureKind.NONE
            self._gesture_fired = False
            return InteractionResult(InteractionAction.NONE)

        if observation.gesture == GestureKind.PINCH:
            self._previous_palm = None
            self._previous_span = None
            self._reset_semantic_gesture(observation)
            snapshot = self._controller.snapshot()
            if snapshot.explosion < 0.45:
                return InteractionResult(InteractionAction.NONE)
            if self._pinch_position is None:
                self._pinch_position = (
                    observation.pointer_x,
                    observation.pointer_y,
                    observation.span,
                )
                return InteractionResult(
                    InteractionAction.FRAGMENT_GRAB_STARTED,
                    observation.pointer_x,
                    observation.pointer_y,
                )
            prev_x, prev_y, prev_span = self._pinch_position
            dx = observation.pointer_x - prev_x
            dy = observation.pointer_y - prev_y
            dz = observation.span - prev_span
            self._pinch_position = (
                observation.pointer_x,
                observation.pointer_y,
                observation.span,
            )
            return InteractionResult(
                InteractionAction.FRAGMENT_GRAB_MOVED,
                observation.pointer_x,
                observation.pointer_y,
                dx,
                dy,
                dz,
            )

        if self._pinch_position is not None:
            self._pinch_position = None
            self._controller.release_fragment()
            return InteractionResult(InteractionAction.FRAGMENT_GRAB_ENDED)

        if observation.hands >= 2:
            if self._previous_span is not None:
                ratio = max(0.88, min(1.14, observation.span / max(self._previous_span, 0.08)))
                if abs(ratio - 1.0) > 0.012:
                    self._controller.zoom(ratio)
            self._previous_span = observation.span
            self._previous_palm = (observation.palm_x, observation.palm_y)
            return InteractionResult(InteractionAction.NONE)

        self._previous_span = None
        if observation.gesture == GestureKind.OPEN_PALM:
            if self._previous_palm is not None:
                dx = observation.palm_x - self._previous_palm[0]
                dy = observation.palm_y - self._previous_palm[1]
                self._controller.rotate(dx * 3.4, dy * 2.6)
            self._previous_palm = (observation.palm_x, observation.palm_y)
            self._reset_semantic_gesture(observation)
            return InteractionResult(InteractionAction.NONE)

        self._previous_palm = None
        if observation.gesture != self._gesture:
            self._gesture = observation.gesture
            self._gesture_started_at = observation.captured_at
            self._gesture_fired = False
            return InteractionResult(InteractionAction.NONE)

        dwell = observation.captured_at - self._gesture_started_at
        minimum = 0.72 if observation.gesture == GestureKind.POINT else 0.48
        if self._gesture_fired or observation.confidence < 0.66 or dwell < minimum:
            return InteractionResult(InteractionAction.NONE)

        action = InteractionAction.NONE
        if observation.gesture == GestureKind.POINT:
            action = InteractionAction.ACTIVATE_POINTER
        elif observation.gesture == GestureKind.VICTORY:
            self._controller.toggle_explosion()
            action = InteractionAction.EXPLOSION_CHANGED
        elif observation.gesture == GestureKind.THREE_FINGERS:
            self._controller.toggle_realistic_activity()
            action = InteractionAction.REALISM_CHANGED
        elif observation.gesture == GestureKind.PINKY:
            self._controller.toggle_spatial_mode()
            action = InteractionAction.SPATIAL_MODE_CHANGED
        elif observation.gesture == GestureKind.FLOW:
            self._controller.toggle_neural_flow()
            action = InteractionAction.FLOW_CHANGED
        elif observation.gesture == GestureKind.THUMB_DOWN:
            self._controller.toggle_interior_view()
            action = InteractionAction.INTERIOR_VIEW_CHANGED
        elif observation.gesture == GestureKind.THUMB_UP:
            self._controller.focus_whole_brain()
            action = InteractionAction.FOCUS_WHOLE
        if action != InteractionAction.NONE:
            self._gesture_fired = True
        return InteractionResult(action, observation.pointer_x, observation.pointer_y)

    def _reset_semantic_gesture(self, observation: HandObservation) -> None:
        self._gesture = GestureKind.NONE
        self._gesture_started_at = observation.captured_at
        self._gesture_fired = False


class NullHandTracker:
    """No-op adapter used for demos and environments without a camera."""

    def start(self, on_observation, on_status) -> None:
        on_status("CÁMARA DESACTIVADA · MODO MOUSE NEUROQUIRÚRGICO")

    def stop(self) -> None:
        return None

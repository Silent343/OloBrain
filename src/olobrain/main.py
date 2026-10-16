"""Composition root for the OloBrain native 3D neurosurgical workstation."""

from __future__ import annotations

import argparse
from pathlib import Path

from olobrain.application.controller import BrainController, NullHandTracker
from olobrain.infrastructure.vision import MediaPipeHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OloBrain 3D spatial neurosurgical workstation (139 structures)")
    parser.add_argument("--windowed", action="store_true", help="Open in a resizable desktop window")
    parser.add_argument("--no-camera", action="store_true", help="Disable hand tracking and use mouse controls")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--smoke-test", action="store_true", help="Validate 139-part atlas composition without opening a window")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    controller = BrainController()
    if arguments.smoke_test:
        from olobrain.infrastructure.circulation_geometry import build_flow_routes
        from olobrain.infrastructure.geometry import build_brain_geometry

        meshes = build_brain_geometry()
        routes = build_flow_routes()
        assert len(controller.catalog) == 139
        assert {mesh.part_id for mesh in meshes} == {part.identifier for part in controller.catalog}
        assert len({mesh.element_key for mesh in meshes}) == 139
        print(
            f"OloBrain smoke test passed: {len(meshes)} meshes, "
            f"{len(controller.catalog)} brain structures, {len(routes)} bioelectric/vascular routes"
        )
        return 0

    # Start in Realistic Biological Tissue & Bioelectric Activity Mode by default
    controller.toggle_realistic_activity()

    tracker = (
        NullHandTracker()
        if arguments.no_camera
        else MediaPipeHandTracker(project_root / "models" / "hand_landmarker.task", arguments.camera)
    )
    from olobrain.presentation.qt_app import QtDesktopApplication, create_qt_application

    qt_process = create_qt_application()
    application = QtDesktopApplication(
        controller,
        tracker,
        fullscreen=not arguments.windowed,
    )
    if arguments.windowed:
        screen = qt_process.primaryScreen().availableGeometry()
        width = min(1540, max(1160, screen.width() - 80))
        height = min(920, max(720, screen.height() - 80))
        application.resize(width, height)
        application.move(
            screen.x() + max(0, (screen.width() - width) // 2),
            screen.y() + max(0, (screen.height() - height) // 2),
        )
        application.show()
    application.start()
    return qt_process.exec()


if __name__ == "__main__":
    raise SystemExit(main())

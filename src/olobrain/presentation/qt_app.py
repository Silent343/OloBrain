"""Qt desktop composition for the hardware-accelerated OloBrain neurosurgical workstation."""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from olobrain.application.controller import (
    BrainController,
    GestureCoordinator,
    HandTracker,
    InteractionAction,
)
from olobrain.domain.model import (
    AnatomicalElement,
    AnatomySystem,
    BrainPart,
    BrainPartId,
    BrainSnapshot,
    HandObservation,
)
from olobrain.infrastructure.geometry import build_brain_geometry
from olobrain.presentation.gpu_renderer import GpuBrainWidget
from olobrain.presentation.scene_layout import object_drag

STYLE_SHEET = """
QMainWindow, QWidget#root {
    background: #01050a;
    color: #dbe8ed;
    font-family: "Segoe UI";
}
QWidget#inspectorContent { background: #020910; }
QFrame#header, QFrame#footer {
    background: #030b13;
    border: 1px solid #132c3c;
}
QFrame#panel {
    background: #020910;
    border: 1px solid #142f41;
    border-radius: 10px;
}
QLabel#brand {
    color: #19dbea;
    font-family: "Bahnschrift SemiBold";
    font-size: 20px;
    font-weight: 700;
}
QLabel#eyebrow {
    color: #6d8794;
    font-family: "Cascadia Mono";
    font-size: 10px;
    letter-spacing: 1px;
}
QLabel#section {
    color: #1de0ef;
    font-family: "Bahnschrift SemiBold";
    font-size: 13px;
    font-weight: 700;
}
QLabel#partTitle {
    color: #e8f1f4;
    font-family: "Bahnschrift SemiBold";
    font-size: 19px;
    font-weight: 700;
}
QLabel#latin {
    color: #22d3ee;
    font-style: italic;
    font-size: 12px;
}
QLabel#body {
    color: #aebfc7;
    font-size: 12px;
    line-height: 1.35;
}
QLabel#metricValue {
    color: #22d3ee;
    font-family: "Cascadia Mono";
    font-size: 15px;
    font-weight: 700;
}
QLineEdit {
    background: #041019;
    border: 1px solid #1b4358;
    border-radius: 6px;
    padding: 5px 8px;
    color: #e2f1f5;
    font-family: "Cascadia Mono";
    font-size: 11px;
}
QLineEdit:focus {
    border-color: #22d3ee;
}
QListWidget {
    background: transparent;
    border: none;
    outline: none;
    color: #8fa4af;
    font-family: "Cascadia Mono";
    font-size: 11px;
}
QListWidget::item {
    min-height: 27px;
    padding: 3px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QListWidget::item:hover {
    background: #061923;
    color: #dcecf0;
}
QListWidget::item:selected {
    background: #082631;
    border-color: #15cfe0;
    color: #ffffff;
}
QCheckBox {
    color: #a9bdc6;
    font-family: "Cascadia Mono";
    font-size: 10px;
    spacing: 7px;
    padding: 1px 0;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #315164;
    border-radius: 3px;
    background: #041019;
}
QCheckBox::indicator:checked {
    border-color: #22d3ee;
    background: #18bed0;
}
QPushButton {
    background: #07131d;
    color: #a9bdc6;
    border: 1px solid #18394c;
    border-radius: 7px;
    padding: 8px 12px;
    font-family: "Cascadia Mono";
    font-size: 10px;
    font-weight: 600;
}
QPushButton:hover {
    color: #ffffff;
    border-color: #22d3ee;
    background: #09202b;
}
QPushButton:checked {
    color: #ffffff;
    border-color: #00e5ff;
    background: #0c3547;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #030b13;
    width: 7px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1b4758;
    min-height: 30px;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class QtDesktopApplication(QMainWindow):
    """Native neurosurgical workstation shell coordinating Qt, OpenGL, and spatial hand input."""

    def __init__(
        self,
        controller: BrainController,
        tracker: HandTracker,
        fullscreen: bool = True,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.tracker = tracker
        self.coordinator = GestureCoordinator(controller)
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.hand = HandObservation.empty(time.monotonic())
        self.camera_status = "INICIANDO SEGUIMIENTO ESPACIAL"
        self.last_frame = time.perf_counter()
        self.status_message = "ESTACIÓN NEUROQUIRÚRGICA 3D LISTA · 139 ESTRUCTURAS"
        self.status_until = time.monotonic() + 3.0
        self._last_ui_signature: tuple | None = None
        self._last_status_update = 0.0
        self._fullscreen = fullscreen

        self.setObjectName("root")
        self.setWindowTitle("OloBrain · Estación Neuroquirúrgica y Conectoma 3D (139 Estructuras)")
        self.setMinimumSize(1160, 720)
        self.resize(1540, 920)
        self.setStyleSheet(STYLE_SHEET)
        self._build_interface()
        self._connect_renderer()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        if fullscreen:
            self.showFullScreen()

    def start(self) -> None:
        self.tracker.start(self.post_observation, self.post_camera_status)
        self.timer.start()

    def post_observation(self, observation: HandObservation) -> None:
        self.events.put(("hand", observation))

    def post_camera_status(self, status: str) -> None:
        self.events.put(("status", status))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.timer.stop()
        self.tracker.stop()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_E:
            self._toggle_explosion()
        elif key in (Qt.Key.Key_R, Qt.Key.Key_Space):
            self._toggle_realism()
        elif key == Qt.Key.Key_S:
            self._toggle_spatial_mode()
        elif key == Qt.Key.Key_I:
            self._toggle_interior()
        elif key == Qt.Key.Key_F:
            self._toggle_flow()
        elif key == Qt.Key.Key_Home:
            self._reset_view()
        else:
            super().keyPressEvent(event)

    def _build_interface(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(5, 5, 5, 5)
        root_layout.setSpacing(4)
        self.setCentralWidget(root)

        self.header = QFrame(root)
        self.header.setObjectName("header")
        self.header.setFixedHeight(48)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(18, 7, 18, 7)
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("OLOBRAIN")
        brand.setObjectName("brand")
        subtitle = QLabel("ATLAS NEUROQUIRÚRGICO · 139 ESTRUCTURAS · CONECTOMA BIOELÉCTRICO")
        subtitle.setObjectName("eyebrow")
        brand_box.addWidget(brand)
        brand_box.addWidget(subtitle)
        header_layout.addLayout(brand_box)

        self.flow_button = QPushButton("CONDUCCIÓN Y FLUJO · F")
        self.flow_button.setCheckable(True)
        self.flow_button.setToolTip("Impulsos sinápticos (Na+/K+) y Polígono de Willis · índice y meñique")
        self.flow_button.clicked.connect(self._toggle_flow)
        header_layout.addWidget(self.flow_button)
        header_layout.addStretch(1)
        self.mode_label = QLabel("PARÉNQUIMA CORTICAL · SOMBREADO PBR")
        self.mode_label.setObjectName("section")
        header_layout.addWidget(self.mode_label)
        header_layout.addStretch(1)
        self.camera_label = QLabel(self.camera_status)
        self.camera_label.setObjectName("eyebrow")
        header_layout.addWidget(self.camera_label)
        root_layout.addWidget(self.header)

        self.body = QWidget(root)
        body_layout = QHBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        root_layout.addWidget(self.body, 1)

        self.left_panel = self._build_left_panel(self.body)
        body_layout.addWidget(self.left_panel)

        center = QWidget(self.body)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(7)
        self.renderer = GpuBrainWidget(build_brain_geometry(), center)
        self.renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.renderer, 1)
        self.control_bar = self._build_control_bar(center)
        center_layout.addWidget(self.control_bar)
        body_layout.addWidget(center, 1)

        self.right_panel = self._build_right_panel(self.renderer)
        self.renderer.set_inspector(self.right_panel)

        self.footer = QFrame(root)
        self.footer.setObjectName("footer")
        self.footer.setFixedHeight(28)
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(16, 5, 16, 5)
        self.status_label = QLabel(self.status_message)
        self.status_label.setObjectName("eyebrow")
        self.fps_label = QLabel("GPU -- FPS · VISIÓN -- FPS")
        self.fps_label.setObjectName("eyebrow")
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.fps_label)
        root_layout.addWidget(self.footer)

    def _build_left_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("panel")
        panel.setFixedWidth(256)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(13, 14, 13, 12)
        layout.setSpacing(6)
        title = QLabel("ATLAS CEREBRAL (139)")
        title.setObjectName("section")
        caption = QLabel("PARCELACIÓN NEUROQUIRÚRGICA")
        caption.setObjectName("eyebrow")
        layout.addWidget(title)
        layout.addWidget(caption)

        systems_title = QLabel("AISLAR SISTEMAS (8)")
        systems_title.setObjectName("eyebrow")
        layout.addSpacing(2)
        layout.addWidget(systems_title)
        visible_systems = self.controller.snapshot().visible_systems
        self.system_checks: dict[AnatomySystem, QCheckBox] = {}
        for system in AnatomySystem:
            checkbox = QCheckBox(system.value, panel)
            checkbox.setChecked(system in visible_systems)
            checkbox.toggled.connect(
                lambda checked, selected_system=system: self._set_system_visibility(
                    selected_system, checked
                )
            )
            self.system_checks[system] = checkbox
            layout.addWidget(checkbox)

        structures_title = QLabel("139 REGIONES SELECCIONABLES")
        structures_title.setObjectName("eyebrow")
        layout.addSpacing(3)
        layout.addWidget(structures_title)

        self.search_input = QLineEdit(panel)
        self.search_input.setPlaceholderText("Filtrar giro, BA, par o función...")
        self.search_input.textChanged.connect(self._filter_part_list)
        layout.addWidget(self.search_input)

        self.part_list = QListWidget(panel)
        self.part_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.part_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.part_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.part_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        for part in self.controller.catalog:
            item = QListWidgetItem(f"  •  {part.display_name}")
            item.setData(Qt.ItemDataRole.UserRole, part.identifier.value)
            item.setToolTip(f"{part.display_name} — {part.latin_name}")
            self.part_list.addItem(item)
        self.part_list.itemClicked.connect(self._select_list_item)
        layout.addWidget(self.part_list, 1)

        help_label = QLabel(
                "MANO ABIERTA  ROTAR\n"
                "DOS MANOS  ZOOM\n"
                "ÍNDICE  SELECCIONAR\n"
                "VICTORIA (E)  ESPARCIR 139 PARTES\n"
                "PINZA  MOVER ESTRUCTURA\n"
                "PULGAR ABAJO (I)  DISECCIÓN\n"
                "ÍNDICE+MEÑIQUE (F)  SINAPSIS"
        )
        help_label.setObjectName("eyebrow")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        return panel

    def _filter_part_list(self, query: str) -> None:
        needle = query.strip().lower()
        for idx in range(self.part_list.count()):
            item = self.part_list.item(idx)
            part_id = BrainPartId(item.data(Qt.ItemDataRole.UserRole))
            part = self.controller.part(part_id)
            if not needle or part is None:
                item.setHidden(False)
                continue
            haystack = f"{part.display_name} {part.latin_name} {part.category.value} {part.function}".lower()
            item.setHidden(needle not in haystack)

    def _build_control_bar(self, parent: QWidget) -> QFrame:
        bar = QFrame(parent)
        bar.setObjectName("panel")
        bar.setFixedHeight(42)
        layout = QGridLayout(bar)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(4)
        self.explosion_button = QPushButton("ESPARCIR (139) · E")
        self.realism_button = QPushButton("BIOELÉCTRICO · R")
        self.spatial_button = QPushButton("ESPACIAL · S")
        self.interior_button = QPushButton("DISECCIÓN PROFUNDA · I")
        self.interior_button.setCheckable(True)
        self.interior_button.clicked.connect(self._toggle_interior)
        reset_button = QPushButton("RECONSTRUIR")
        clear_button = QPushButton("QUITAR SELECCIÓN")
        clear_button.clicked.connect(lambda: self._activate_part(None))
        self.explosion_button.setCheckable(True)
        self.realism_button.setCheckable(True)
        self.spatial_button.setCheckable(True)
        self.explosion_button.clicked.connect(self._toggle_explosion)
        self.realism_button.clicked.connect(self._toggle_realism)
        self.spatial_button.clicked.connect(self._toggle_spatial_mode)
        reset_button.clicked.connect(self._reset_view)
        for index, button in enumerate(
            (
                self.interior_button,
                self.explosion_button,
                self.realism_button,
                self.spatial_button,
                clear_button,
                reset_button,
            )
        ):
            layout.addWidget(button, 0, index)
        return bar

    def _build_right_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(panel)
        self.info_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content.setObjectName("inspectorContent")
        self.info_layout = QVBoxLayout(content)
        self.info_layout.setContentsMargins(18, 18, 18, 18)
        self.info_layout.setSpacing(9)
        self.part_title = QLabel("ENCÉFALO HUMANO")
        self.part_title.setObjectName("partTitle")
        self.part_title.setWordWrap(True)
        self.part_latin = QLabel("Encephalon · 139 Estructuras Parceladas (HCP-MMP1.0 / Rhoton)")
        self.part_latin.setObjectName("latin")
        self.part_latin.setWordWrap(True)
        self.part_category = QLabel("ESTACIÓN NEUROQUIRÚRGICA INTERACTIVA")
        self.part_category.setObjectName("eyebrow")
        self.part_category.setWordWrap(True)
        self.part_summary = QLabel(
            "Selecciona cualquiera de las 139 estructuras cerebrales con la mano, el mouse o el "
            "buscador izquierdo para inspeccionar su función corporal, del habla, conectoma y correlato quirúrgico."
        )
        self.part_summary.setObjectName("body")
        self.part_summary.setWordWrap(True)
        self.info_layout.addWidget(self.part_title)
        self.info_layout.addWidget(self.part_latin)
        self.info_layout.addWidget(self.part_category)
        self.info_layout.addSpacing(6)
        self.info_layout.addWidget(self.part_summary)

        self.metric_labels: list[tuple[QLabel, QLabel, QLabel]] = []
        for _ in range(3):
            metric_frame = QFrame(content)
            metric_frame.setObjectName("panel")
            metric_layout = QVBoxLayout(metric_frame)
            metric_layout.setContentsMargins(11, 8, 11, 8)
            metric_layout.setSpacing(2)
            label = QLabel("")
            label.setObjectName("eyebrow")
            label.setWordWrap(True)
            value = QLabel("")
            value.setWordWrap(True)
            value.setObjectName("metricValue")
            reference = QLabel("")
            reference.setObjectName("eyebrow")
            reference.setWordWrap(True)
            metric_layout.addWidget(label)
            metric_layout.addWidget(value)
            metric_layout.addWidget(reference)
            self.info_layout.addWidget(metric_frame)
            self.metric_labels.append((label, value, reference))

        self.function_title = QLabel("FUNCIÓN CORPORAL, HABLA Y NEUROFISIOLOGÍA")
        self.function_title.setObjectName("section")
        self.function_text = QLabel(
            "Integración sensoriomotora, comprensión y producción del lenguaje (red dorsal/ventral), "
            "consolidación mnésica, control autonómico y conducción de potenciales de acción."
        )
        self.function_text.setObjectName("body")
        self.function_text.setWordWrap(True)

        self.location_title = QLabel("RELACIONES MICROQUIRÚRGICAS (RHOTON)")
        self.location_title.setObjectName("section")
        self.location_text = QLabel(
            "Compartimentos supratentorial e infratentorial, irrigados por el Polígono de Willis y drenados por senos durales."
        )
        self.location_text.setObjectName("body")
        self.location_text.setWordWrap(True)

        self.info_layout.addSpacing(8)
        self.info_layout.addWidget(self.function_title)
        self.info_layout.addWidget(self.function_text)
        self.info_layout.addSpacing(6)
        self.info_layout.addWidget(self.location_title)
        self.info_layout.addWidget(self.location_text)
        self.info_layout.addItem(
            QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        scroll.setWidget(content)
        outer.addWidget(scroll)

        navigation = QHBoxLayout()
        navigation.setContentsMargins(8, 4, 8, 8)
        for caption, direction in (("↑ TEXTO", -1), ("TEXTO ↓", 1)):
            button = QPushButton(caption, panel)
            button.clicked.connect(
                lambda checked=False, step=direction: scroll.verticalScrollBar().setValue(
                    scroll.verticalScrollBar().value() + step * 160
                )
            )
            navigation.addWidget(button)
        outer.addLayout(navigation)
        return panel

    def _connect_renderer(self) -> None:
        self.renderer.part_activated.connect(self._activate_part)
        self.renderer.fragment_pressed.connect(self._begin_fragment)
        self.renderer.fragment_moved.connect(self._move_fragment)
        self.renderer.fragment_released.connect(self._release_fragment)
        self.renderer.rotation_requested.connect(self.controller.rotate)
        self.renderer.zoom_requested.connect(self.controller.zoom)

    def _tick(self) -> None:
        now = time.perf_counter()
        delta = min(0.08, now - self.last_frame)
        self.last_frame = now
        self._pump_events()
        snapshot = self.controller.tick(delta)
        self.renderer.set_snapshot(snapshot)
        origin = self.renderer.mapTo(self, QPoint(0, 0))
        self.renderer.set_hand_overlay(
            self.hand,
            float(self.width()),
            float(self.height()),
            float(origin.x()),
            float(origin.y()),
        )
        signature = (
            snapshot.selected_part,
            snapshot.selected_element,
            snapshot.flow_enabled,
            snapshot.realistic,
            snapshot.spatial_mode,
            snapshot.interior_view,
            snapshot.explosion >= 0.45,
            tuple(sorted(system.value for system in snapshot.visible_systems)),
        )
        if signature != self._last_ui_signature:
            self._last_ui_signature = signature
            self._update_interface(snapshot)
        if now - self._last_status_update >= 0.25:
            self._last_status_update = now
            message = self.status_message if now < self.status_until else self.camera_status
            self.status_label.setText(message)
            self.camera_label.setText(self.camera_status)
            if snapshot.realistic:
                self.mode_label.setText(
                    f"{snapshot.cycle.phase.value.upper()} · EEG {snapshot.cycle.dominant_eeg_hz:.0f} HZ"
                )
                if snapshot.selected_part is None:
                    self._show_cycle_status(snapshot)
            self.fps_label.setText(
                f"GPU {self.renderer.render_fps:02.0f} FPS  ·  139 ESTRUCTURAS  ·  SIN VIDEO EN PANTALLA"
            )

    def _pump_events(self) -> None:
        latest: HandObservation | None = None
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "hand":
                latest = payload  # type: ignore[assignment]
            elif kind == "status":
                self.camera_status = str(payload)
        if latest is None:
            return
        self.hand = latest
        result = self.coordinator.process(latest)
        if result.action == InteractionAction.ACTIVATE_POINTER:
            self._activate_from_normalized(result.pointer_x, result.pointer_y)
        elif result.action == InteractionAction.FRAGMENT_GRAB_STARTED:
            part = self._pick_from_normalized(result.pointer_x, result.pointer_y)
            if part is not None:
                self._begin_fragment(part)
        elif result.action == InteractionAction.FRAGMENT_GRAB_MOVED:
            self._move_fragment(result.delta_x, result.delta_y, result.delta_depth)
        elif result.action == InteractionAction.FRAGMENT_GRAB_ENDED:
            self._release_fragment()
        elif result.action == InteractionAction.EXPLOSION_CHANGED:
            self._flash("ESPARCIMIENTO DE LAS 139 ESTRUCTURAS CEREBRALES ACTIVO")
        elif result.action == InteractionAction.REALISM_CHANGED:
            self._flash("TEJIDO ENCEFÁLICO VIVO · ACTIVIDAD BIOELÉCTRICA SINCRONIZADA")
        elif result.action == InteractionAction.SPATIAL_MODE_CHANGED:
            self._flash("VISTA ESPACIAL NEUROQUIRÚRGICA A PANTALLA COMPLETA")
        elif result.action == InteractionAction.INTERIOR_VIEW_CHANGED:
            self._flash(
                "DISECCIÓN SUBCORTICAL PROFUNDA"
                if self.controller.snapshot().interior_view
                else "CONVEXIDAD CORTICAL EXTERIOR"
            )
        elif result.action == InteractionAction.FLOW_CHANGED:
            self._flash(
                "CONDUCCIÓN SINÁPTICA Y FLUJO VASCULAR "
                + ("ACTIVOS" if self.controller.snapshot().flow_enabled else "OCULTOS")
            )
        elif result.action == InteractionAction.FOCUS_WHOLE:
            self._flash("ENCÉFALO COMPLETO RESTAURADO")

    def _pick_from_normalized(self, x: float, y: float) -> AnatomicalElement | None:
        origin = self.renderer.mapTo(self, QPoint(0, 0))
        local_x = x * self.width() - origin.x()
        local_y = y * self.height() - origin.y()
        if not (0 <= local_x < self.renderer.width() and 0 <= local_y < self.renderer.height()):
            return None
        return self.renderer.pick_part(local_x, local_y)

    def _activate_from_normalized(self, x: float, y: float) -> None:
        point = QPoint(round(x * self.width()), round(y * self.height()))
        widget = self.childAt(point)
        while widget is not None and widget is not self:
            if isinstance(widget, (QCheckBox, QPushButton)):
                widget.click()
                return
            if widget is self.part_list.viewport():
                item = self.part_list.itemAt(widget.mapFrom(self, point))
                if item is not None:
                    self._select_list_item(item)
                return
            if widget is self.right_panel:
                return
            widget = widget.parentWidget()
        self._activate_part(self._pick_from_normalized(x, y))

    def _activate_part(self, identifier: BrainPartId | AnatomicalElement | None) -> None:
        if identifier is None:
            if self.controller.snapshot().selected_part is not None:
                self.controller.focus_whole_brain()
                self._flash("ENCÉFALO COMPLETO RESTAURADO")
            return
        self.controller.select_part(identifier)
        part = self.controller.part(identifier)
        self._flash("ESTRUCTURA SELECCIONADA · " + (part.display_name.upper() if part else ""))

    def _begin_fragment(self, identifier: BrainPartId | AnatomicalElement) -> None:
        if self.controller.begin_fragment_drag(identifier):
            part = self.controller.part(identifier)
            self._flash("ESTRUCTURA TOMADA · " + (part.display_name.upper() if part else ""))

    def _move_fragment(self, horizontal: float, vertical: float, depth: float) -> None:
        self.controller.move_grabbed_fragment(
            *object_drag(self.controller.snapshot(), horizontal, vertical, depth)
        )

    def _release_fragment(self) -> None:
        grabbed = self.controller.snapshot().grabbed_fragment
        self.controller.release_fragment()
        if grabbed is not None:
            self._flash("ESTRUCTURA LIBERADA")

    def _select_list_item(self, item: QListWidgetItem) -> None:
        self._activate_part(BrainPartId(item.data(Qt.ItemDataRole.UserRole)))

    def _update_interface(self, snapshot: BrainSnapshot) -> None:
        self.flow_button.setChecked(snapshot.flow_enabled)
        self.explosion_button.setChecked(snapshot.explosion >= 0.45)
        self.realism_button.setChecked(snapshot.realistic)
        self.spatial_button.setChecked(snapshot.spatial_mode)
        self.interior_button.setChecked(snapshot.interior_view)
        self.mode_label.setText(
            "TEJIDO ENCEFÁLICO VIVO · BIOELECTRICIDAD ACTIVA"
            if snapshot.realistic
            else ("DISECCIÓN SUBCORTICAL" if snapshot.interior_view else "CONVEXIDAD ANATÓMICA")
        )
        self.left_panel.setVisible(not snapshot.spatial_mode)
        self.header.setVisible(not snapshot.spatial_mode)
        self.footer.setVisible(not snapshot.spatial_mode)
        self.control_bar.setVisible(not snapshot.spatial_mode)
        part = self.controller.part(snapshot.selected_element or snapshot.selected_part)
        self._show_part(part)
        self.renderer.set_callout(part)
        self.right_panel.setVisible(part is not None or not snapshot.spatial_mode)
        for system, checkbox in self.system_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(system in snapshot.visible_systems)
            checkbox.blockSignals(False)
        if part is None:
            self.part_list.clearSelection()
        else:
            for index in range(self.part_list.count()):
                item = self.part_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == part.identifier.value:
                    self.part_list.setCurrentItem(item)
                    self.part_list.scrollToItem(item)
                    break

    def _show_part(self, part: BrainPart | None) -> None:
        identifier = (part.identifier, part.display_name) if part else None
        if getattr(self, "_inspected_part", None) != identifier:
            self.info_scroll.verticalScrollBar().setValue(0)
        self._inspected_part = identifier
        if part is None:
            self.part_title.setText("ENCÉFALO HUMANO (139 PARTES)")
            self.part_latin.setText("Encephalon · Glasser HCP-MMP1.0 & Rhoton Microsurgical Atlas")
            self.part_category.setText("EXPLORACIÓN NEUROQUIRÚRGICA Y FUNCIONAL")
            self.part_summary.setText(
                "Selecciona cualquiera de las 139 estructuras con la mano o desde el panel izquierdo. "
                "Presiona E (o gesto de Victoria) para el esparcimiento 3D completo de las 139 regiones, "
                "I para disección subcortical y F para visualizar potenciales de acción y flujo arterial."
            )
            metrics = (
                ("ESTRUCTURAS PARCELADAS", "139 Regiones", "Glasser Nature 2016 / Rhoton"),
                ("NEURONAS Y SINAPSIS", "≈ 86×10⁹ neuronas", "Herculano-Houzel 2009"),
                ("FLUJO SANGUÍNEO (CBF)", "50 mL/100g/min", "Polígono de Willis (750 mL/min)"),
            )
            self.function_text.setText(
                "Gobierna el movimiento voluntario contralateral (vía piramidal), la comprensión y articulación "
                "del habla (Wernicke-Arqueado-Broca, Hickok & Poeppel), la memoria episódica (circuito de Papez), "
                "los sentidos especiales y la homeostasis cardiorrespiratoria y neuroendocrina."
            )
            self.location_text.setText(
                "Cavidad craneal supratentorial (hemisferios y diencéfalo) e infratentorial (tronco y cerebelo)."
            )
        else:
            self.part_title.setText(part.display_name.upper())
            self.part_latin.setText(part.latin_name)
            self.part_category.setText(f"{part.category.value}  ·  {part.system.value.upper()}")
            self.part_summary.setText(part.summary)
            metrics = tuple((m.label, m.value, m.reference) for m in part.metrics[:3])
            self.function_text.setText(part.function)
            self.location_text.setText(part.location)

        for idx, (lbl_w, val_w, ref_w) in enumerate(self.metric_labels):
            if idx < len(metrics):
                lbl, val, ref = metrics[idx]
                lbl_w.setText(lbl)
                val_w.setText(val)
                ref_w.setText(ref)
                lbl_w.parentWidget().setVisible(True)
            else:
                lbl_w.parentWidget().setVisible(False)

    def _show_cycle_status(self, snapshot: BrainSnapshot) -> None:
        cycle = snapshot.cycle
        metrics = (
            ("FASE NEUROFISIOLÓGICA", cycle.phase.value, "Ciclo sensoriomotor y del habla"),
            ("OSCILACIÓN EEG DOMINANTE", f"{cycle.dominant_eeg_hz:.1f} Hz", "Buzsáki, Science 2004"),
            ("PERFUSIÓN CEREBRAL", f"{int(cycle.cerebral_perfusion * 100)}%", "Acoplamiento neurovascular BOLD"),
        )
        for idx, (lbl_w, val_w, ref_w) in enumerate(self.metric_labels):
            lbl, val, ref = metrics[idx]
            lbl_w.setText(lbl)
            val_w.setText(val)
            ref_w.setText(ref)

    def _set_system_visibility(self, system: AnatomySystem, visible: bool) -> None:
        self.controller.set_system_visibility(system, visible)
        self._flash(f"SISTEMA {'MOSTRADO' if visible else 'OCULTO'} · {system.value.upper()}")

    def _toggle_explosion(self) -> None:
        self.controller.toggle_explosion()
        state = self.controller.snapshot()
        self._flash(
            "ESPARCIMIENTO 3D DE LAS 139 PARTES ACTIVO"
            if state.explosion >= 0.45 or self.explosion_button.isChecked()
            else "ENCÉFALO ENSAMBLADO"
        )

    def _toggle_interior(self) -> None:
        self.controller.toggle_interior_view()
        state = self.controller.snapshot()
        self._flash("DISECCIÓN SUBCORTICAL ACTIVA" if state.interior_view else "VISTA CORTICAL EXTERIOR")

    def _toggle_flow(self) -> None:
        self.controller.toggle_neural_flow()
        self._flash(
            "POTENCIALES DE ACCIÓN Y POLÍGONO DE WILLIS "
            + ("ACTIVOS" if self.controller.snapshot().flow_enabled else "OCULTOS")
        )

    def _toggle_realism(self) -> None:
        self.controller.toggle_realistic_activity()
        state = self.controller.snapshot()
        self._flash(
            "TEJIDO BIOLÓGICO REALISTA · ACTIVIDAD BIOELÉCTRICA"
            if state.realistic
            else "MODO ANALÍTICO PARCELADO"
        )

    def _toggle_spatial_mode(self) -> None:
        self.controller.toggle_spatial_mode()
        state = self.controller.snapshot()
        self._flash(
            "VISTA ESPACIAL A PANTALLA COMPLETA"
            if state.spatial_mode
            else "ESTACIÓN NEUROQUIRÚRGICA RESTAURADA"
        )

    def _reset_view(self) -> None:
        self.controller.reset_view()
        self._flash("VISTA NEUROANATÓMICA RESTAURADA")

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.showFullScreen() if self._fullscreen else self.showNormal()

    def _flash(self, message: str) -> None:
        self.status_message = message
        self.status_until = time.monotonic() + 2.4
        self.status_label.setText(message)


def configure_opengl() -> None:
    """Request one multisampled OpenGL 3.3 Core context before QApplication is created."""
    from PySide6.QtGui import QSurfaceFormat

    surface = QSurfaceFormat()
    surface.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    surface.setSamples(2)
    surface.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface)


def create_qt_application() -> QApplication:
    """Create the native Qt process after OpenGL 3.3 Core format is configured."""
    configure_opengl()
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("OloBrain")
    application.setOrganizationName("OloBrain")
    return application

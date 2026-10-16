"""Hardware-accelerated 3D neuroanatomical renderer v2.0 with Mouse-Hover RBC/Synaptic Microscope."""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np
import OpenGL

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BACK,
    GL_BLEND,
    GL_CCW,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DITHER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LEQUAL,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glCullFace,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDrawElements,
    glEnable,
    glEnableVertexAttribArray,
    glFrontFace,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glReadPixels,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from olobrain.domain.model import (
    AnatomicalElement,
    AnatomySystem,
    BrainActivityPhase,
    BrainPart,
    BrainPartId,
    BrainSnapshot,
    HandObservation,
    anatomy_system_for,
)
from olobrain.infrastructure.circulation_geometry import circulation_visible
from olobrain.infrastructure.geometry import AnatomicalMesh
from olobrain.presentation.flow_renderer import NeuralFlowRenderer
from olobrain.presentation.scene_layout import placement_matrix, visible_mesh

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform float uCorticalPulse;
uniform float uSubcorticalPulse;
uniform float uContractileGroup;
uniform float uIsTissue;
uniform vec3 uDeformCenter;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec3 vObjectPosition;
out vec3 vObjectNormal;

void main() {
    vec3 position = aPosition;
    vec3 objectNormal = aNormal;
    if (uIsTissue > 0.5 && uContractileGroup > 0.5) {
        float pulse = uContractileGroup < 1.5
            ? uCorticalPulse * 0.010
            : uSubcorticalPulse * 0.014;
        vec3 deformation = vec3(1.0 + pulse, 1.0 + pulse * 0.8, 1.0 + pulse);
        position = uDeformCenter + (position - uDeformCenter) * deformation;
        objectNormal = normalize(aNormal / deformation);
    }
    vec4 world = uModel * vec4(position, 1.0);
    vWorldPosition = world.xyz;
    vObjectPosition = position;
    vObjectNormal = aNormal;
    vNormal = normalize(transpose(inverse(mat3(uModel))) * objectNormal);
    gl_Position = uProjection * uView * world;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vWorldPosition;
in vec3 vNormal;
in vec3 vObjectPosition;
in vec3 vObjectNormal;

uniform vec3 uBaseColor;
uniform vec3 uCameraPosition;
uniform vec3 uPickColor;
uniform vec3 uMeshCenter;
uniform float uRoughness;
uniform float uSheen;
uniform float uOpacity;
uniform float uIsTissue;
uniform float uHighlighted;
uniform float uHovered;
uniform float uAnalytical;
uniform float uActivation;
uniform float uCycleProgress;
uniform float uBioelectricConduction;
uniform float uArrivalPhase;
uniform int uPicking;

out vec4 fragmentColor;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float tissueNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x),
            mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x),
            mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y),
        f.z
    );
}

void main() {
    if (uPicking == 1) {
        fragmentColor = vec4(uPickColor, 1.0);
        return;
    }

    float organic = uIsTissue > 0.5 ? tissueNoise(vObjectPosition * 14.0) : 0.5;
    float microSulci = uIsTissue > 0.5 ? tissueNoise(vObjectPosition * 34.0) : 0.5;
    vec3 wp = vObjectPosition + 0.085 * vec3(
        sin(vObjectPosition.y * 5.2 + vObjectPosition.z * 4.4),
        cos(vObjectPosition.z * 5.0 - vObjectPosition.x * 4.6),
        sin(vObjectPosition.x * 5.4 + vObjectPosition.y * 4.8)
    );
    float gyralA = sin(wp.x * 8.2 + wp.y * 7.8 + 1.20 * cos(wp.z * 7.5 - wp.y * 6.5));
    float gyralB = cos(wp.z * 8.5 - wp.y * 7.5 + 1.10 * sin(wp.x * 8.0 + wp.z * 6.8));
    float sulcalField = gyralA;

    vec3 normal = normalize(gl_FrontFacing ? vNormal : -vNormal);
    if (uIsTissue > 0.5) {
        vec3 microNormal = vec3(gyralA, gyralB, (microSulci - 0.5) * 1.8);
        normal = normalize(normal + microNormal * 0.042);
    }

    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);
    vec3 keyDirection = normalize(vec3(-0.42, 0.74, 1.15));
    vec3 fillDirection = normalize(vec3(0.76, -0.20, 0.60));
    vec3 rimDirection = normalize(vec3(-0.55, 0.72, -0.48));

    float key = max(dot(normal, keyDirection), 0.0);
    float fill = max(dot(normal, fillDirection), 0.0);
    float rear = max(dot(normal, rimDirection), 0.0);
    vec3 halfVector = normalize(keyDirection + viewDirection);
    float glossPower = mix(92.0, 26.0, uRoughness);
    float specular = pow(max(dot(normal, halfVector), 0.0), glossPower);
    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    // Realistic deep sulcal vascular darkening and pial capillary network
    float s1 = exp(-pow(gyralA / 0.28, 2.0)); float s2 = exp(-pow(gyralB / 0.24, 2.0)) * exp(-pow((gyralA - 0.50) / 0.42, 2.0)); float sulcusDepth = max(s1, s2 * 0.85);
    float pialCapillary = exp(-pow(sin(vObjectPosition.y * 36.0 + organic * 6.0) / 0.06, 2.0));
    vec3 base = uBaseColor * (0.94 + organic * 0.12);
    if (uIsTissue > 0.5 && uAnalytical < 0.5) {
        // Darken sulcal troughs with vascular reddish-brown and tint fine pial vessels
        base = mix(base, vec3(0.36, 0.13, 0.12), sulcusDepth * 0.42);
        base = mix(base, vec3(0.76, 0.16, 0.15), pialCapillary * 0.16);
    }

    vec3 diffuse = base * (0.24 + key * 0.88 + fill * 0.34 + rear * 0.12);
    vec3 arachnoidWetSheen = vec3(1.0, 0.90, 0.85) * specular * (0.20 + uSheen * 0.46);
    vec3 subsurfacePial = mix(base, vec3(0.48, 0.08, 0.06), 0.55)
                        * fresnel * (0.12 + uSheen * 0.20);
    vec3 color = diffuse + arachnoidWetSheen + subsurfacePial;

    color += base * uActivation * 0.32;
    if (uBioelectricConduction > 0.5) {
        float spatialWave = sin(vObjectPosition.z * 10.0 - vObjectPosition.y * 8.0 - uCycleProgress * 18.85);
        float spike = pow(max(0.0, spatialWave), 6.0);
        float arrivalWindow = exp(-pow(fract(uCycleProgress - uArrivalPhase + 1.0) - 0.15, 2.0) / 0.035);
        vec3 actionPotentialGlow = mix(vec3(0.10, 0.90, 1.0), vec3(1.0, 0.84, 0.28), spike);
        color += actionPotentialGlow * (0.26 * spike + 0.38 * arrivalWindow);
    }

    if (uHovered > 0.5) {
        color += fresnel * vec3(0.15, 0.65, 0.75) + vec3(0.06, 0.18, 0.22);
    }
    if (uHighlighted > 0.5) {
        color += fresnel * vec3(0.08, 0.52, 0.58) + base * 0.12;
    }
    if (uAnalytical > 0.5) {
        color = mix(color, uBaseColor * (0.28 + key * 0.82 + fill * 0.28), 0.65);
    }

    color = vec3(1.0) - exp(-color * 1.12);
    fragmentColor = vec4(color, uOpacity);
}
"""


@dataclass(slots=True)
class _GpuMesh:
    part_id: BrainPartId
    vertices: np.ndarray
    center: np.ndarray
    vao: int
    vertex_buffer: int
    index_buffer: int
    index_count: int
    analysis_color: tuple[float, float, float]
    realistic_color: tuple[float, float, float]
    explosion: np.ndarray
    tissue: bool
    default_visible: bool
    view: str
    motion: str
    opacity: float
    element_key: str
    element_name: str
    deformation_center: np.ndarray


def _identity() -> np.ndarray:
    return np.identity(4, dtype=np.float32)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = _identity()
    matrix[:3, 3] = (x, y, z)
    return matrix


def _perspective(vertical_fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    factor = 1.0 / math.tan(vertical_fov * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = factor / max(aspect, 1e-6)
    matrix[1, 1] = factor
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _hex_color(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore[return-value]


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    area_normals = np.cross(second - first, third - first)
    normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(normals, faces[:, 0], area_normals)
    np.add.at(normals, faces[:, 1], area_normals)
    np.add.at(normals, faces[:, 2], area_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-9
    normals[valid] /= lengths[valid, None]
    normals[~valid, 2] = 1.0
    return normals


class _GpuOverlay(QWidget):
    """Transparent QWidget layer for gesture landmarks, Hover RBC/Neuron Microscope, and callouts."""

    def __init__(self, renderer: "GpuBrainWidget") -> None:
        super().__init__(renderer)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        renderer = self.parentWidget()
        if isinstance(renderer, GpuBrainWidget):
            renderer._paint_overlay(painter)
        painter.end()


class GpuBrainWidget(QOpenGLWidget):
    """Render and pick the 139 3D brain structures with Mouse Hover RBC & Synaptic Flow."""

    part_activated = Signal(object)
    part_hovered = Signal(object)
    fragment_pressed = Signal(object)
    fragment_moved = Signal(float, float, float)
    fragment_released = Signal()
    rotation_requested = Signal(float, float)
    zoom_requested = Signal(float)

    def __init__(self, meshes: tuple[AnatomicalMesh, ...], parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False)
        self._source_meshes = meshes
        self._gpu_meshes: list[_GpuMesh] = []
        self._program = 0
        self._uniforms: dict[str, int] = {}
        self._snapshot: BrainSnapshot | None = None
        self._last_mouse: QPointF | None = None
        self._press_position: QPointF | None = None
        self._drag_distance = 0.0
        self._dragging_fragment = False
        self._hovered_element: AnatomicalElement | None = None
        self._hover_mouse_pos: QPointF | None = None
        self._last_hover_pick_at = 0.0
        self._hover_clock = 0.0
        self._hand_points: tuple[tuple[float, float], ...] = ()
        self._pointer: tuple[float, float] | None = None
        self._gesture_label = ""
        self._last_overlay_observation = -1.0
        self._callout: BrainPart | None = None
        self._inspector: QWidget | None = None
        self._last_leader_anchor: QPointF | None = None
        self._last_leader_update = 0.0
        self._flow_overlay_state = None
        self._paint_frames = 0
        self._paint_sample_at = time.perf_counter()
        self._render_fps = 0.0

        elements = {
            mesh.element_key or mesh.part_id.value: AnatomicalElement(
                mesh.part_id, mesh.element_key or mesh.part_id.value, mesh.element_name
            )
            for mesh in meshes
        }
        self._pick_id_by_element = {key: index + 1 for index, key in enumerate(elements)}
        self._element_by_pick_id = {
            self._pick_id_by_element[key]: element for key, element in elements.items()
        }
        self._flow_renderer = NeuralFlowRenderer()
        self._overlay = _GpuOverlay(self)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()

    @property
    def render_fps(self) -> float:
        return self._render_fps

    def set_snapshot(self, snapshot: BrainSnapshot) -> None:
        previous = self._last_leader_anchor
        self._snapshot = snapshot
        self._hover_clock = (self._hover_clock + 0.018) % 100.0
        self.update()
        if self._hovered_element is not None or snapshot.flow_enabled:
            self._overlay.update()
        status = (snapshot.flow_enabled, circulation_visible(snapshot), snapshot.beating)
        if status != self._flow_overlay_state:
            self._flow_overlay_state = status
            self._overlay.update(8, 8, 760, 90)
        if self._callout is not None:
            current = self._selected_anchor()
            moved = (previous is None) != (current is None) or (
                previous is not None
                and current is not None
                and (previous - current).manhattanLength() >= 0.75
            )
            now = time.perf_counter()
            if moved and now - self._last_leader_update >= 1.0 / 30.0:
                self._overlay.update(
                    self._leader_bounds(previous).united(self._leader_bounds(current)).toAlignedRect()
                )
                self._last_leader_anchor = current
                self._last_leader_update = now

    def _leader_bounds(self, anchor: QPointF | None) -> QRectF:
        if anchor is None or self._inspector is None:
            return QRectF()
        card = QRectF(self._inspector.geometry())
        end = QPointF(
            card.left(),
            max(card.top() + 28.0, min(card.bottom() - 28.0, anchor.y() - 48.0)),
        )
        return QRectF(anchor, end).normalized().adjusted(-360.0, -52.0, 360.0, 12.0)

    def set_hand_overlay(
        self,
        observation: HandObservation,
        window_width: float,
        window_height: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        stale = observation.hands == 0 or time.monotonic() - observation.captured_at > 0.7
        if stale:
            if not self._hand_points and self._pointer is None:
                return
            self._hand_points = ()
            self._pointer = None
            self._gesture_label = ""
            self._last_overlay_observation = observation.captured_at
        else:
            if observation.captured_at == self._last_overlay_observation:
                return
            self._last_overlay_observation = observation.captured_at
            self._hand_points = tuple(
                (x * window_width - origin_x, y * window_height - origin_y)
                for x, y in observation.landmarks
            )
            self._pointer = (
                observation.pointer_x * window_width - origin_x,
                observation.pointer_y * window_height - origin_y,
            )
            self._gesture_label = observation.gesture.value.upper()
        self._overlay.update()

    def set_callout(self, part: BrainPart | None) -> None:
        self._callout = part
        self._last_leader_anchor = self._selected_anchor()
        self._overlay.update()

    def set_inspector(self, inspector: QWidget) -> None:
        self._inspector = inspector
        inspector.setParent(self)
        self._layout_inspector()

    def _layout_inspector(self) -> None:
        if self._inspector is None:
            return
        width = min(360, max(290, round(self.width() * 0.28)))
        height = min(640, max(1, self.height() - 32))
        self._inspector.setGeometry(
            self.width() - width - 16,
            (self.height() - height) // 2,
            width,
            height,
        )
        self._inspector.raise_()
        self._overlay.raise_()

    def initializeGL(self) -> None:  # noqa: N802
        self._program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )
        uniform_names = (
            "uModel",
            "uView",
            "uProjection",
            "uCorticalPulse",
            "uSubcorticalPulse",
            "uContractileGroup",
            "uIsTissue",
            "uBaseColor",
            "uCameraPosition",
            "uPickColor",
            "uRoughness",
            "uSheen",
            "uOpacity",
            "uHighlighted",
            "uHovered",
            "uAnalytical",
            "uPicking",
            "uMeshCenter",
            "uActivation",
            "uCycleProgress",
            "uBioelectricConduction",
            "uArrivalPhase",
            "uDeformCenter",
        )
        self._uniforms = {
            name: glGetUniformLocation(self._program, name) for name in uniform_names
        }
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDisable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._upload_meshes()
        self._flow_renderer.initialize()

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802
        ratio = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(width * ratio)), max(1, int(height * ratio)))
        self._overlay.setGeometry(0, 0, width, height)
        self._layout_inspector()
        self._overlay.raise_()

    def paintGL(self) -> None:  # noqa: N802
        glClearColor(0.003, 0.012, 0.025, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._program and self._snapshot is not None:
            glUseProgram(self._program)
            self._draw_scene(picking=False)
            projection, view, scale = self._scene_camera(self._snapshot)
            hovered_id = self._hovered_element.part_id if self._hovered_element else self._snapshot.selected_part
            self._flow_renderer.draw(
                self._snapshot,
                projection,
                view,
                scale,
                self.devicePixelRatioF(),
                hovered_part=hovered_id,
                hover_clock=self._hover_clock,
            )
            glBindVertexArray(0)
            glUseProgram(0)
        self._paint_frames += 1
        elapsed = time.perf_counter() - self._paint_sample_at
        if elapsed >= 1.0:
            self._render_fps = self._paint_frames / elapsed
            self._paint_frames = 0
            self._paint_sample_at = time.perf_counter()

    def pick_part(self, x: float, y: float) -> AnatomicalElement | None:
        if not self.isValid() or self._snapshot is None:
            return None
        if (
            self._inspector is not None
            and self._inspector.isVisible()
            and QRectF(self._inspector.geometry()).contains(QPointF(x, y))
        ):
            return None
        self.makeCurrent()
        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        fbo_format = QOpenGLFramebufferObjectFormat()
        fbo_format.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        fbo_format.setSamples(0)
        picking_fbo = QOpenGLFramebufferObject(QSize(width, height), fbo_format)
        if not picking_fbo.isValid() or not picking_fbo.bind():
            self.doneCurrent()
            return None
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)
        glDisable(GL_DITHER)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self._program)
        self._draw_scene(picking=True)
        pixel = glReadPixels(
            max(0, int(x * ratio)),
            max(0, int((self.height() - y - 1.0) * ratio)),
            1,
            1,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
        )
        glBindVertexArray(0)
        glUseProgram(0)
        picking_fbo.release()
        del picking_fbo
        glEnable(GL_DITHER)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        self.doneCurrent()
        self.update()
        values = np.frombuffer(pixel, dtype=np.uint8)
        if values.size < 3:
            return None
        identifier = int(values[0]) | (int(values[1]) << 8) | (int(values[2]) << 16)
        return self._element_by_pick_id.get(identifier)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._last_mouse = event.position()
        self._press_position = event.position()
        self._drag_distance = 0.0
        snapshot = self._snapshot
        if snapshot and snapshot.explosion >= 0.45:
            part = self.pick_part(event.position().x(), event.position().y())
            if part is not None:
                self._dragging_fragment = True
                self.fragment_pressed.emit(part)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            # Hover mode: probe the 3D structure under the cursor to reveal live red blood cells & neural impulses!
            now = time.perf_counter()
            if now - self._last_hover_pick_at >= 0.055:
                self._last_hover_pick_at = now
                picked = self.pick_part(pos.x(), pos.y())
                if picked != self._hovered_element:
                    self._hovered_element = picked
                    self.part_hovered.emit(picked)
                self._hover_mouse_pos = pos if picked is not None else None
                self._overlay.update()
            return

        if self._last_mouse is None:
            return
        delta = pos - self._last_mouse
        self._last_mouse = pos
        self._drag_distance += abs(delta.x()) + abs(delta.y())
        if self._dragging_fragment:
            self.fragment_moved.emit(
                delta.x() / max(1.0, self.width()),
                delta.y() / max(1.0, self.height()),
                0.0,
            )
        else:
            self.rotation_requested.emit(delta.x() * 0.009, delta.y() * 0.009)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_fragment = self._dragging_fragment
        self._dragging_fragment = False
        self.fragment_released.emit()
        if not was_fragment and self._drag_distance < 9.0:
            self.part_activated.emit(self.pick_part(event.position().x(), event.position().y()))
        self._last_mouse = None
        self._press_position = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        direction = 1.0 if event.angleDelta().y() > 0 else -1.0
        snapshot = self._snapshot
        if snapshot and snapshot.grabbed_fragment is not None:
            self.fragment_moved.emit(0.0, 0.0, direction * 0.04)
        else:
            self.zoom_requested.emit(1.10 if direction > 0 else 0.91)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.isValid():
            self.makeCurrent()
            for mesh in self._gpu_meshes:
                glDeleteBuffers(1, [mesh.vertex_buffer])
                glDeleteBuffers(1, [mesh.index_buffer])
                glDeleteVertexArrays(1, [mesh.vao])
            self._flow_renderer.dispose()
            self.doneCurrent()
        super().closeEvent(event)

    def _upload_meshes(self) -> None:
        grouped: dict[tuple, list[AnatomicalMesh]] = {}
        for source in self._source_meshes:
            key = (
                source.part_id,
                source.analysis_color,
                source.realistic_color,
                source.explosion_vector,
                source.tissue,
                source.default_visible,
                source.view,
                source.motion,
                source.opacity,
                source.element_key,
                source.deformation_center,
            )
            grouped.setdefault(key, []).append(source)

        for sources in grouped.values():
            source = sources[0]
            vertex_chunks: list[np.ndarray] = []
            normal_chunks: list[np.ndarray] = []
            index_chunks: list[np.ndarray] = []
            vertex_offset = 0
            for component in sources:
                component_vertices = np.asarray(
                    [(v.x, v.y, v.z) for v in component.vertices], dtype=np.float32
                )
                component_faces = np.asarray(component.faces, dtype=np.uint32)[:, (0, 2, 1)].copy()
                vertex_chunks.append(component_vertices)
                normal_chunks.append(_vertex_normals(component_vertices, component_faces.astype(np.int32)))
                index_chunks.append(component_faces.reshape(-1) + vertex_offset)
                vertex_offset += len(component_vertices)
            vertices = np.concatenate(vertex_chunks, axis=0)
            normals = np.concatenate(normal_chunks, axis=0)
            interleaved = np.column_stack((vertices, normals)).astype(np.float32)
            indices = np.concatenate(index_chunks).astype(np.uint32, copy=False)
            vao = int(glGenVertexArrays(1))
            vertex_buffer = int(glGenBuffers(1))
            index_buffer = int(glGenBuffers(1))
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            stride = 6 * np.dtype(np.float32).itemsize
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * np.dtype(np.float32).itemsize)
            )
            self._gpu_meshes.append(
                _GpuMesh(
                    source.part_id,
                    vertices,
                    (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5,
                    vao,
                    vertex_buffer,
                    index_buffer,
                    int(indices.size),
                    _hex_color(source.analysis_color),
                    _hex_color(source.realistic_color),
                    np.asarray(
                        (source.explosion_vector.x, source.explosion_vector.y, source.explosion_vector.z),
                        dtype=np.float32,
                    ),
                    source.tissue,
                    source.default_visible,
                    source.view,
                    source.motion,
                    source.opacity,
                    source.element_key or source.part_id.value,
                    source.element_name,
                    np.asarray(
                        (
                            source.deformation_center.x,
                            source.deformation_center.y,
                            source.deformation_center.z,
                        )
                        if source.deformation_center
                        else (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5,
                        dtype=np.float32,
                    ),
                )
            )
        glBindVertexArray(0)

    def _draw_scene(self, picking: bool) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        ratio = self.devicePixelRatioF()
        glViewport(0, 0, max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        projection, view, scale = self._scene_camera(snapshot)
        glUniformMatrix4fv(self._uniforms["uView"], 1, GL_TRUE, view)
        glUniformMatrix4fv(self._uniforms["uProjection"], 1, GL_TRUE, projection)
        glUniform3f(self._uniforms["uCameraPosition"], 0.0, 0.08, 6.0)
        glUniform1i(self._uniforms["uPicking"], 1 if picking else 0)
        glUniform1f(self._uniforms["uCorticalPulse"], snapshot.cycle.cortical_depolarization)
        glUniform1f(self._uniforms["uSubcorticalPulse"], snapshot.cycle.subcortical_modulation)
        glUniform1f(self._uniforms["uCycleProgress"], snapshot.cycle.progress)
        self._draw_mesh_group(snapshot, scale, picking)

    def _scene_camera(self, snapshot: BrainSnapshot) -> tuple[np.ndarray, np.ndarray, float]:
        projection = _perspective(math.radians(33.0), self.width() / max(self.height(), 1), 0.1, 40.0)
        projection[0, 2] = 0.0 if snapshot.spatial_mode else 0.24
        view = _translation(0.0, -0.04, -6.0)
        aspect = self.width() / max(self.height(), 1)
        fit = min(1.0, aspect / 0.90)
        scale = 1.18 * snapshot.zoom * fit / (1.0 + snapshot.explosion * 0.42)
        return projection, view, scale

    def _selected_anchor(self) -> QPointF | None:
        snapshot = self._snapshot
        if snapshot is None or self._callout is None:
            return None
        candidates = [
            mesh
            for mesh in self._gpu_meshes
            if mesh.part_id == self._callout.identifier
            and visible_mesh(mesh, snapshot)
            and (snapshot.selected_element is None or mesh.element_key == snapshot.selected_element.key)
        ]
        if not candidates:
            return None
        mesh = max(candidates, key=lambda item: item.index_count)
        projection, view, scale = self._scene_camera(snapshot)
        model = placement_matrix(snapshot, mesh.explosion, mesh.part_id, scale, mesh.element_key)
        clip = projection @ view @ model @ np.append(mesh.center, 1.0)
        if clip[3] <= 0:
            return None
        point = clip[:3] / clip[3]
        if not -1.0 <= point[2] <= 1.0:
            return None
        return QPointF(
            float((point[0] + 1.0) * self.width() * 0.5),
            float((1.0 - point[1]) * self.height() * 0.5),
        )

    def _draw_mesh_group(self, snapshot: BrainSnapshot, scale: float, picking: bool) -> None:
        flowing = circulation_visible(snapshot)
        hovered_id = self._hovered_element.part_id if self._hovered_element else None

        def opacity_of(mesh: _GpuMesh) -> float:
            system = anatomy_system_for(mesh.part_id)
            if flowing and system in {
                AnatomySystem.CEREBRAL_CORTEX_LOBES,
                AnatomySystem.MENINGES_AND_VENTRICLES,
            }:
                return 0.26
            # When hovering over a cortical region with the mouse, make that parcel semi-translucent
            # so you see the underlying red blood cells (glóbulos rojos) & synaptic impulses moving!
            if not picking and hovered_id == mesh.part_id and snapshot.explosion < 0.05:
                if system in {
                    AnatomySystem.CEREBRAL_CORTEX_LOBES,
                    AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
                }:
                    return 0.42
            return mesh.opacity

        for mesh in sorted(self._gpu_meshes, key=lambda item: opacity_of(item) < 1.0):
            if not visible_mesh(mesh, snapshot):
                continue
            model = placement_matrix(snapshot, mesh.explosion, mesh.part_id, scale, mesh.element_key)
            glUniformMatrix4fv(self._uniforms["uModel"], 1, GL_TRUE, model)
            glUniform1f(self._uniforms["uIsTissue"], 1.0 if mesh.tissue else 0.0)
            system = anatomy_system_for(mesh.part_id)
            group = 1.0 if system in {
                AnatomySystem.CEREBRAL_CORTEX_LOBES,
                AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
            } else 2.0
            glUniform1f(self._uniforms["uContractileGroup"], group)
            opacity = opacity_of(mesh)
            glDepthMask(GL_TRUE if picking or opacity >= 1.0 else GL_FALSE)
            glUniform1f(self._uniforms["uOpacity"], 1.0 if picking else opacity)
            glUniform1f(
                self._uniforms["uHighlighted"],
                float(
                    snapshot.selected_part == mesh.part_id
                    and (snapshot.selected_element is None or snapshot.selected_element.key == mesh.element_key)
                ),
            )
            glUniform1f(
                self._uniforms["uHovered"],
                float(hovered_id == mesh.part_id and not picking),
            )
            glUniform1f(self._uniforms["uAnalytical"], 0.0 if snapshot.realistic else 1.0)

            conduction_active = (snapshot.beating or hovered_id == mesh.part_id) and system in {
                AnatomySystem.WHITE_MATTER_CONNECTOME,
                AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS,
                AnatomySystem.LIMBIC_AND_MEMORY_SYSTEM,
            }
            glUniform1f(self._uniforms["uBioelectricConduction"], float(conduction_active))
            arrival_map = {
                AnatomySystem.ELOQUENT_SPEECH_AND_MOTOR_AREAS: 0.25,
                AnatomySystem.LIMBIC_AND_MEMORY_SYSTEM: 0.42,
                AnatomySystem.WHITE_MATTER_CONNECTOME: 0.58,
            }
            glUniform1f(self._uniforms["uArrivalPhase"], arrival_map.get(system, 0.15))
            glUniform1f(self._uniforms["uActivation"], self._activation(mesh.part_id, snapshot))
            color = mesh.realistic_color if snapshot.realistic else mesh.analysis_color
            glUniform3f(self._uniforms["uBaseColor"], *color)
            glUniform3f(self._uniforms["uMeshCenter"], *mesh.center)
            glUniform3f(self._uniforms["uDeformCenter"], *mesh.deformation_center)
            roughness = 0.44 if mesh.tissue else 0.28
            sheen = 0.64 if mesh.tissue else 0.36
            glUniform1f(self._uniforms["uRoughness"], roughness)
            glUniform1f(self._uniforms["uSheen"], sheen)
            pick_id = self._pick_id_by_element[mesh.element_key]
            glUniform3f(
                self._uniforms["uPickColor"],
                (pick_id & 255) / 255.0,
                ((pick_id >> 8) & 255) / 255.0,
                ((pick_id >> 16) & 255) / 255.0,
            )
            glBindVertexArray(mesh.vao)
            glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, None)
        glDepthMask(GL_TRUE)

    @staticmethod
    def _activation(part: BrainPartId, snapshot: BrainSnapshot) -> float:
        if not snapshot.realistic:
            return 0.0
        phase = snapshot.cycle.phase
        system = anatomy_system_for(part)
        if phase == BrainActivityPhase.SENSORY_TRANSDUCTION and part in {
            BrainPartId.THALAMUS_L,
            BrainPartId.THALAMUS_R,
            BrainPartId.PRIMARY_SOMATOSENSORY_S1_L,
            BrainPartId.PRIMARY_SOMATOSENSORY_S1_R,
            BrainPartId.PRIMARY_VISUAL_CALCARINE_V1_L,
            BrainPartId.PRIMARY_VISUAL_CALCARINE_V1_R,
            BrainPartId.PRIMARY_AUDITORY_HESCHL_BI,
        }:
            return 0.42
        if phase == BrainActivityPhase.WERNICKE_COMPREHENSION and part in {
            BrainPartId.WERNICKE_AREA_L,
            BrainPartId.SUPERIOR_TEMPORAL_GYRUS_L,
            BrainPartId.MIDDLE_TEMPORAL_GYRUS_L,
            BrainPartId.GESCHWIND_TERRITORY_L,
        }:
            return 0.48
        if phase == BrainActivityPhase.LIMBIC_HIPPOCAMPAL_INTEGRATION and system == AnatomySystem.LIMBIC_AND_MEMORY_SYSTEM:
            return 0.40
        if phase == BrainActivityPhase.ARCUATE_BROCA_SYNTHESIS and part in {
            BrainPartId.ARCUATE_FASCICULUS_L,
            BrainPartId.BROCA_PARS_OPERCULARIS_L,
            BrainPartId.BROCA_PARS_TRIANGULARIS_L,
            BrainPartId.INSULAR_CORTEX_L,
        }:
            return 0.52
        if phase == BrainActivityPhase.MOTOR_PLANNING_BASAL_CEREBELLAR and system in {
            AnatomySystem.BASAL_GANGLIA_AND_DIENCEPHALON,
            AnatomySystem.BRAINSTEM_CEREBELLUM_CRANIAL_NERVES,
        }:
            return 0.36
        if phase == BrainActivityPhase.CORTICOSPINAL_EXECUTION and part in {
            BrainPartId.PRIMARY_MOTOR_CORTEX_M1_L,
            BrainPartId.PRIMARY_MOTOR_CORTEX_M1_R,
            BrainPartId.INTERNAL_CAPSULE_CORTICOSPINAL_L,
            BrainPartId.INTERNAL_CAPSULE_CORTICOSPINAL_R,
            BrainPartId.MEDULLA_OBLONGATA_PYRAMIDS,
        }:
            return 0.52
        return 0.06

    def _paint_overlay(self, painter: QPainter) -> None:
        state = self._snapshot
        # 1. Top status legend when flow or hover is active
        if state and (state.flow_enabled or self._hovered_element is not None):
            painter.save()
            font = painter.font()
            font.setFamily("Segoe UI")
            font.setPointSize(9)
            painter.setFont(font)
            painter.setPen(QColor("#f54242"))
            painter.drawText(QPointF(22, 28), "●  GLÓBULOS ROJOS (ERITROCITOS ARTERIALES)")
            painter.setPen(QColor("#2eedff"))
            painter.drawText(QPointF(305, 28), "●  IMPULSOS ELÉCTRICOS SINÁPTICOS (Na⁺/K⁺)")
            painter.setPen(QColor("#c26bff"))
            painter.drawText(QPointF(595, 28), "●  MODULACIÓN GABA")
            painter.setPen(QColor("#a8bec9"))
            if self._hovered_element is not None:
                message = (
                    f"Microscopio Hover activo en: {self._hovered_element.name} · "
                    f"Glóbulos rojos y descarga neuronal en vivo (EEG {state.cycle.dominant_eeg_hz:.0f} Hz)"
                )
            elif not circulation_visible(state):
                message = "Flujo en pausa · reconstruye las 139 estructuras para conectar axones y arterias"
            else:
                message = f"Fase activa: {state.cycle.phase.value} · Ritmo EEG dominante: {state.cycle.dominant_eeg_hz:.0f} Hz"
            painter.drawText(QPointF(22, 50), message)
            painter.restore()

        # 2. Live Cellular & Bioelectric Hover Microscope Card (shows animated Red Blood Cells & Firing Neurons!)
        target_element = self._hovered_element
        if target_element is not None and state is not None:
            self._paint_hover_microscope(painter, target_element, state)

        # 3. Selected structure leader callout
        anchor = self._selected_anchor()
        if anchor is not None and self._inspector is not None and self._inspector.isVisible():
            card = QRectF(self._inspector.geometry())
            if not card.contains(anchor):
                endpoint = QPointF(
                    card.left(),
                    max(card.top() + 28.0, min(card.bottom() - 28.0, anchor.y() - 48.0)),
                )
                elbow = QPointF(endpoint.x() - 26.0, endpoint.y())
                painter.setPen(QPen(QColor(32, 220, 234, 190), 1.3, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(anchor, elbow)
                painter.drawLine(elbow, endpoint)
                painter.setPen(QPen(QColor("#20dcea"), 1.3))
                painter.drawEllipse(anchor, 5.0, 5.0)
                painter.setBrush(QColor("#20dcea"))
                painter.drawEllipse(anchor, 1.8, 1.8)
                painter.save()
                font = painter.font()
                font.setFamily("Segoe UI")
                font.setPointSize(9)
                painter.setFont(font)
                text = QFontMetrics(font).elidedText(
                    self._callout.display_name, Qt.TextElideMode.ElideRight, 320
                )
                width = QFontMetrics(font).horizontalAdvance(text) + 22
                x = anchor.x() + 12
                if x + width > card.left() - 8:
                    x = anchor.x() - width - 12
                label = QRectF(max(8, x), max(64, anchor.y() - 36), width, 26)
                painter.setPen(QPen(QColor("#248c9a"), 1))
                painter.setBrush(QColor(2, 12, 20, 236))
                painter.drawRoundedRect(label, 5, 5)
                painter.setPen(QColor("#c6f4f1"))
                painter.drawText(label.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, text)
                painter.restore()

        # 4. Hand tracking landmarks
        pen = QPen(QColor("#16889a"), 1.2)
        painter.setPen(pen)
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        )
        for start in range(0, len(self._hand_points), 21):
            hand = self._hand_points[start : start + 21]
            if len(hand) < 21:
                continue
            for first, second in connections:
                painter.drawLine(QPointF(*hand[first]), QPointF(*hand[second]))
            painter.setBrush(QColor("#75d5df"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in hand:
                painter.drawEllipse(QPointF(*point), 2.0, 2.0)
            painter.setPen(pen)
        if self._pointer is not None:
            pointer_color = QColor(
                "#30f29b" if self._snapshot and self._snapshot.grabbed_fragment else "#23d9e8"
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(pointer_color, 2.0))
            painter.drawEllipse(QPointF(*self._pointer), 12.0, 12.0)
            painter.setPen(pointer_color)
            painter.drawText(
                QPointF(self._pointer[0] + 17.0, self._pointer[1] - 14.0), self._gesture_label
            )

    def _paint_hover_microscope(
        self, painter: QPainter, element: AnatomicalElement, state: BrainSnapshot
    ) -> None:
        """Render a live cellular & synaptic zoom lens showing moving Red Blood Cells and Firing Neurons."""
        painter.save()
        box = QRectF(18.0, self.height() - 154.0, 330.0, 136.0)
        painter.setPen(QPen(QColor("#1b6d82"), 1.3))
        painter.setBrush(QColor(3, 12, 20, 232))
        painter.drawRoundedRect(box, 9.0, 9.0)

        font = painter.font()
        font.setFamily("Cascadia Mono")
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#22d3ee"))
        painter.drawText(box.adjusted(12, 8, -12, -104), Qt.AlignmentFlag.AlignLeft, "MICROSCOPIO NEUROVASCULAR Y SINÁPTICO EN VIVO")

        font.setBold(False)
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor("#e2f1f5"))
        elided_name = QFontMetrics(font).elidedText(element.name, Qt.TextElideMode.ElideRight, 304)
        painter.drawText(box.adjusted(12, 24, -12, -86), Qt.AlignmentFlag.AlignLeft, elided_name)

        # Top microvessel channel: Animated Biconcave Red Blood Cells (Glóbulos Rojos)
        vessel_rect = QRectF(box.left() + 12, box.top() + 48, 306, 32)
        painter.setPen(QPen(QColor("#6e2226"), 1.2))
        painter.setBrush(QColor(46, 10, 14, 210))
        painter.drawRoundedRect(vessel_rect, 14.0, 14.0)

        t = self._hover_clock * 1.4
        for idx in range(7):
            prog = ((t + idx / 7.0) % 1.0)
            rx = vessel_rect.left() + 16.0 + prog * (vessel_rect.width() - 32.0)
            ry = vessel_rect.center().y() + math.sin(prog * math.tau * 2.0 + idx) * 4.5
            # Outer erythrocyte disc
            painter.setPen(QPen(QColor("#ff7575"), 1.0))
            painter.setBrush(QColor("#d92626"))
            painter.drawEllipse(QPointF(rx, ry), 9.0, 6.2)
            # Inner biconcave dimple (central pallor)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#8f1518"))
            painter.drawEllipse(QPointF(rx, ry), 4.2, 2.8)

        # Bottom axonal channel: Pyramidal Neurons & Propagating Bioelectric Action Potentials
        axon_y = box.top() + 106.0
        painter.setPen(QPen(QColor("#1e5b6e"), 2.2))
        painter.drawLine(QPointF(box.left() + 26, axon_y), QPointF(box.right() - 26, axon_y))
        # Soma & dendrites at left and right
        for soma_x in (box.left() + 28, box.left() + 168, box.right() - 28):
            painter.setPen(QPen(QColor("#29b6f6"), 1.2))
            painter.setBrush(QColor("#0a3446"))
            painter.drawEllipse(QPointF(soma_x, axon_y), 7.5, 7.5)
            for ang in (-0.8, -0.3, 0.3, 0.8):
                dx = math.cos(ang) * 14.0
                dy = math.sin(ang) * 12.0
                painter.drawLine(QPointF(soma_x, axon_y), QPointF(soma_x - dx, axon_y + dy))

        # Traveling action potential spikes along the axon
        for s_idx in range(4):
            s_prog = ((t * 1.6 + s_idx * 0.25) % 1.0)
            sx = box.left() + 28.0 + s_prog * (box.width() - 56.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(34, 235, 255, 95))
            painter.drawEllipse(QPointF(sx, axon_y), 11.0, 6.5)
            painter.setBrush(QColor("#fff176"))
            painter.drawEllipse(QPointF(sx, axon_y), 4.5, 4.5)

        painter.restore()


GpuHeartWidget = GpuBrainWidget

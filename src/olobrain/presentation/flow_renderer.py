"""Hardware-accelerated GPU renderer for 3D Red Blood Cells (Erythrocytes) and Synaptic Action Potentials.

Supports two interactive modes:
1. Global Flow Mode (F): Renders all active cerebrovascular RBC streams and axonal action-potential wave-packets.
2. Mouse Hover Microscope Mode: Automatically illuminates and animates the red blood cells and electrical impulses
   belonging to the brain structure or vessel currently under the user's mouse cursor!
"""

from __future__ import annotations

import ctypes
import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from olobrain.domain.model import BrainPartId, anatomy_system_for
from olobrain.infrastructure.circulation_geometry import build_flow_routes, circulation_visible
from olobrain.presentation.scene_layout import placement_matrix

VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
layout(location=2) in float sizeScale;
uniform mat4 mvp;
uniform float pixelRatio;
out vec3 tint;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    gl_PointSize = sizeScale * pixelRatio;
    tint = color;
}
"""

FRAGMENT = """#version 330 core
in vec3 tint;
uniform int primitiveMode;
out vec4 frag;
void main() {
    if (primitiveMode == 0) {
        frag = vec4(tint, 0.72);
        return;
    }
    vec2 uv = (gl_PointCoord - vec2(0.5)) * 2.0;
    float r = length(uv);
    if (r > 1.0) discard;

    if (primitiveMode == 1) {
        // 3D Biconcave Red Blood Cell (Erythrocyte) shader with central pallor and rim lighting
        float rim = smoothstep(0.25, 0.72, r) * (1.0 - smoothstep(0.82, 1.0, r));
        float centerDimple = 0.52 + 0.48 * rim;
        vec3 n = vec3(uv * 0.85, sqrt(max(0.0, 1.0 - dot(uv, uv))));
        float spec = pow(max(0.0, dot(n, normalize(vec3(-0.4, 0.6, 0.7)))), 16.0);
        vec3 rbcColor = tint * centerDimple + vec3(1.0, 0.65, 0.60) * spec * 0.45;
        float alpha = 1.0 - smoothstep(0.84, 1.0, r);
        frag = vec4(rbcColor, alpha);
    } else {
        // Bioluminescent Synaptic Action Potential Spike with glowing core & corona
        float core = exp(-r * r * 4.2);
        float halo = 1.0 - smoothstep(0.0, 1.0, r);
        vec3 glow = mix(tint, vec3(1.0, 0.98, 0.86), core * 0.85);
        frag = vec4(glow, halo * 0.95);
    }
}
"""


class NeuralFlowRenderer:
    """Render 3D biconcave red blood cells and glowing axonal electrical impulses."""

    def __init__(self) -> None:
        self.routes = build_flow_routes()
        self.program = 0
        self.vao = 0
        self.vbo = 0
        self.uniforms: dict[str, int] = {}
        self._cached_geometry: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._rbc_offsets = np.linspace(0.0, 1.0, 14, endpoint=False)
        self._spike_offsets = np.linspace(0.0, 1.0, 10, endpoint=False)

        for route in self.routes:
            if route.kind == "neural":
                color = np.asarray(
                    (0.15, 0.92, 1.0) if route.oxygenated else (0.78, 0.44, 1.0),
                    dtype=np.float32,
                )
            else:
                color = np.asarray(
                    (0.94, 0.16, 0.18) if route.oxygenated else (0.24, 0.56, 0.96),
                    dtype=np.float32,
                )
            points = route.points
            ends = points[::3]
            line_positions = np.stack((ends[:-1], ends[1:]), axis=1).reshape(-1, 3)
            line_scales = np.ones((len(line_positions), 1), dtype=np.float32)
            line = np.column_stack((line_positions, np.tile(color * 0.42, (len(line_positions), 1)), line_scales))

            arrows = []
            for sample in (30, 68):
                tip = points[sample]
                tangent = points[sample + 1] - points[sample - 1]
                tangent /= max(float(np.linalg.norm(tangent)), 1e-5)
                side = np.cross(tangent, (0.0, 0.0, 1.0))
                if np.linalg.norm(side) < 0.1:
                    side = np.cross(tangent, (0.0, 1.0, 0.0))
                side /= max(float(np.linalg.norm(side)), 1e-5)
                tail = tip - tangent * 0.042
                arrows.extend((tip, tail + side * 0.016, tail - side * 0.016))
            arrow_scales = np.ones((len(arrows), 1), dtype=np.float32)
            arrow = np.column_stack((arrows, np.tile(color * 0.88, (len(arrows), 1)), arrow_scales))
            dim = arrow.copy()
            dim[:, 3:6] *= 0.32
            self._cached_geometry[route.key] = (
                color,
                line.astype(np.float32),
                arrow.astype(np.float32),
                dim.astype(np.float32),
            )

    def initialize(self) -> None:
        self.program = compileProgram(
            compileShader(VERTEX, GL_VERTEX_SHADER),
            compileShader(FRAGMENT, GL_FRAGMENT_SHADER),
        )
        self.uniforms = {
            name: glGetUniformLocation(self.program, name)
            for name in ("mvp", "pixelRatio", "primitiveMode")
        }
        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 7 * np.dtype(np.float32).itemsize
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glBindVertexArray(0)

    def draw(
        self,
        state,
        projection: np.ndarray,
        view: np.ndarray,
        scale: float,
        pixel_ratio: float,
        hovered_part: BrainPartId | None = None,
        hover_clock: float = 0.0,
    ) -> None:
        global_flow = circulation_visible(state)
        if not self.program or (not global_flow and hovered_part is None):
            return
        if state.explosion >= 0.035 or state.fragment_offsets:
            return

        hovered_sys = anatomy_system_for(hovered_part) if hovered_part is not None else None
        lines, arrows, rbc_particles, spike_particles = [], [], [], []

        for route in self.routes:
            if any(anatomy_system_for(part) not in state.visible_systems for part in route.parts):
                continue
            # Check if this route is activated by mouse hover or by global flow
            route_matches_hover = hovered_part is not None and (
                hovered_part in route.parts
                or any(anatomy_system_for(p) == hovered_sys for p in route.parts)
            )
            if not global_flow and not route_matches_hover:
                continue

            color, line, arrow, dim = self._cached_geometry[route.key]
            points = route.points
            lines.append(line)
            active = True if route_matches_hover else route.active(state)
            arrows.append(arrow if active else dim)

            if active:
                base_clock = state.flow_times[route.gate] if global_flow else hover_clock
                if route.kind == "vascular":
                    phase = (base_clock * 0.62) % 1.0
                    samples = ((phase + self._rbc_offsets) % 1.0) * 99.0
                    indices = samples.astype(np.int32)
                    weight = (samples - indices)[:, None]
                    positions = points[indices] * (1.0 - weight) + points[np.minimum(indices + 1, 99)] * weight
                    sizes = np.full((len(self._rbc_offsets), 1), 10.5 if route_matches_hover else 8.5, dtype=np.float32)
                    rbc_particles.append(
                        np.column_stack((positions, np.tile(color, (len(self._rbc_offsets), 1)), sizes))
                    )
                else:
                    phase = (base_clock * 1.05) % 1.0
                    samples = ((phase + self._spike_offsets) % 1.0) * 99.0
                    indices = samples.astype(np.int32)
                    weight = (samples - indices)[:, None]
                    positions = points[indices] * (1.0 - weight) + points[np.minimum(indices + 1, 99)] * weight
                    # Add trailing axonal action potential wave-packet points behind each spike head
                    for trail_step, (trail_offset, size_px, brightness) in enumerate(
                        ((0.0, 11.5, 1.0), (-2.2, 8.0, 0.75), (-4.4, 5.5, 0.48))
                    ):
                        t_samples = np.clip(samples + trail_offset, 0.0, 99.0)
                        t_idx = t_samples.astype(np.int32)
                        t_w = (t_samples - t_idx)[:, None]
                        t_pos = points[t_idx] * (1.0 - t_w) + points[np.minimum(t_idx + 1, 99)] * t_w
                        t_col = np.tile(color * brightness, (len(self._spike_offsets), 1))
                        if trail_step == 0:
                            t_col[::2] = np.asarray((1.0, 0.92, 0.42), dtype=np.float32)
                        t_sizes = np.full((len(self._spike_offsets), 1), size_px, dtype=np.float32)
                        spike_particles.append(np.column_stack((t_pos, t_col, t_sizes)))

        model = placement_matrix(state, (0.0, 0.0, 0.0), self.routes[0].parts[0], scale)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.uniforms["mvp"], 1, GL_TRUE, projection @ view @ model)
        glUniform1f(self.uniforms["pixelRatio"], pixel_ratio)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for primitive, data, mode in (
            (GL_LINES, lines, 0),
            (GL_TRIANGLES, arrows, 0),
            (GL_POINTS, rbc_particles, 1),
            (GL_POINTS, spike_particles, 2),
        ):
            if not data:
                continue
            array = np.concatenate(data).astype(np.float32, copy=False)
            glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_DYNAMIC_DRAW)
            glUniform1i(self.uniforms["primitiveMode"], mode)
            glDrawArrays(primitive, 0, len(array))
        glBindVertexArray(0)

    def dispose(self) -> None:
        if self.program:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
            glDeleteProgram(self.program)
            self.program = 0


BloodFlowRenderer = NeuralFlowRenderer

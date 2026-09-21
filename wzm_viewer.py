from __future__ import annotations

"""Interactive SUN WZM and EWZ viewer using pyglet + ModernGL.

Controls: left drag rotates, wheel zooms, T textures, E EWZ effects,
G preview glow, L lighting, W wireframe, R reset, Esc closes.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import moderngl
import numpy as np
import pyglet
from PIL import Image
from pyglet.window import key, mouse

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from wzm_unpack import WZMError, WZMModel, _bone_rotation, inspect_wzu_materials, parse_wzm
from ewz_inspect import EffectCatalog


VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_tangent;
in vec2 in_uv;
out vec3 v_position;
out vec3 v_normal;
out vec3 v_tangent;
out vec2 v_uv;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_position = world.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_tangent = normalize(mat3(u_model) * in_tangent);
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_specular_texture;
uniform bool u_use_texture;
uniform bool u_use_normal_texture;
uniform bool u_use_specular_texture;
uniform bool u_use_lighting;
uniform bool u_preview_effect;
uniform float u_time;
uniform vec3 u_base_color;
in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 texel = u_use_texture ? texture(u_texture, v_uv) : vec4(u_base_color, 1.0);
    if (texel.a < 0.04) discard;
    vec3 normal = normalize(v_normal);
    if (u_use_normal_texture) {
        vec3 tangent = normalize(v_tangent - normal * dot(normal, v_tangent));
        vec3 bitangent = normalize(cross(normal, tangent));
        vec3 sampledNormal = texture(u_normal_texture, v_uv).xyz * 2.0 - 1.0;
        normal = normalize(tangent * sampledNormal.x + bitangent * sampledNormal.y + normal * sampledNormal.z);
    }
    vec3 viewDir = normalize(vec3(0.0, 0.0, 3.2) - v_position);
    vec3 lightDir = normalize(vec3(-0.4, 0.8, 1.0));
    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(reflect(-lightDir, normal), viewDir), 0.0), 28.0);
    vec3 color = texel.rgb;
    if (u_use_lighting) color *= (0.34 + 0.78 * diffuse);
    if (u_use_lighting) color += vec3(0.32) * specular;
    if (u_use_specular_texture) {
        float mask = dot(texture(u_specular_texture, v_uv).rgb, vec3(0.3333));
        color += vec3(0.45) * specular * mask;
    }
    if (u_preview_effect) {
        float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.2);
        float pulse = 0.76 + 0.24 * sin(u_time * 2.4);
        color += vec3(0.18, 0.48, 1.0) * rim * pulse;
    }
    fragColor = vec4(color, texel.a);
}
"""


EFFECT_VERTEX_SHADER = """
#version 330
uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform float u_time_ms;
uniform float u_duration_ms;
uniform vec3 u_velocity;
uniform vec3 u_acceleration;
uniform vec2 u_scale_start;
uniform vec2 u_scale_end;
uniform vec4 u_color_start;
uniform vec4 u_color_end;
uniform int u_frame_count;
uniform int u_frame_columns;
uniform int u_frame_rows;
uniform float u_frame_ms;
in vec2 in_corner;
in vec2 in_uv;
in vec3 in_spawn;
in float in_birth_ms;
in float in_base_size;
out vec2 v_uv;
out vec4 v_color;
void main() {
    float duration = max(u_duration_ms, 1.0);
    float age_ms = mod(max(u_time_ms - in_birth_ms, 0.0), duration);
    float life = clamp(age_ms / duration, 0.0, 1.0);
    float seconds = age_ms * 0.001;
    vec3 center = in_spawn + u_velocity * seconds + 0.5 * u_acceleration * seconds * seconds;
    vec2 size = max(mix(u_scale_start, u_scale_end, life), vec2(0.001)) * in_base_size;
    vec4 view_position = u_modelview * vec4(center, 1.0);
    view_position.xy += in_corner * size;
    gl_Position = u_projection * view_position;
    v_color = mix(u_color_start, u_color_end, life);
    v_uv = in_uv;
    if (u_frame_count > 1) {
        int frame = int(floor(age_ms / max(u_frame_ms, 1.0))) % u_frame_count;
        int column = frame % max(u_frame_columns, 1);
        int row = frame / max(u_frame_columns, 1);
        v_uv = (in_uv + vec2(float(column), float(row))) /
            vec2(float(max(u_frame_columns, 1)), float(max(u_frame_rows, 1)));
    }
}
"""


EFFECT_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
in vec4 v_color;
out vec4 fragColor;
void main() {
    vec4 texel = texture(u_texture, v_uv);
    float alpha = texel.a * clamp(v_color.a, 0.0, 1.0);
    if (alpha < 0.015) discard;
    fragColor = vec4(texel.rgb * max(v_color.rgb, vec3(0.0)), alpha);
}
"""


def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype="f4")


def translation(z: float) -> np.ndarray:
    matrix = np.eye(4, dtype="f4")
    matrix[2, 3] = z
    return matrix


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype="f4")


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype="f4")


def scale_matrix(value: float) -> np.ndarray:
    return np.diag([value, value, value, 1.0]).astype("f4")


def effect_vector(values) -> np.ndarray:
    x, y, z = (float(value) for value in values)
    return np.asarray((x, z, -y), dtype="f4")


def curve_endpoints(change: dict, index: int, default: float) -> tuple[float, float]:
    channels = change.get("channels", [])
    if index >= len(channels) or not channels[index].get("key_count"):
        return default, default
    channel = channels[index]
    first = float(channel.get("first", {}).get("value", default))
    last = float(channel.get("last", {}).get("value", first))
    return first, last


def alpha_value(value: float) -> float:
    if value <= 0.001:
        return 0.0
    if value <= 4.0:
        return min(value, 1.0)
    return min(value / 255.0, 1.0)


def low_discrepancy(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    current = index + 1
    while current:
        value += factor * (current % base)
        current //= base
        factor /= base
    return value


def resolve_texture(directory: Path, texture_name: str) -> Path | None:
    target = Path(texture_name).name.casefold()
    for path in directory.iterdir():
        if path.is_file() and path.name.casefold() == target:
            return path
    return None


class RenderMesh:
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, vertices: np.ndarray, indices: np.ndarray, texture_path: Path | None, normal_path: Path | None = None, specular_path: Path | None = None):
        self.vertex_buffer = ctx.buffer(vertices.astype("f4").tobytes())
        self.index_buffer = ctx.buffer(indices.astype("u4").tobytes())
        self.vao = ctx.vertex_array(
            program,
            [(self.vertex_buffer, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")],
            self.index_buffer,
            index_element_size=4,
        )
        self.texture = self._load_texture(ctx, texture_path, (145, 162, 190, 255))
        self.normal_texture = self._load_texture(ctx, normal_path, (128, 128, 255, 255))
        self.specular_texture = self._load_texture(ctx, specular_path, (0, 0, 0, 255))
        self.has_texture = texture_path is not None
        self.has_normal_texture = normal_path is not None
        self.has_specular_texture = specular_path is not None

    @staticmethod
    def _load_texture(ctx: moderngl.Context, path: Path | None, fallback: tuple[int, int, int, int]):
        if path and path.is_file():
            with Image.open(path) as image:
                image = image.convert("RGBA")
                texture = ctx.texture(image.size, 4, image.tobytes())
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.repeat_x = True
            texture.repeat_y = True
            return texture
        return ctx.texture((1, 1), 4, bytes(fallback))

    def release(self) -> None:
        self.vao.release()
        self.vertex_buffer.release()
        self.index_buffer.release()
        self.texture.release()
        self.normal_texture.release()
        self.specular_texture.release()


class EffectLayer:
    def __init__(
        self,
        ctx: moderngl.Context,
        program: moderngl.Program,
        element: dict,
        create: dict,
        move: dict,
        change: dict,
        visual: dict,
        texture_path: Path,
        spawn_offset: np.ndarray | None = None,
        size_multiplier: float = 1.0,
    ):
        self.element = element
        self.create = create
        self.move = move
        self.change = change
        self.visual = visual
        self.duration_ms = max(float(create.get("duration_ms", 1000)), 1.0)
        self.additive = int(visual.get("render_type", 0)) in (2, 7)
        self.spawn_offset = np.asarray(spawn_offset if spawn_offset is not None else (0.0, 0.0, 0.0), dtype="f4")
        self.size_multiplier = max(abs(float(size_multiplier)), 0.01)

        corners = np.asarray([
            (-1.0, -1.0, 0.0, 1.0),
            (1.0, -1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0, 0.0),
        ], dtype="f4")
        indices = np.asarray((0, 1, 2, 0, 2, 3), dtype="u4")
        instances = self._build_instances()
        self.instance_count = len(instances)
        self.vertex_buffer = ctx.buffer(corners.tobytes())
        self.index_buffer = ctx.buffer(indices.tobytes())
        self.instance_buffer = ctx.buffer(instances.astype("f4").tobytes())
        self.vao = ctx.vertex_array(
            program,
            [
                (self.vertex_buffer, "2f 2f", "in_corner", "in_uv"),
                (self.instance_buffer, "3f 1f 1f /i", "in_spawn", "in_birth_ms", "in_base_size"),
            ],
            self.index_buffer,
            index_element_size=4,
        )
        with Image.open(texture_path) as source:
            image = source.convert("RGBA")
            self.texture_size = image.size
            self.texture = ctx.texture(image.size, 4, image.tobytes())
        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = True
        self.texture.repeat_y = True

        sx0, sx1 = curve_endpoints(change, 0, 1.0)
        sy0, sy1 = curve_endpoints(change, 1, sx0)
        colors = [curve_endpoints(change, index, 255.0) for index in range(3, 6)]
        alpha = curve_endpoints(change, 6, 255.0)
        if max(alpha) <= 0.001:
            alpha = (255.0, 255.0)
        self.scale_start = (max(abs(sx0), 0.001), max(abs(sy0), 0.001))
        self.scale_end = (max(abs(sx1), 0.001), max(abs(sy1), 0.001))
        self.color_start = tuple(max(0.0, min(pair[0] / 255.0, 2.0)) for pair in colors) + (alpha_value(alpha[0]),)
        self.color_end = tuple(max(0.0, min(pair[1] / 255.0, 2.0)) for pair in colors) + (alpha_value(alpha[1]),)
        self.velocity, self.acceleration = self._motion_vectors()
        self.frame_count, self.frame_columns, self.frame_rows, self.frame_ms = self._animation_info()

    def _particle_count(self) -> int:
        sequence_count = int(self.create.get("sequence_count", 0))
        amount = int(self.create.get("amount", 0))
        return max(1, min(max(sequence_count, amount, 1), 160))

    def _spawn_position(self, index: int, count: int) -> np.ndarray:
        shape = int(self.create.get("shape_type", 0))
        if shape == 0:
            return effect_vector(self.create.get("point", (0.0, 0.0, 0.0)))
        if shape == 1:
            low = np.asarray(self.create.get("minimum", (0.0, 0.0, 0.0)), dtype="f4")
            high = np.asarray(self.create.get("maximum", (0.0, 0.0, 0.0)), dtype="f4")
            weights = np.asarray((low_discrepancy(index, 2), low_discrepancy(index, 3), low_discrepancy(index, 5)), dtype="f4")
            return effect_vector(low + (high - low) * weights)
        if shape == 2:
            radius = abs(float(self.create.get("radius", 0.0)))
            angle = math.tau * low_discrepancy(index, 2)
            radial = radius * math.sqrt(low_discrepancy(index, 3))
            return effect_vector((math.cos(angle) * radial, math.sin(angle) * radial, 0.0))
        control = self.create.get("control_points", {})
        keys = control.get("keys") or [control.get("first"), control.get("last")]
        keys = [key for key in keys if key]
        if keys:
            key = keys[index % len(keys)]
            return effect_vector(key.get("value", (0.0, 0.0, 0.0)))
        low = np.asarray(self.create.get("minimum", (0.0, 0.0, 0.0)), dtype="f4")
        high = np.asarray(self.create.get("maximum", low), dtype="f4")
        weight = low_discrepancy(index, 2)
        return effect_vector(low + (high - low) * weight)

    def _build_instances(self) -> np.ndarray:
        count = self._particle_count()
        sequence = list(self.create.get("sequence", []))
        minimum = float(self.visual.get("minimum_size", 0.0) or 0.0)
        maximum = float(self.visual.get("maximum_size", 0.0) or 0.0)
        if minimum <= 0.0 and maximum <= 0.0:
            scale = abs(float(self.visual.get("resource_scale", 1.0) or 1.0))
            minimum = maximum = 0.18 * scale
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        rows = []
        for index in range(count):
            position = self._spawn_position(index, count) + self.spawn_offset
            birth = float(sequence[index % len(sequence)]) if sequence else self.duration_ms * index / count
            weight = low_discrepancy(index, 7)
            size = (minimum + (maximum - minimum) * weight) * self.size_multiplier
            rows.append((*position, birth % self.duration_ms, max(size, 0.01)))
        return np.asarray(rows, dtype="f4")

    def _motion_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = effect_vector(self.move.get("velocity", (0.0, 0.0, 0.0)))
        speed = float(self.move.get("speed", 0.0) or 0.0)
        if float(np.linalg.norm(velocity)) < 1e-6:
            velocity = effect_vector(self.create.get("direction", (0.0, 0.0, 0.0)))
        length = float(np.linalg.norm(velocity))
        if length > 1e-6:
            velocity = velocity / length * speed
        acceleration = effect_vector(self.move.get("acceleration_direction", (0.0, 0.0, 0.0)))
        acceleration *= float(self.move.get("acceleration", 0.0) or 0.0)
        return velocity, acceleration

    def _animation_info(self) -> tuple[int, int, int, float]:
        if not self.visual.get("animated"):
            return 1, 1, 1, self.duration_ms
        fields = self.visual.get("animation_fields", [0, 0, 1, 100])
        frame_width, frame_height, frame_count, frame_ms = (int(value) for value in fields)
        width, height = self.texture_size
        columns = max(width // max(frame_width, 1), 1)
        rows = max(height // max(frame_height, 1), 1)
        return max(frame_count, 1), columns, rows, max(float(frame_ms), 1.0)

    def render(self, program: moderngl.Program, modelview: np.ndarray, projection: np.ndarray, time_ms: float) -> None:
        program["u_modelview"].write(modelview.T.astype("f4").tobytes())
        program["u_projection"].write(projection.T.astype("f4").tobytes())
        program["u_time_ms"].value = time_ms
        program["u_duration_ms"].value = self.duration_ms
        program["u_velocity"].value = tuple(float(value) for value in self.velocity)
        program["u_acceleration"].value = tuple(float(value) for value in self.acceleration)
        program["u_scale_start"].value = self.scale_start
        program["u_scale_end"].value = self.scale_end
        program["u_color_start"].value = self.color_start
        program["u_color_end"].value = self.color_end
        program["u_frame_count"].value = self.frame_count
        program["u_frame_columns"].value = self.frame_columns
        program["u_frame_rows"].value = self.frame_rows
        program["u_frame_ms"].value = self.frame_ms
        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES, instances=self.instance_count)

    def release(self) -> None:
        self.vao.release()
        self.vertex_buffer.release()
        self.index_buffer.release()
        self.instance_buffer.release()
        self.texture.release()


class WZMViewer:
    def __init__(self, model: WZMModel, effect_code: str = "", wzu_path: Path | None = None, attachments: list[str] | None = None, texture_overrides: dict[int, Path] | None = None, texture_enabled: bool = True, effect_enabled: bool = True, preview_effect: bool = False, screenshot: Path | None = None, close_after_frame: int = 0, visible: bool = True):
        self.model = model
        self.effect_codes = [
            code.strip() for code in effect_code.split(",")
            if code.strip().lower() not in ("", "0", "null")
        ]
        self.wzu_path = wzu_path
        self.attachment_args = attachments or []
        self.texture_overrides = texture_overrides or {}
        self.effect_codes = list(dict.fromkeys(self.effect_codes))
        self.texture_enabled = texture_enabled
        self.preview_effect = preview_effect
        self.effect_enabled = effect_enabled and bool(self.effect_codes)
        self.lighting = True
        self.wireframe = False
        self.yaw = math.radians(-18)
        self.pitch = math.radians(-6)
        self.zoom = 1.0
        self.start_time = time.perf_counter()
        self.screenshot = screenshot
        self.close_after_frame = close_after_frame
        self.visible = visible
        self.frame_count = 0

        config = pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        self.window = pyglet.window.Window(
            1000, 760,
            resizable=True,
            config=config,
            caption="SUN WZM 交互式 3D 预览",
            visible=visible,
        )
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.program["u_texture"].value = 0
        self.program["u_normal_texture"].value = 1
        self.program["u_specular_texture"].value = 2
        self.meshes = self._build_meshes()
        self.bone_positions = self._build_bone_positions()
        self.effect_program = self.ctx.program(
            vertex_shader=EFFECT_VERTEX_SHADER,
            fragment_shader=EFFECT_FRAGMENT_SHADER,
        )
        self.effect_program["u_texture"].value = 0
        self.effect_layers = self._build_effect_layers()
        self.effect_enabled = self.effect_enabled and bool(self.effect_layers)
        self._update_caption()
        self._install_events()

    def _build_meshes(self) -> list[RenderMesh]:
        transformed_positions = np.asarray([(x, z, -y) for x, y, z in self.model.positions], dtype="f4")
        minimum = transformed_positions.min(axis=0)
        maximum = transformed_positions.max(axis=0)
        center = (minimum + maximum) / 2.0
        factor = 2.15 / max(float((maximum - minimum).max()), 1e-6)
        self.model_center = center
        self.model_factor = factor
        transformed_positions = (transformed_positions - center) * factor

        meshes: list[RenderMesh] = []
        role_paths: dict[str, Path] = {}
        companion = self.model.source.with_suffix(".WZU")
        if companion.is_file():
            for row in inspect_wzu_materials(companion):
                role = str(row["role"])
                path = resolve_texture(self.model.source.parent, str(row["name"]))
                if path and role not in role_paths:
                    role_paths[role] = path
        colors = ((0.58, 0.68, 0.82), (0.76, 0.62, 0.48), (0.54, 0.76, 0.67))
        for mesh_index, submesh in enumerate(self.model.submeshes):
            rows = []
            for vertex in submesh.vertices:
                position = transformed_positions[vertex.position_index]
                nx, ny, nz = vertex.normal
                normal = np.asarray((nx, nz, -ny), dtype="f4")
                length = float(np.linalg.norm(normal))
                if length:
                    normal /= length
                tx, ty, tz = vertex.tangent
                tangent = np.asarray((tx, tz, -ty), dtype="f4")
                tangent_length = float(np.linalg.norm(tangent))
                if tangent_length:
                    tangent /= tangent_length
                rows.append((*position, *normal, *vertex.uv, *tangent))
            texture_path = self.texture_overrides.get(mesh_index)
            if texture_path is None or not texture_path.is_file():
                texture_path = resolve_texture(self.model.source.parent, submesh.diffuse)
            render_mesh = RenderMesh(
                self.ctx,
                self.program,
                np.asarray(rows, dtype="f4"),
                np.asarray(submesh.indices, dtype="u4"),
                texture_path,
                role_paths.get("normal"),
                role_paths.get("specular") or resolve_texture(self.model.source.parent, submesh.specular),
            )
            render_mesh.base_color = colors[mesh_index % len(colors)]
            meshes.append(render_mesh)
        return meshes

    def _build_bone_positions(self) -> list[np.ndarray]:
        local_matrices: list[np.ndarray] = []
        for bone in self.model.bones:
            x, y, z, w = _bone_rotation(self.model, bone)
            matrix = np.eye(4, dtype="f8")
            matrix[:3, :3] = (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            )
            matrix[:3, 3] = bone.translation
            local_matrices.append(matrix)

        global_matrices: list[np.ndarray | None] = [None] * len(local_matrices)
        visiting: set[int] = set()

        def global_matrix(index: int) -> np.ndarray:
            cached = global_matrices[index]
            if cached is not None:
                return cached
            if index in visiting:
                return local_matrices[index]
            visiting.add(index)
            parent = self.model.bones[index].parent
            if 0 <= parent < len(local_matrices) and parent != index:
                value = global_matrix(parent) @ local_matrices[index]
            else:
                value = local_matrices[index]
            visiting.discard(index)
            global_matrices[index] = value
            return value

        result = []
        for index in range(len(local_matrices)):
            x, y, z = global_matrix(index)[:3, 3]
            transformed = np.asarray((x, z, -y), dtype="f4")
            result.append((transformed - self.model_center) * self.model_factor)
        return result

    def _effect_attachments(self) -> dict[str, list[tuple[np.ndarray, float]]]:
        result: dict[str, list[tuple[np.ndarray, float]]] = {}

        def add(code: str, bone_index: int | None, local_position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> None:
            position = np.zeros(3, dtype="f4")
            if bone_index is not None and 0 <= bone_index < len(self.bone_positions):
                position += self.bone_positions[bone_index]
            position += effect_vector(local_position)
            multiplier = sum(abs(float(value)) for value in scale) / 3.0
            key = (tuple(np.round(position, 5)), round(multiplier, 5))
            existing = result.setdefault(code, [])
            if not any((tuple(np.round(item[0], 5)), round(item[1], 5)) == key for item in existing):
                existing.append((position, multiplier))

        for value in self.attachment_args:
            code, separator, raw_bone = value.partition(":")
            if code in self.effect_codes:
                try:
                    bone = int(raw_bone) if separator else None
                except ValueError:
                    bone = None
                add(code, bone)

        if self.wzu_path and self.wzu_path.is_file():
            catalog = EffectCatalog()
            for node in catalog.inspect_wzu(self.wzu_path):
                if node.code in self.effect_codes:
                    add(node.code, node.bone_index, node.position, node.scale)
        return result

    def _build_effect_layers(self) -> list[EffectLayer]:
        if not self.effect_codes:
            return []
        catalog = EffectCatalog()
        layers: list[EffectLayer] = []
        attachments = self._effect_attachments()
        for effect_code in self.effect_codes:
            for inspection in catalog.inspect_code(effect_code):
                variants = attachments.get(effect_code) or [(np.zeros(3, dtype="f4"), 1.0)]
                for position, multiplier in variants:
                    layers.extend(self._build_inspection_layers(inspection, position, multiplier))
        return layers

    def _build_inspection_layers(self, inspection, spawn_offset: np.ndarray, size_multiplier: float) -> list[EffectLayer]:
        layers: list[EffectLayer] = []
        structure = inspection.structure
        if not structure.get("supported"):
            return layers
        textures = {
            row.code: Path(row.actual_path)
            for row in inspection.textures
            if row.actual_path and Path(row.actual_path).is_file()
        }
        creates = structure.get("creates", [])
        moves = structure.get("moves", [])
        changes = structure.get("changes", [])
        visuals = structure.get("visuals", [])
        for element in structure.get("elements", []):
            indices = (
                int(element.get("create_index", -1)),
                int(element.get("move_index", -1)),
                int(element.get("change_index", -1)),
                int(element.get("visual_index", -1)),
            )
            if (
                indices[0] not in range(len(creates))
                or indices[1] not in range(len(moves))
                or indices[2] not in range(len(changes))
                or indices[3] not in range(len(visuals))
            ):
                continue
            visual = visuals[indices[3]]
            texture_path = textures.get(visual.get("resource_code", ""))
            if texture_path is None or int(visual.get("resource_type", 0)) not in (1, 2):
                continue
            layers.append(EffectLayer(
                self.ctx,
                self.effect_program,
                element,
                creates[indices[0]],
                moves[indices[1]],
                changes[indices[2]],
                visual,
                texture_path,
                spawn_offset,
                size_multiplier,
            ))
        return layers

    def _update_caption(self) -> None:
        texture = "开" if self.texture_enabled else "关"
        effect = f"开/{len(self.effect_codes)}组/{len(self.effect_layers)}层" if self.effect_enabled else "关"
        preview = "开" if self.preview_effect else "关"
        lighting = "开" if self.lighting else "关"
        wire = "开" if self.wireframe else "关"
        self.window.set_caption(
            f"SUN 3D | {self.model.source.name} | 贴图 {texture} | EWZ {effect} | 轮廓光 {preview} | "
            f"光照 {lighting} | 线框 {wire} | 左拖旋转/滚轮缩放/T/E/G/L/W/R"
        )

    def _install_events(self) -> None:
        @self.window.event
        def on_draw():
            self.draw()
            self.frame_count += 1
            if self.screenshot and self.frame_count == max(self.close_after_frame, 2):
                self.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pyglet.image.get_buffer_manager().get_color_buffer().save(str(self.screenshot))
                self.screenshot = None
                if self.close_after_frame:
                    pyglet.clock.schedule_once(lambda _dt: self.window.close(), 0)

        @self.window.event
        def on_resize(width: int, height: int):
            self.ctx.viewport = (0, 0, width, height)

        @self.window.event
        def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
            if buttons & mouse.LEFT:
                self.yaw += dx * 0.009
                self.pitch = max(math.radians(-88), min(math.radians(88), self.pitch - dy * 0.009))

        @self.window.event
        def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
            self.zoom = max(0.28, min(4.5, self.zoom * (1.0 + scroll_y * 0.1)))

        @self.window.event
        def on_key_press(symbol, _modifiers):
            if symbol == key.T:
                self.texture_enabled = not self.texture_enabled
            elif symbol == key.E:
                self.effect_enabled = not self.effect_enabled if self.effect_layers else False
            elif symbol == key.G:
                self.preview_effect = not self.preview_effect
            elif symbol == key.L:
                self.lighting = not self.lighting
            elif symbol == key.W:
                self.wireframe = not self.wireframe
            elif symbol == key.R:
                self.yaw, self.pitch, self.zoom = math.radians(-18), math.radians(-6), 1.0
            elif symbol == key.ESCAPE:
                self.window.close()
            self._update_caption()

        @self.window.event
        def on_close():
            for mesh in self.meshes:
                mesh.release()
            for layer in self.effect_layers:
                layer.release()
            self.effect_program.release()
            self.program.release()

    def draw(self) -> None:
        self.ctx.clear(0.055, 0.065, 0.082, 1.0, depth=1.0)
        self.ctx.wireframe = self.wireframe
        aspect = max(self.window.width, 1) / max(self.window.height, 1)
        model_matrix = rotation_y(self.yaw) @ rotation_x(self.pitch) @ scale_matrix(self.zoom)
        view_matrix = translation(-3.25)
        projection = perspective(42.0, aspect, 0.05, 100.0)
        mvp = projection @ view_matrix @ model_matrix
        self.program["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self.program["u_model"].write(model_matrix.T.astype("f4").tobytes())
        self.program["u_use_lighting"].value = self.lighting
        self.program["u_preview_effect"].value = self.preview_effect
        self.program["u_time"].value = float(time.perf_counter() - self.start_time)
        for mesh in self.meshes:
            mesh.texture.use(0)
            mesh.normal_texture.use(1)
            mesh.specular_texture.use(2)
            self.program["u_use_texture"].value = self.texture_enabled and mesh.has_texture
            self.program["u_use_normal_texture"].value = mesh.has_normal_texture
            self.program["u_use_specular_texture"].value = mesh.has_specular_texture
            self.program["u_base_color"].value = mesh.base_color
            mesh.vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False
        if self.effect_enabled:
            self.ctx.depth_mask = False
            modelview = view_matrix @ model_matrix
            effect_time = float((time.perf_counter() - self.start_time) * 1000.0)
            for layer in self.effect_layers:
                self.ctx.blend_func = (
                    (moderngl.SRC_ALPHA, moderngl.ONE)
                    if layer.additive
                    else (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                )
                layer.render(self.effect_program, modelview, projection, effect_time)
            self.ctx.depth_mask = True
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def run(self) -> None:
        if not self.visible:
            self.window.switch_to()
            self.ctx.viewport = (0, 0, self.window.width, self.window.height)
            for _ in range(max(self.close_after_frame, 2)):
                self.draw()
            self.ctx.finish()
            if self.screenshot:
                self.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pixels = self.ctx.screen.read(components=3, alignment=1)
                image = Image.frombytes("RGB", (self.window.width, self.window.height), pixels)
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image.save(self.screenshot)
                self.screenshot = None
            self.window.close()
            return
        pyglet.app.run(interval=1 / 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="SUN WZM 交互式 3D 查看器")
    parser.add_argument("source", type=Path)
    parser.add_argument("--effect-code", default="")
    parser.add_argument("--wzu", type=Path)
    parser.add_argument("--attachment", action="append", default=[])
    parser.add_argument(
        "--texture-override", action="append", default=[], metavar="SLOT=PATH",
        help="手动指定从 0 开始的子网格贴图；可重复使用",
    )
    parser.add_argument("--no-effect", action="store_true")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--preview-effect", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--close-after-frame", type=int, default=0)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    try:
        texture_overrides: dict[int, Path] = {}
        for value in args.texture_override:
            slot_text, separator, path_text = value.partition("=")
            if not separator or not slot_text.isdigit() or not path_text:
                parser.error(f"无效的 --texture-override：{value!r}，应为 SLOT=PATH")
            texture_overrides[int(slot_text)] = Path(path_text)
        model = parse_wzm(args.source)
        viewer = WZMViewer(
            model,
            effect_code=args.effect_code,
            wzu_path=args.wzu,
            attachments=args.attachment,
            texture_overrides=texture_overrides,
            texture_enabled=not args.no_texture,
            effect_enabled=not args.no_effect,
            preview_effect=args.preview_effect,
            screenshot=args.screenshot,
            close_after_frame=args.close_after_frame,
            visible=not args.hidden,
        )
        viewer.run()
    except (OSError, WZMError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

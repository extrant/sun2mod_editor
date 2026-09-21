from __future__ import annotations

"""Rebuild SUN WZM mesh/weight chunks from an edited glTF.

The original WZM remains the template: its header, skeleton,
animations (stored separately in WZA), and unknown chunks are preserved byte
for byte. Only the mesh and variable-weight chunks are replaced. Material image
names may optionally be replaced while rebuilding the mesh chunk.
"""

import base64
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from wzm_unpack import (
    CHUNK_MESH,
    CHUNK_WEIGHTS,
    Reader,
    SubMesh,
    WZMError,
    WZMModel,
    inspect_wzu_materials,
    parse_wzm,
)


COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
INTEGER_COMPONENTS = {5120, 5121, 5122, 5123, 5125}


@dataclass(slots=True)
class PrimitiveData:
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    uvs: np.ndarray
    indices: list[int]
    weights: list[list[tuple[int, float]]]
    material_name: str


@dataclass(slots=True)
class RepackReport:
    destination: Path
    vertex_count: int
    triangle_count: int
    submesh_count: int
    weighted_vertex_count: int
    normal_count: int
    removed_bones: list[str]
    fallback_weight_vertices: int
    warnings: list[str]


class GltfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        self.buffers = [self._load_buffer(row) for row in self.document.get("buffers", [])]

    def _load_buffer(self, row: dict) -> bytes:
        uri = unquote(str(row.get("uri", "")))
        if uri.startswith("data:"):
            try:
                return base64.b64decode(uri.split(",", 1)[1])
            except (IndexError, ValueError) as exc:
                raise WZMError("glTF 内嵌缓冲区无效") from exc
        path = (self.path.parent / uri).resolve()
        if not path.is_file():
            raise WZMError(f"glTF 缓冲区不存在：{path}")
        return path.read_bytes()

    def accessor(self, index: int) -> np.ndarray:
        accessors = self.document.get("accessors", [])
        if not 0 <= index < len(accessors):
            raise WZMError(f"glTF accessor 越界：{index}")
        accessor = accessors[index]
        if "sparse" in accessor:
            raise WZMError("暂不支持 glTF sparse accessor；请由 Blender 重新导出")
        if "bufferView" not in accessor:
            raise WZMError("glTF accessor 没有 bufferView")
        views = self.document.get("bufferViews", [])
        view = views[int(accessor["bufferView"])]
        component_type = int(accessor["componentType"])
        if component_type not in COMPONENT_FORMATS:
            raise WZMError(f"不支持的 glTF 分量类型：{component_type}")
        accessor_type = str(accessor["type"])
        if accessor_type not in TYPE_COMPONENTS:
            raise WZMError(f"不支持的 glTF accessor 类型：{accessor_type}")
        fmt, component_size = COMPONENT_FORMATS[component_type]
        width = TYPE_COMPONENTS[accessor_type]
        item_size = component_size * width
        stride = int(view.get("byteStride", item_size))
        if stride < item_size:
            raise WZMError("glTF bufferView 的 byteStride 小于元素大小")
        buffer_index = int(view.get("buffer", 0))
        raw = self.buffers[buffer_index]
        offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        count = int(accessor.get("count", 0))
        result = np.empty((count, width), dtype=np.float64)
        unpack_format = "<" + fmt * width
        for row_index in range(count):
            start = offset + row_index * stride
            if start + item_size > len(raw):
                raise WZMError("glTF accessor 读取越界")
            result[row_index] = struct.unpack_from(unpack_format, raw, start)
        if accessor.get("normalized") and component_type in INTEGER_COMPONENTS:
            if component_type == 5120:
                result = np.maximum(result / 127.0, -1.0)
            elif component_type == 5121:
                result /= 255.0
            elif component_type == 5122:
                result = np.maximum(result / 32767.0, -1.0)
            elif component_type == 5123:
                result /= 65535.0
            elif component_type == 5125:
                result /= 4294967295.0
        return result[:, 0] if width == 1 else result


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.size != 16:
            raise WZMError("glTF 节点矩阵长度不是 16")
        return values.reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    x, y, z, w = (float(value) for value in node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    rotation = np.asarray((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _global_node_matrices(nodes: list[dict]) -> list[np.ndarray]:
    parents = [-1] * len(nodes)
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            child_index = int(child)
            if 0 <= child_index < len(nodes):
                parents[child_index] = parent_index
    cache: list[np.ndarray | None] = [None] * len(nodes)
    visiting: set[int] = set()

    def resolve(index: int) -> np.ndarray:
        cached = cache[index]
        if cached is not None:
            return cached
        if index in visiting:
            raise WZMError("glTF 节点层级存在循环")
        visiting.add(index)
        local = _node_matrix(nodes[index])
        parent = parents[index]
        result = resolve(parent) @ local if parent >= 0 else local
        visiting.remove(index)
        cache[index] = result
        return result

    return [resolve(index) for index in range(len(nodes))]


def _normalise_rows(values: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-12
    result[valid] /= lengths[valid, None]
    result[~valid] = fallback
    return result


def _generate_tangents(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: list[int]) -> np.ndarray:
    accumulated = np.zeros_like(positions, dtype=np.float64)
    bitangents = np.zeros_like(positions, dtype=np.float64)
    for offset in range(0, len(indices), 3):
        i0, i1, i2 = indices[offset:offset + 3]
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
        edge1, edge2 = p1 - p0, p2 - p0
        duv1, duv2 = uv1 - uv0, uv2 - uv0
        denominator = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(denominator) < 1e-12:
            continue
        reciprocal = 1.0 / denominator
        tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * reciprocal
        bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * reciprocal
        for index in (i0, i1, i2):
            accumulated[index] += tangent
            bitangents[index] += bitangent
    xyz = accumulated - normals * np.sum(normals * accumulated, axis=1)[:, None]
    xyz = _normalise_rows(xyz, (1.0, 0.0, 0.0))
    signs = np.where(np.sum(np.cross(normals, xyz) * bitangents, axis=1) < 0.0, -1.0, 1.0)
    return np.column_stack((xyz, signs))


def _material_name(document: dict, primitive: dict) -> str:
    index = primitive.get("material")
    materials = document.get("materials", [])
    if isinstance(index, int) and 0 <= index < len(materials):
        return str(materials[index].get("name", ""))
    return ""


def _clean_material_name(value: str) -> str:
    value = re.sub(r"\.\d{3}$", "", value.casefold())
    return re.sub(r"[^a-z0-9]+", "", value)


def _original_material_name(submesh: SubMesh, index: int) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", submesh.diffuse).strip("._")
    return value or f"material_{index}"


def _merge_primitives(rows: list[PrimitiveData], material_name: str) -> PrimitiveData:
    position_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    tangent_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    indices: list[int] = []
    weights: list[list[tuple[int, float]]] = []
    offset = 0
    for row in rows:
        position_parts.append(row.positions)
        normal_parts.append(row.normals)
        tangent_parts.append(row.tangents)
        uv_parts.append(row.uvs)
        indices.extend(index + offset for index in row.indices)
        weights.extend(row.weights)
        offset += len(row.positions)
    return PrimitiveData(
        positions=np.concatenate(position_parts, axis=0),
        normals=np.concatenate(normal_parts, axis=0),
        tangents=np.concatenate(tangent_parts, axis=0),
        uvs=np.concatenate(uv_parts, axis=0),
        indices=indices,
        weights=weights,
        material_name=material_name,
    )


def _extract_primitives(
    reader: GltfReader,
    original: WZMModel,
) -> tuple[list[PrimitiveData], list[str], list[str], int]:
    document = reader.document
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    global_matrices = _global_node_matrices(nodes)
    original_bones = {(bone.name or f"bone_{bone.index}"): bone.index for bone in original.bones}
    original_bones_folded = {name.casefold(): index for name, index in original_bones.items()}
    warnings: list[str] = ["骨架层级和 WZA 动画保持原文件不变；顶点权重按骨骼名称写回"]
    result: list[PrimitiveData] = []
    removed_bones: set[str] = set()
    fallback_weight_vertices = 0
    node_parents: dict[int, int] = {}
    for parent_index, parent_node in enumerate(nodes):
        for child_index in parent_node.get("children", []):
            node_parents[int(child_index)] = parent_index

    def resolve_original_bone(joint_node: int) -> tuple[int | None, str]:
        current = joint_node
        first_name = ""
        visited: set[int] = set()
        while 0 <= current < len(nodes) and current not in visited:
            visited.add(current)
            name = str(nodes[current].get("name", ""))
            if not first_name:
                first_name = name
            bone_index = original_bones.get(name, original_bones_folded.get(name.casefold()))
            if bone_index is not None:
                return bone_index, first_name
            current = node_parents.get(current, -1)
        return None, first_name

    reachable: set[int] = set()
    scenes = document.get("scenes", [])
    scene_index = int(document.get("scene", 0))
    pending = list(scenes[scene_index].get("nodes", [])) if 0 <= scene_index < len(scenes) else list(range(len(nodes)))
    while pending:
        index = int(pending.pop())
        if not 0 <= index < len(nodes) or index in reachable:
            continue
        reachable.add(index)
        pending.extend(nodes[index].get("children", []))
    mesh_nodes = [
        (index, nodes[index]) for index in sorted(reachable)
        if isinstance(nodes[index].get("mesh"), int)
    ]
    if not mesh_nodes and meshes:
        mesh_nodes = [(-1, {"mesh": index}) for index in range(len(meshes))]

    for node_index, node in mesh_nodes:
        mesh_index = int(node["mesh"])
        if not 0 <= mesh_index < len(meshes):
            continue
        world = global_matrices[node_index] if node_index >= 0 else np.eye(4, dtype=np.float64)
        linear = world[:3, :3]
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError as exc:
            raise WZMError("Blender 网格对象使用了不可逆缩放") from exc
        skin_index = node.get("skin")
        joint_nodes: list[int] = []
        if isinstance(skin_index, int) and 0 <= skin_index < len(skins):
            joint_nodes = [int(value) for value in skins[skin_index].get("joints", [])]

        for primitive in meshes[mesh_index].get("primitives", []):
            if int(primitive.get("mode", 4)) != 4:
                raise WZMError("WZM 只能回流三角形；请在 Blender 中将网格三角化")
            attributes = primitive.get("attributes", {})
            missing = [name for name in ("POSITION", "NORMAL", "TEXCOORD_0") if name not in attributes]
            if missing:
                raise WZMError(f"Blender 网格缺少必要属性：{', '.join(missing)}")
            positions = np.asarray(reader.accessor(int(attributes["POSITION"])), dtype=np.float64)
            normals = np.asarray(reader.accessor(int(attributes["NORMAL"])), dtype=np.float64)
            uvs = np.asarray(reader.accessor(int(attributes["TEXCOORD_0"])), dtype=np.float64)
            if len(positions) != len(normals) or len(positions) != len(uvs):
                raise WZMError("glTF 顶点属性数量不一致")
            homogeneous = np.column_stack((positions, np.ones(len(positions), dtype=np.float64)))
            positions = (homogeneous @ world.T)[:, :3]
            normals = _normalise_rows(normals @ normal_matrix.T, (0.0, 1.0, 0.0))
            if "indices" in primitive:
                indices = [int(value) for value in reader.accessor(int(primitive["indices"])).tolist()]
            else:
                indices = list(range(len(positions)))
            if len(indices) % 3:
                raise WZMError("glTF 索引数量不是 3 的倍数")
            if indices and (min(indices) < 0 or max(indices) >= len(positions)):
                raise WZMError("glTF 三角形索引越界")

            if "TANGENT" in attributes:
                tangents = np.asarray(reader.accessor(int(attributes["TANGENT"])), dtype=np.float64)
                tangent_xyz = _normalise_rows(tangents[:, :3] @ normal_matrix.T, (1.0, 0.0, 0.0))
                tangent_w = tangents[:, 3] if tangents.shape[1] >= 4 else np.ones(len(tangents))
                if np.linalg.det(linear) < 0.0:
                    tangent_w = -tangent_w
                tangents = np.column_stack((tangent_xyz, tangent_w))
            else:
                tangents = _generate_tangents(positions, normals, uvs, indices)
                warnings.append(f"材质 {len(result) + 1} 缺少切线，已根据 UV 自动重建")

            vertex_weights: list[list[tuple[int, float]]] = [[] for _ in range(len(positions))]
            joint_keys = sorted(
                (name for name in attributes if name.startswith("JOINTS_")),
                key=lambda name: int(name.split("_", 1)[1]),
            )
            for joint_key in joint_keys:
                suffix = joint_key.split("_", 1)[1]
                weight_key = f"WEIGHTS_{suffix}"
                if weight_key not in attributes:
                    continue
                joints = np.asarray(reader.accessor(int(attributes[joint_key])), dtype=np.int64)
                weights = np.asarray(reader.accessor(int(attributes[weight_key])), dtype=np.float64)
                if len(joints) != len(positions) or len(weights) != len(positions):
                    raise WZMError("glTF 权重属性数量与顶点数不一致")
                for vertex_index in range(len(positions)):
                    for joint_ordinal, weight in zip(joints[vertex_index], weights[vertex_index]):
                        if weight <= 1e-7:
                            continue
                        ordinal = int(joint_ordinal)
                        if not 0 <= ordinal < len(joint_nodes):
                            raise WZMError(f"glTF 权重引用了无效关节：{ordinal}")
                        joint_node = joint_nodes[ordinal]
                        bone_index, original_name = resolve_original_bone(joint_node)
                        direct_name = str(nodes[joint_node].get("name", "")) if 0 <= joint_node < len(nodes) else ""
                        direct_index = original_bones.get(
                            direct_name, original_bones_folded.get(direct_name.casefold()),
                        )
                        if direct_index is None:
                            removed_bones.add(direct_name or original_name or str(ordinal))
                        if bone_index is None:
                            continue
                        vertex_weights[vertex_index].append((bone_index, float(weight)))

            if original.bones and not joint_keys:
                raise WZMError("Blender 网格没有蒙皮权重；请保留 Armature 和顶点组")
            for vertex_index, influences in enumerate(vertex_weights):
                combined: dict[int, float] = {}
                for bone_index, weight in influences:
                    combined[bone_index] = combined.get(bone_index, 0.0) + weight
                ordered = sorted(combined.items(), key=lambda pair: pair[1], reverse=True)
                if original.bones and not ordered:
                    ordered = [(0, 1.0)]
                    fallback_weight_vertices += 1
                total = sum(weight for _bone, weight in ordered) or 1.0
                vertex_weights[vertex_index] = [(bone, weight / total) for bone, weight in ordered]

            result.append(PrimitiveData(
                positions=positions,
                normals=normals,
                tangents=tangents,
                uvs=uvs,
                indices=indices,
                weights=vertex_weights,
                material_name=_material_name(document, primitive),
            ))

    original_names = [_clean_material_name(_original_material_name(mesh, index)) for index, mesh in enumerate(original.submeshes)]
    edited_names = [_clean_material_name(primitive.material_name) for primitive in result]
    if not result:
        raise WZMError("Blender 中没有可回流的三角形网格")
    if len(original.submeshes) == 1 and result:
        if len(result) > 1:
            warnings.append(
                f"Blender 的 {len(result)} 个三角形材质组已自动合并到原 WZM 的单一子网格"
            )
        result = [_merge_primitives(result, _original_material_name(original.submeshes[0], 0))]
    elif len(set(original_names)) == len(original_names) and set(edited_names).issubset(set(original_names)):
        grouped = {
            name: [primitive for edited_name, primitive in zip(edited_names, result) if edited_name == name]
            for name in original_names
        }
        if all(grouped.values()):
            result = [_merge_primitives(grouped[name], _original_material_name(original.submeshes[index], index))
                      for index, name in enumerate(original_names)]
        elif len(result) == len(original.submeshes):
            warnings.append("Blender 缺少部分原材质槽名称，已按当前材质槽顺序对应")
        else:
            warnings.append(
                f"Blender 材质组为 {len(result)}、原 WZM 子网格为 {len(original.submeshes)}；"
                "已自动合并为一个 WZM 子网格"
            )
            result = [_merge_primitives(result, result[0].material_name or _original_material_name(original.submeshes[0], 0))]
    elif len(result) != len(original.submeshes):
        warnings.append(
            f"Blender 材质组为 {len(result)}、原 WZM 子网格为 {len(original.submeshes)}；"
            "已自动合并为一个 WZM 子网格"
        )
        result = [_merge_primitives(result, result[0].material_name or _original_material_name(original.submeshes[0], 0))]
    elif original_names != edited_names:
        warnings.append("Blender 材质名称已变化，已按材质槽顺序对应原 WZM 子网格")
    if removed_bones:
        preview = "、".join(sorted(removed_bones)[:8])
        suffix = f" 等 {len(removed_bones)} 个" if len(removed_bones) > 8 else ""
        warnings.append(f"已从回流数据删除 Blender 异常骨骼：{preview}{suffix}")
    if fallback_weight_vertices:
        warnings.append(
            f"删除异常骨骼后有 {fallback_weight_vertices} 个顶点没有有效权重，已安全绑定到原骨架根骨"
        )
    return result, warnings, sorted(removed_bones), fallback_weight_vertices


def _mesh_templates(model: WZMModel) -> tuple[bytes, list[dict[str, bytes]]]:
    raw = model.source.read_bytes()
    chunk = next((row for row in model.chunks if row["id"] == CHUNK_MESH), None)
    if chunk is None:
        raise WZMError("原 WZM 没有网格区块")
    reader = Reader(raw)
    reader.seek(chunk["offset"] + 6)
    position_count = reader.u32()
    reader.skip(position_count * 14)
    post_positions = reader.read(1) if model.version in (117, 118, 119) else b""
    submesh_count = reader.u8()
    reader.u32()
    templates: list[dict[str, bytes]] = []
    for _ in range(submesh_count):
        before_names = reader.read(4) if model.version >= 117 else b""
        reader.short_string()
        reader.short_string()
        after_names = reader.read(5)
        vertex_count = reader.u32()
        triangle_count = reader.u32()
        vertex_unknown = b"\xff" * (4 if model.version <= 114 else 8)
        if vertex_count:
            reader.skip(18)
            vertex_unknown = reader.read(4 if model.version <= 114 else 8)
            reader.skip(12 + 1 + 8)
            remaining_vertices = vertex_count - 1
            reader.skip(remaining_vertices * (43 if model.version <= 114 else 47))
        reader.skip(triangle_count * 12)
        tail = reader.read(4) if model.version >= 118 else b""
        templates.append({
            "before_names": before_names,
            "after_names": after_names,
            "vertex_unknown": vertex_unknown,
            "tail": tail,
        })
    return post_positions, templates


def _short_string(value: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WZMError(f"WZM 材质名必须是 ASCII：{value}") from exc
    if len(encoded) > 255:
        raise WZMError(f"WZM 材质名过长：{value}")
    return struct.pack("<B", len(encoded)) + encoded


def _encode_weight_reference(
    influences: list[tuple[int, float]],
    groups: list[list[tuple[int, float]]],
    group_indices: dict[tuple[tuple[int, int], ...], int],
) -> int:
    if not influences:
        return 0
    if len(influences) == 1 and abs(influences[0][1] - 1.0) <= 1e-6:
        return int(influences[0][0])
    if len(influences) > 255:
        raise WZMError("单个顶点的骨骼影响超过 255")
    for bone, weight in influences:
        if not 0 <= bone <= 255:
            raise WZMError(f"WZM 骨骼索引超出 u8：{bone}")
        if not math.isfinite(weight):
            raise WZMError("顶点权重包含 NaN/Inf")
    key = tuple((bone, round(weight * 1_000_000_000)) for bone, weight in influences)
    group_index = group_indices.get(key)
    if group_index is None:
        group_index = len(groups)
        if group_index >= 32768:
            raise WZMError("多重权重组超过 WZM i16 引用上限 32768")
        group_indices[key] = group_index
        groups.append(influences)
    return -(group_index + 1)


def _build_chunks(
    model: WZMModel,
    primitives: list[PrimitiveData],
    material_names: dict[int, dict[str, str]] | None = None,
) -> tuple[bytes, bytes, int]:
    if len(primitives) > 255:
        raise WZMError("WZM 子网格数量超过 u8 上限 255")
    post_positions, templates = _mesh_templates(model)
    coordinate_inverse = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    positions: list[tuple[float, float, float]] = []
    position_references: list[int] = []
    encoded_submeshes: list[tuple[PrimitiveData, list[int], int]] = []
    weight_groups: list[list[tuple[int, float]]] = []
    group_indices: dict[tuple[tuple[int, int], ...], int] = {}
    weighted_vertex_count = 0

    for primitive in primitives:
        base = len(positions)
        source_positions = (primitive.positions @ coordinate_inverse.T) * 100.0
        refs: list[int] = []
        for vertex_index, point in enumerate(source_positions):
            if not np.all(np.isfinite(point)):
                raise WZMError("Blender 顶点位置包含 NaN/Inf")
            influences = primitive.weights[vertex_index]
            reference = _encode_weight_reference(influences, weight_groups, group_indices)
            weighted_vertex_count += int(len(influences) > 1)
            positions.append(tuple(float(value) for value in point))
            position_references.append(reference)
            refs.append(reference)
        encoded_submeshes.append((primitive, refs, base))

    mesh_payload = bytearray(struct.pack("<I", len(positions)))
    for reference, point in zip(position_references, positions):
        mesh_payload.extend(struct.pack("<h3f", reference, *point))
    mesh_payload.extend(post_positions)
    mesh_payload.extend(struct.pack("<BI", len(primitives), sum(len(row.positions) for row in primitives)))

    for mesh_index, (primitive, refs, base) in enumerate(encoded_submeshes):
        original = model.submeshes[mesh_index]
        replacement_names = (material_names or {}).get(mesh_index, {})
        diffuse_name = replacement_names.get("diffuse", original.diffuse)
        specular_name = replacement_names.get("specular", original.specular)
        template = templates[mesh_index]
        mesh_payload.extend(template["before_names"])
        mesh_payload.extend(_short_string(diffuse_name))
        mesh_payload.extend(_short_string(specular_name))
        mesh_payload.extend(template["after_names"])
        mesh_payload.extend(struct.pack("<II", len(primitive.positions), len(primitive.indices) // 3))
        source_normals = _normalise_rows(primitive.normals @ coordinate_inverse.T, (0.0, 0.0, 1.0))
        source_tangents = _normalise_rows(primitive.tangents[:, :3] @ coordinate_inverse.T, (1.0, 0.0, 0.0))
        for vertex_index in range(len(primitive.positions)):
            mesh_payload.extend(struct.pack("<hI3f", refs[vertex_index], base + vertex_index, *source_normals[vertex_index]))
            mesh_payload.extend(template["vertex_unknown"])
            mesh_payload.extend(struct.pack("<3f", *source_tangents[vertex_index]))
            mesh_payload.extend(struct.pack("<B", 1 if primitive.tangents[vertex_index, 3] >= 0.0 else 0))
            mesh_payload.extend(struct.pack("<2f", *primitive.uvs[vertex_index]))
        mesh_payload.extend(struct.pack(f"<{len(primitive.indices)}I", *primitive.indices))
        mesh_payload.extend(template["tail"])

    weight_payload = bytearray(struct.pack("<H", len(weight_groups)))
    for group in weight_groups:
        weight_payload.extend(struct.pack("<B", len(group)))
        for bone_index, weight in group:
            weight_payload.extend(struct.pack("<Bf", bone_index, weight))

    mesh_chunk = struct.pack("<HI", CHUNK_MESH, len(mesh_payload) + 6) + mesh_payload
    weight_chunk = struct.pack("<HI", CHUNK_WEIGHTS, len(weight_payload) + 6) + weight_payload
    return mesh_chunk, weight_chunk, weighted_vertex_count


def _replace_chunks(model: WZMModel, replacements: dict[int, bytes]) -> bytes:
    source = model.source.read_bytes()
    output = bytearray()
    cursor = 0
    replaced: set[int] = set()
    for chunk in model.chunks:
        start = chunk["offset"]
        end = start + chunk["size"]
        output.extend(source[cursor:start])
        replacement = replacements.get(chunk["id"])
        if replacement is not None:
            output.extend(replacement)
            replaced.add(chunk["id"])
        else:
            output.extend(source[start:end])
        cursor = end
    output.extend(source[cursor:])
    missing = set(replacements) - replaced
    if missing:
        raise WZMError(f"原 WZM 缺少待替换区块：{sorted(missing)}")
    return bytes(output)


def repack_wzm_from_gltf(
    original_path: str | Path,
    gltf_path: str | Path,
    destination: str | Path,
    material_names: dict[int, dict[str, str]] | None = None,
) -> RepackReport:
    original = parse_wzm(original_path)
    reader = GltfReader(gltf_path)
    primitives, warnings, removed_bones, fallback_weight_vertices = _extract_primitives(reader, original)
    mesh_chunk, weight_chunk, weighted_vertex_count = _build_chunks(
        original, primitives, material_names=material_names,
    )
    rebuilt = _replace_chunks(original, {CHUNK_MESH: mesh_chunk, CHUNK_WEIGHTS: weight_chunk})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(rebuilt)
    try:
        parsed = parse_wzm(temporary)
        if len(parsed.submeshes) != len(primitives):
            raise WZMError("重建后的 WZM 子网格数量校验失败")
        if len(parsed.bones) != len(original.bones):
            raise WZMError("重建后的 WZM 骨骼数量校验失败")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return RepackReport(
        destination=destination,
        vertex_count=sum(len(row.positions) for row in primitives),
        triangle_count=sum(len(row.indices) // 3 for row in primitives),
        submesh_count=len(primitives),
        weighted_vertex_count=weighted_vertex_count,
        normal_count=sum(len(row.normals) for row in primitives),
        removed_bones=removed_bones,
        fallback_weight_vertices=fallback_weight_vertices,
        warnings=warnings,
    )


def rewrite_wzu_texture_names(
    original_path: str | Path,
    destination: str | Path,
    replacements: dict[str, str],
) -> int:
    """Replace length-prefixed texture names while preserving every other WZU field."""
    source = Path(original_path)
    data = source.read_bytes()
    if data[:4] != b"WZUT" or len(data) < 12:
        raise WZMError(f"不是受支持的 WZU 文件：{source}")
    lookup = {old.casefold(): new for old, new in replacements.items() if old and old != new}
    if not lookup:
        Path(destination).write_bytes(data)
        return 0
    known_rows = inspect_wzu_materials(source)
    known_names = {str(row["name"]).casefold() for row in known_rows}
    missing = set(lookup) - known_names
    if missing:
        raise WZMError(f"WZU 中没有找到待替换贴图名：{', '.join(sorted(missing))}")

    pattern = re.compile(rb"[A-Za-z0-9_.-]{3,}\.(?:dds|tga|bmp)", re.I)
    output = bytearray(data[:6])
    cursor = 6
    replaced = 0
    while cursor + 6 <= len(data):
        chunk_id, chunk_size = struct.unpack_from("<HI", data, cursor)
        if chunk_size < 6 or cursor + chunk_size > len(data):
            break
        payload = bytearray(data[cursor + 6:cursor + chunk_size])
        edits: list[tuple[int, int, bytes]] = []
        for length_offset, length in enumerate(payload):
            if length < 3 or length_offset + 1 + length > len(payload):
                continue
            raw_name = bytes(payload[length_offset + 1:length_offset + 1 + length])
            if pattern.fullmatch(raw_name) is None:
                continue
            old_name = raw_name.decode("ascii")
            new_name = lookup.get(old_name.casefold())
            if new_name is None:
                continue
            encoded = _short_string(new_name)
            edits.append((length_offset, length_offset + 1 + length, encoded))
        for start, end, encoded in reversed(edits):
            payload[start:end] = encoded
            replaced += 1
        output.extend(struct.pack("<HI", chunk_id, len(payload) + 6))
        output.extend(payload)
        cursor += chunk_size
    output.extend(data[cursor:])
    if replaced == 0:
        raise WZMError("WZU 没有替换任何贴图引用")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    written_names = {str(row["name"]).casefold() for row in inspect_wzu_materials(destination)}
    expected_names = {name.casefold() for name in lookup.values()}
    if not expected_names.issubset(written_names):
        missing_names = sorted(expected_names - written_names)
        destination.unlink(missing_ok=True)
        raise WZMError(f"WZU 贴图引用写回校验失败，缺少：{', '.join(missing_names)}")
    return replaced

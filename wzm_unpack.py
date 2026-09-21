from __future__ import annotations

"""SUN WZM read-only extractor.

The chunk layout was independently implemented from local samples and the
public format notes embodied by demonsangel's fmt_SUN_wzm.py:
https://github.com/RoadTrain/noesis-plugins-official/blob/master/demonsangel/fmt_SUN_wzm.py

This module never writes to the source directory and does not implement WZM
packing/writing. It exports static OBJ/MTL plus a Blender-ready glTF skin with
materials, skeleton hierarchy and per-vertex bone weights.
"""

import argparse
import json
import math
import re
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


SUPPORTED_VERSIONS = {112, 113, 114, 117, 118, 119, 120}
CHUNK_SKELETON = 47066
CHUNK_MESH = 47057
CHUNK_WEIGHTS = 47058


class WZMError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def tell(self) -> int:
        return self.offset

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def seek(self, offset: int) -> None:
        if not 0 <= offset <= len(self.data):
            raise WZMError(f"读取位置越界：{offset} / {len(self.data)}")
        self.offset = offset

    def skip(self, size: int) -> None:
        self.seek(self.offset + size)

    def read(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise WZMError(f"读取越界：offset={self.offset}, size={size}")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def unpack(self, fmt: str):
        fmt = "<" + fmt
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))

    def u8(self) -> int:
        return self.unpack("B")[0]

    def i16(self) -> int:
        return self.unpack("h")[0]

    def u16(self) -> int:
        return self.unpack("H")[0]

    def u32(self) -> int:
        return self.unpack("I")[0]

    def f32(self) -> float:
        return self.unpack("f")[0]

    def vec(self, count: int) -> tuple[float, ...]:
        return self.unpack(f"{count}f")

    def short_string(self) -> str:
        raw = self.read(self.u8())
        return raw.decode("ascii", errors="replace")


@dataclass(slots=True)
class Bone:
    index: int
    name: str
    parent: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(slots=True)
class Vertex:
    position_index: int
    bone_reference: int
    normal: tuple[float, float, float]
    tangent: tuple[float, float, float]
    uv: tuple[float, float]
    weights: list[tuple[int, float]]


@dataclass(slots=True)
class SubMesh:
    diffuse: str
    specular: str
    vertices: list[Vertex]
    indices: list[int]


@dataclass(slots=True)
class WZMModel:
    source: Path
    version: int
    bones: list[Bone]
    positions: list[tuple[float, float, float]]
    position_bones: list[int]
    submeshes: list[SubMesh]
    extra_weights: list[list[tuple[int, float]]]
    chunks: list[dict[str, int]]


def _read_transform(reader: Reader, version: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if version == 119:
        reader.u16()
    translation = reader.vec(3)
    if version == 112:
        # Old files store Euler radians. Keep the raw values in xyz and mark w=0.
        rotation3 = reader.vec(3)
        rotation = (rotation3[0], rotation3[1], rotation3[2], 0.0)
    else:
        rotation = reader.vec(4)
    return translation, rotation


def _parse_skeleton(reader: Reader, version: int) -> list[Bone]:
    bone_count = reader.u8()
    if version <= 112:
        reader.u8()
    elif version <= 118:
        reader.u16()
    else:
        for _ in range(bone_count):
            reader.u16()
    reader.f32()

    parents: list[int] = []
    names: list[str] = []
    for _ in range(bone_count):
        parents.append(reader.u8())
        names.append(reader.short_string())

    transform_version = version
    if version <= 113:
        transforms = [_read_transform(reader, transform_version) for _ in range(bone_count)]
    else:
        transform_flag = reader.u32()
        if transform_flag == 1 and version in (114, 118):
            reader.skip(4)
        if transform_flag == 0 and version in (117, 119):
            transform_version = 118
        transforms = [_read_transform(reader, transform_version) for _ in range(bone_count)]

    return [
        Bone(
            index=index,
            name=names[index],
            parent=(-1 if parents[index] == 255 else parents[index]),
            translation=transforms[index][0],
            rotation=transforms[index][1],
        )
        for index in range(bone_count)
    ]


def _parse_vertex(reader: Reader, version: int) -> Vertex:
    bone_reference = reader.i16()
    position_index = reader.u32()
    normal = reader.vec(3)
    reader.skip(4 if version <= 114 else 8)
    tangent = reader.vec(3)
    reader.skip(1)
    uv = reader.vec(2)
    return Vertex(position_index, bone_reference, normal, tangent, uv, [])


def _parse_submesh(reader: Reader, version: int) -> SubMesh:
    if version >= 117:
        reader.skip(4)
    diffuse = reader.short_string()
    specular = reader.short_string()
    reader.u32()
    reader.u8()
    vertex_count = reader.u32()
    triangle_count = reader.u32()
    vertices = [_parse_vertex(reader, version) for _ in range(vertex_count)]
    indices = list(reader.unpack(f"{triangle_count * 3}I")) if triangle_count else []
    if version >= 118:
        reader.skip(4)
    return SubMesh(diffuse, specular, vertices, indices)


def _parse_mesh(reader: Reader, version: int) -> tuple[list[tuple[float, float, float]], list[int], list[SubMesh]]:
    position_count = reader.u32()
    positions: list[tuple[float, float, float]] = []
    position_bones: list[int] = []
    for _ in range(position_count):
        position_bones.append(reader.i16())
        positions.append(reader.vec(3))
    if version in (117, 118, 119):
        reader.skip(1)
    submesh_count = reader.u8()
    reader.u32()  # Total vertex count; individual submeshes carry authoritative counts.
    return positions, position_bones, [_parse_submesh(reader, version) for _ in range(submesh_count)]


def _parse_weights(reader: Reader, chunk_size: int) -> list[list[tuple[int, float]]]:
    if chunk_size == 8:
        reader.skip(2)
        return []
    group_count = reader.u16()
    groups: list[list[tuple[int, float]]] = []
    for _ in range(group_count):
        influence_count = reader.u8()
        groups.append([(reader.u8(), reader.f32()) for _ in range(influence_count)])
    return groups


def parse_wzm(path: str | Path) -> WZMModel:
    source = Path(path)
    reader = Reader(source.read_bytes())
    if reader.read(4) != b"WZMD":
        raise WZMError("不是 WZMD 模型文件")
    version = reader.u16()
    if version not in SUPPORTED_VERSIONS:
        raise WZMError(f"暂不支持 WZM 版本 {version}；支持 {sorted(SUPPORTED_VERSIONS)}")
    if version != 120:
        reader.skip(reader.u16())

    bones: list[Bone] = []
    positions: list[tuple[float, float, float]] = []
    position_bones: list[int] = []
    submeshes: list[SubMesh] = []
    extra_weights: list[list[tuple[int, float]]] = []
    chunks: list[dict[str, int]] = []

    while reader.remaining() >= 6:
        chunk_start = reader.tell()
        chunk_id = reader.u16()
        chunk_size = reader.u32()
        chunk_end = chunk_start + chunk_size
        if chunk_size < 6 or chunk_end > len(reader.data):
            raise WZMError(f"损坏的区块：id={chunk_id}, offset={chunk_start}, size={chunk_size}")
        chunks.append({"id": chunk_id, "offset": chunk_start, "size": chunk_size})
        if chunk_id == CHUNK_SKELETON:
            bones = _parse_skeleton(reader, version)
        elif chunk_id == CHUNK_MESH:
            positions, position_bones, submeshes = _parse_mesh(reader, version)
        elif chunk_id == CHUNK_WEIGHTS:
            extra_weights = _parse_weights(reader, chunk_size)
        reader.seek(chunk_end)

    if not positions or not submeshes:
        raise WZMError("文件中没有可导出的网格区块")

    for submesh in submeshes:
        for vertex in submesh.vertices:
            if not 0 <= vertex.position_index < len(positions):
                raise WZMError(f"顶点引用越界：{vertex.position_index} / {len(positions)}")
            if vertex.bone_reference >= 0:
                vertex.weights = [(vertex.bone_reference, 1.0)]
            else:
                weight_index = abs(vertex.bone_reference) - 1
                if not 0 <= weight_index < len(extra_weights):
                    raise WZMError(f"权重引用越界：{weight_index} / {len(extra_weights)}")
                vertex.weights = extra_weights[weight_index]
        if submesh.indices and max(submesh.indices) >= len(submesh.vertices):
            raise WZMError("三角形索引超出子网格顶点范围")

    return WZMModel(source, version, bones, positions, position_bones, submeshes, extra_weights, chunks)


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or fallback


def _find_case_insensitive(directory: Path, name: str) -> Path | None:
    target = name.casefold()
    for path in directory.iterdir():
        if path.is_file() and path.name.casefold() == target:
            return path
    return None


def _export_texture(source_dir: Path, texture_name: str, texture_dir: Path) -> str | None:
    source = _find_case_insensitive(source_dir, Path(texture_name).name)
    if source is None:
        return None
    texture_dir.mkdir(parents=True, exist_ok=True)
    if Image is not None and source.suffix.casefold() in (".dds", ".tga", ".bmp"):
        destination = texture_dir / f"{source.stem}.png"
        try:
            with Image.open(source) as image:
                image.convert("RGBA").save(destination)
            return f"textures/{destination.name}"
        except Exception:
            pass
    destination = texture_dir / source.name
    shutil.copy2(source, destination)
    return f"textures/{destination.name}"


def _texture_role(name: str) -> str:
    stem = Path(name).stem.casefold()
    if re.search(r"(?:^|[_-])no(?:$|[_-])", stem):
        return "normal"
    if re.search(r"(?:^|[_-])sp(?:$|[_-])", stem):
        return "specular"
    if "glow" in stem or "emiss" in stem:
        return "glow"
    if "var_di" in stem or "variant" in stem:
        return "diffuse_variant"
    if "di_h" in stem or stem.endswith("_di") or "diffuse" in stem:
        return "diffuse"
    if "internal" in stem or "inner" in stem:
        return "internal"
    if stem.startswith("test"):
        return "effect_auxiliary"
    return "unknown"


def inspect_wzu_materials(path: str | Path) -> list[dict[str, str | int]]:
    """List WZU texture references and classify SUN's conventional roles."""
    source = Path(path)
    data = source.read_bytes()
    rows: list[dict[str, str | int]] = []
    seen: set[tuple[int, str]] = set()
    pattern = re.compile(rb"[A-Za-z0-9_.-]{3,}\.(?:dds|tga|bmp)", re.I)
    for match in pattern.finditer(data):
        name = match.group().decode("ascii", errors="replace")
        key = (match.start(), name.casefold())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"offset": match.start(), "name": name, "role": _texture_role(name)})
    return rows


def _normalise_quaternion(quaternion: tuple[float, float, float, float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in quaternion))
    if length < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [value / length for value in quaternion]


def _bone_rotation(model: WZMModel, bone: Bone) -> list[float]:
    """Convert SUN's bind rotation to the local quaternion used by glTF."""
    x, y, z, w = bone.rotation
    if model.version == 112:
        # The old format stores XYZ Euler radians. This mirrors the conversion
        # used by the public Noesis importer before it creates the bind bone.
        cx, cy, cz = math.cos(x / 2), math.cos(y / 2), math.cos(z / 2)
        sx, sy, sz = math.sin(x / 2), math.sin(y / 2), math.sin(z / 2)
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
        qw = cx * cy * cz - sx * sy * sz
        return _normalise_quaternion((qx, -qy, qz, -qw))
    # Noesis Mat43 uses a different row/column convention. After converting
    # that bind matrix to glTF's column-vector convention, the stored WZM
    # quaternion is the local joint rotation (conjugating it turns the whole
    # humanoid skeleton sideways and separates it from the equipment).
    return _normalise_quaternion((x, y, z, w))


def export_gltf(model: WZMModel, output_dir: Path, materials: list[dict[str, str | None]]) -> Path:
    """Export a glTF 2.0 skin that Blender imports as mesh + armature + weights."""
    try:
        import numpy as np
    except ImportError as exc:
        raise WZMError("缺少 numpy，无法导出带骨骼的 glTF") from exc

    binary = bytearray()
    buffer_views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []

    def add_accessor(
        payload: bytes,
        component_type: int,
        accessor_type: str,
        count: int,
        *,
        target: int | None = None,
        minimum: list[float] | None = None,
        maximum: list[float] | None = None,
    ) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        view: dict[str, object] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        view_index = len(buffer_views)
        buffer_views.append(view)
        accessor: dict[str, object] = {
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if minimum is not None:
            accessor["min"] = minimum
        if maximum is not None:
            accessor["max"] = maximum
        accessors.append(accessor)
        return len(accessors) - 1

    images: list[dict[str, object]] = []
    textures: list[dict[str, object]] = []
    image_indices: dict[str, int] = {}
    gltf_materials: list[dict[str, object]] = []
    extensions_used: set[str] = set()

    def ensure_image(uri: str) -> int:
        if uri not in image_indices:
            image_indices[uri] = len(images)
            images.append({"uri": uri, "name": Path(uri).stem})
            textures.append({"sampler": 0, "source": image_indices[uri]})
        return image_indices[uri]

    for material in materials:
        pbr: dict[str, object] = {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.72,
        }
        diffuse = material.get("diffuse")
        if diffuse:
            pbr["baseColorTexture"] = {"index": ensure_image(diffuse), "texCoord": 0}
        gltf_material: dict[str, object] = {
            "name": material["name"],
            "pbrMetallicRoughness": pbr,
            "alphaMode": "MASK",
            "alphaCutoff": 0.04,
            "doubleSided": True,
            "extras": {
                "SUN_specularTexture": material.get("specular"),
                "SUN_normalTexture": material.get("normal"),
                "SUN_glowTexture": material.get("glow"),
                "SUN_diffuseVariantTexture": material.get("diffuse_variant"),
                "SUN_internalTexture": material.get("internal"),
                "SUN_effectAuxiliaryTexture": material.get("effect_auxiliary"),
            },
        }
        normal = material.get("normal")
        if normal:
            gltf_material["normalTexture"] = {"index": ensure_image(normal), "scale": 1.0}
        specular = material.get("specular")
        if specular:
            specular_index = ensure_image(specular)
            gltf_material["extensions"] = {
                "KHR_materials_specular": {
                    "specularFactor": 1.0,
                    "specularColorFactor": [1.0, 1.0, 1.0],
                    "specularColorTexture": {"index": specular_index},
                }
            }
            extensions_used.add("KHR_materials_specular")
        gltf_materials.append(gltf_material)

    primitives: list[dict[str, object]] = []
    unit_scale = 0.01  # SUN positions are centimetre-like; glTF/Blender use metres.
    coordinate = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    for mesh_index, submesh in enumerate(model.submeshes):
        source_positions = np.asarray([
            model.positions[vertex.position_index] for vertex in submesh.vertices
        ], dtype=np.float64)
        positions = ((source_positions @ coordinate.T) * unit_scale).astype("<f4")
        source_normals = np.asarray([vertex.normal for vertex in submesh.vertices], dtype=np.float64)
        normals = (source_normals @ coordinate.T).astype("<f4")
        source_tangents = np.asarray([vertex.tangent for vertex in submesh.vertices], dtype=np.float64)
        tangent_xyz = (source_tangents @ coordinate.T).astype("<f4")
        tangents = np.column_stack((tangent_xyz, np.ones(len(tangent_xyz), dtype="<f4"))).astype("<f4")
        uvs = np.asarray([vertex.uv for vertex in submesh.vertices], dtype="<f4")
        indices = np.asarray(submesh.indices, dtype="<u4")

        attributes: dict[str, int] = {
            "POSITION": add_accessor(
                positions.tobytes(), 5126, "VEC3", len(positions), target=34962,
                minimum=positions.min(axis=0).astype(float).tolist(),
                maximum=positions.max(axis=0).astype(float).tolist(),
            ),
            "NORMAL": add_accessor(normals.tobytes(), 5126, "VEC3", len(normals), target=34962),
            "TANGENT": add_accessor(tangents.tobytes(), 5126, "VEC4", len(tangents), target=34962),
            # glTF uses the same top-left texture origin as SUN/DirectX.
            "TEXCOORD_0": add_accessor(uvs.tobytes(), 5126, "VEC2", len(uvs), target=34962),
        }

        if model.bones:
            joints = np.zeros((len(submesh.vertices), 4), dtype="<u2")
            weights = np.zeros((len(submesh.vertices), 4), dtype="<f4")
            for vertex_index, vertex in enumerate(submesh.vertices):
                valid = [(bone, weight) for bone, weight in vertex.weights if 0 <= bone < len(model.bones) and weight > 0]
                valid.sort(key=lambda pair: pair[1], reverse=True)
                valid = valid[:4] or [(0, 1.0)]
                total = sum(weight for _bone, weight in valid) or 1.0
                for influence_index, (bone, weight) in enumerate(valid):
                    joints[vertex_index, influence_index] = bone
                    weights[vertex_index, influence_index] = weight / total
            attributes["JOINTS_0"] = add_accessor(joints.tobytes(), 5123, "VEC4", len(joints), target=34962)
            attributes["WEIGHTS_0"] = add_accessor(weights.tobytes(), 5126, "VEC4", len(weights), target=34962)

        primitives.append({
            "attributes": attributes,
            "indices": add_accessor(indices.tobytes(), 5125, "SCALAR", len(indices), target=34963),
            "material": mesh_index,
            "mode": 4,
        })

    nodes: list[dict[str, object]] = []
    local_matrices: list[object] = []
    for bone in model.bones:
        rotation = _bone_rotation(model, bone)
        x, y, z, w = rotation
        source_matrix = np.eye(4, dtype=np.float64)
        source_matrix[:3, :3] = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        source_matrix[:3, 3] = np.asarray(bone.translation, dtype=np.float64) * unit_scale
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = coordinate @ source_matrix[:3, :3] @ coordinate.T
        matrix[:3, 3] = coordinate @ source_matrix[:3, 3]
        node: dict[str, object] = {
            "name": bone.name or f"bone_{bone.index}",
            "matrix": matrix.T.reshape(16).astype(float).tolist(),
        }
        nodes.append(node)
        local_matrices.append(matrix)

    root_bones: list[int] = []
    for bone in model.bones:
        if 0 <= bone.parent < len(model.bones) and bone.parent != bone.index:
            nodes[bone.parent].setdefault("children", []).append(bone.index)
        else:
            root_bones.append(bone.index)

    skin_index: int | None = None
    skins: list[dict[str, object]] = []
    if model.bones:
        global_matrices: list[object | None] = [None] * len(model.bones)
        visiting: set[int] = set()

        def global_matrix(index: int):
            cached = global_matrices[index]
            if cached is not None:
                return cached
            if index in visiting:
                return local_matrices[index]
            visiting.add(index)
            parent = model.bones[index].parent
            if 0 <= parent < len(model.bones) and parent != index:
                result = global_matrix(parent) @ local_matrices[index]
            else:
                result = local_matrices[index]
            visiting.discard(index)
            global_matrices[index] = result
            return result

        inverse_bind = np.asarray([
            np.linalg.inv(global_matrix(index)).T.reshape(16) for index in range(len(model.bones))
        ], dtype="<f4")
        inverse_accessor = add_accessor(inverse_bind.tobytes(), 5126, "MAT4", len(model.bones))
        skins.append({
            "name": f"{model.source.stem}_Armature",
            "inverseBindMatrices": inverse_accessor,
            "joints": list(range(len(model.bones))),
            "skeleton": root_bones[0] if root_bones else 0,
        })
        skin_index = 0

    mesh_node_index = len(nodes)
    mesh_node: dict[str, object] = {"name": model.source.stem, "mesh": 0}
    if skin_index is not None:
        mesh_node["skin"] = skin_index
    nodes.append(mesh_node)

    bin_path = output_dir / f"{model.source.stem}.bin"
    gltf_path = output_dir / f"{model.source.stem}.gltf"
    bin_path.write_bytes(binary)
    document: dict[str, object] = {
        "asset": {
            "version": "2.0",
            "generator": "SUN GM Tool WZM extractor",
            "extras": {"source": str(model.source), "wzmVersion": model.version},
        },
        "scene": 0,
        "scenes": [{"name": "SUN model", "nodes": [*root_bones, mesh_node_index]}],
        "nodes": nodes,
        "meshes": [{"name": model.source.stem, "primitives": primitives}],
        "buffers": [{"uri": bin_path.name, "byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": gltf_materials,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
    }
    if extensions_used:
        document["extensionsUsed"] = sorted(extensions_used)
    if images:
        document["images"] = images
        document["textures"] = textures
    if skins:
        document["skins"] = skins
    gltf_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return gltf_path


def export_wzm(model: WZMModel, output_root: str | Path, companion_unit: str | Path | None = None) -> Path:
    output_dir = Path(output_root) / model.source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / f"{model.source.stem}.obj"
    mtl_path = output_dir / f"{model.source.stem}.mtl"
    metadata_path = output_dir / f"{model.source.stem}.json"
    weights_path = output_dir / f"{model.source.stem}.weights.json"
    texture_dir = output_dir / "textures"

    unit_path = Path(companion_unit) if companion_unit else model.source.with_suffix(".WZU")
    unit_textures = inspect_wzu_materials(unit_path) if unit_path.is_file() else []
    role_sources: dict[str, str] = {}
    for row in unit_textures:
        role = str(row["role"])
        if role not in role_sources:
            role_sources[role] = str(row["name"])

    materials: list[dict[str, str | None]] = []
    for index, submesh in enumerate(model.submeshes):
        materials.append({
            "name": _safe_name(submesh.diffuse, f"material_{index}"),
            "diffuse": _export_texture(model.source.parent, submesh.diffuse, texture_dir),
            "specular": _export_texture(model.source.parent, submesh.specular, texture_dir),
            "normal": _export_texture(model.source.parent, role_sources["normal"], texture_dir) if "normal" in role_sources else None,
            "glow": _export_texture(model.source.parent, role_sources["glow"], texture_dir) if "glow" in role_sources else None,
            "diffuse_variant": _export_texture(model.source.parent, role_sources["diffuse_variant"], texture_dir) if "diffuse_variant" in role_sources else None,
            "internal": _export_texture(model.source.parent, role_sources["internal"], texture_dir) if "internal" in role_sources else None,
            "effect_auxiliary": _export_texture(model.source.parent, role_sources["effect_auxiliary"], texture_dir) if "effect_auxiliary" in role_sources else None,
        })

    with mtl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for material in materials:
            stream.write(f"newmtl {material['name']}\nKd 1.0 1.0 1.0\n")
            if material["diffuse"]:
                stream.write(f"map_Kd {material['diffuse']}\n")
            if material["specular"]:
                stream.write(f"map_Ks {material['specular']}\n")
            stream.write("\n")

    obj_vertex_offset = 1
    exported_weights: list[dict[str, object]] = []
    with obj_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# SUN WZM v{model.version} read-only export\n")
        stream.write(f"mtllib {mtl_path.name}\n")
        for mesh_index, submesh in enumerate(model.submeshes):
            stream.write(f"\ng submesh_{mesh_index}\nusemtl {materials[mesh_index]['name']}\n")
            for vertex in submesh.vertices:
                x, y, z = model.positions[vertex.position_index]
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for vertex in submesh.vertices:
                u, v = vertex.uv
                stream.write(f"vt {u:.9g} {1.0 - v:.9g}\n")
            for vertex in submesh.vertices:
                x, y, z = vertex.normal
                stream.write(f"vn {x:.9g} {y:.9g} {z:.9g}\n")
            for index in range(0, len(submesh.indices), 3):
                face = [submesh.indices[index + n] + obj_vertex_offset for n in range(3)]
                stream.write("f " + " ".join(f"{value}/{value}/{value}" for value in face) + "\n")
            exported_weights.append({
                "submesh": mesh_index,
                "obj_vertex_start": obj_vertex_offset,
                "weights": [vertex.weights for vertex in submesh.vertices],
            })
            obj_vertex_offset += len(submesh.vertices)

    metadata = {
        "source": str(model.source),
        "companion_wzu": str(unit_path) if unit_path.is_file() else None,
        "wzu_texture_roles": unit_textures,
        "version": model.version,
        "bone_count": len(model.bones),
        "position_count": len(model.positions),
        "submesh_count": len(model.submeshes),
        "triangle_count": sum(len(mesh.indices) // 3 for mesh in model.submeshes),
        "chunks": model.chunks,
        "bones": [asdict(bone) for bone in model.bones],
        "submeshes": [
            {
                "index": index,
                "diffuse": mesh.diffuse,
                "specular": mesh.specular,
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.indices) // 3,
            }
            for index, mesh in enumerate(model.submeshes)
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    weights_path.write_text(
        json.dumps({"bones": [bone.name for bone in model.bones], "meshes": exported_weights}, ensure_ascii=False),
        encoding="utf-8",
    )
    export_gltf(model, output_dir, materials)
    return output_dir


def render_preview(model: WZMModel, destination: str | Path, azimuth: float = -70, elevation: float = 12) -> Path:
    """Render a static shaded mesh preview. This is geometry-only, not a game-accurate material renderer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise WZMError("缺少 matplotlib/numpy，无法生成三维预览") from exc

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    vertex_offset = 0
    for submesh in model.submeshes:
        vertices.extend(model.positions[vertex.position_index] for vertex in submesh.vertices)
        triangles.extend(
            (
                submesh.indices[index] + vertex_offset,
                submesh.indices[index + 1] + vertex_offset,
                submesh.indices[index + 2] + vertex_offset,
            )
            for index in range(0, len(submesh.indices), 3)
        )
        vertex_offset += len(submesh.vertices)
    if not vertices or not triangles:
        raise WZMError("模型没有可渲染的三角形")

    points = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    figure = plt.figure(figsize=(6, 6), dpi=100, facecolor="#25282d")
    axis = figure.add_subplot(111, projection="3d", facecolor="#25282d")
    axis.plot_trisurf(
        points[:, 0], points[:, 1], points[:, 2],
        triangles=faces,
        color="#93a7c5",
        edgecolor="#26313f",
        linewidth=0.08,
        antialiased=True,
        shade=True,
    )
    spans = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-5)
    axis.set_box_aspect(tuple(spans))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return destination


def unpack_wzm(source: str | Path, output_root: str | Path, companion_unit: str | Path | None = None) -> tuple[WZMModel, Path]:
    model = parse_wzm(source)
    return model, export_wzm(model, output_root, companion_unit=companion_unit)


def main() -> int:
    parser = argparse.ArgumentParser(description="SUN WZM 只读解包器：导出 OBJ 与带骨骼权重的 glTF")
    parser.add_argument("source", type=Path, help="WZM 文件")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "exports")
    parser.add_argument("--wzu", type=Path, help="配套 WZU；用于识别 normal/specular/glow 等额外贴图")
    args = parser.parse_args()
    try:
        model, output_dir = unpack_wzm(args.source, args.output, args.wzu)
        render_preview(model, output_dir / "preview.png")
    except (OSError, WZMError) as exc:
        parser.error(str(exc))
    print(f"版本: {model.version}")
    print(f"骨骼: {len(model.bones)}")
    print(f"子网格: {len(model.submeshes)}")
    print(f"三角形: {sum(len(mesh.indices) // 3 for mesh in model.submeshes)}")
    print(f"输出: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

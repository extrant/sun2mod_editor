from __future__ import annotations

"""Read-only SUN effect resource resolver and EWZ dependency inspector.

This intentionally does not pretend that finding a DDS equals reproducing the
effect. EWZ also contains emitter timing, transforms, colours, blend state and
element-type data that must be decoded before game-accurate rendering.
"""

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EFFECT_DIR = Path(r"D:\sun\客户端\Data\Effect")
RESOURCE_FILES = (
    "effectResource.txt",
    "effectResource0.txt",
    "effectResource1.txt",
    "effectResource2.txt",
    "effectResource3.txt",
    "EffectResource_BG.txt",
    "D_EffectResource.txt",
    "DeleteEffectResource.txt",
)
TEXTURE_FILES = ("Effect.txt", "EffectTexture_BG.txt")
MESH_FILES = ("Effect-Mesh.txt", "Effect-Mesh_BG.txt")
ROW_RE = re.compile(r'^\s*([A-Za-z0-9_]{4})\s+"([^"]+)"')
TOKEN_RE = re.compile(rb"[A-Za-z0-9_]{4}")


@dataclass(slots=True)
class MappedResource:
    code: str
    declared_path: str
    table: str
    actual_path: str | None


@dataclass(slots=True)
class EWZInspection:
    source: str
    version: str
    magic: str
    textures: list[MappedResource]
    meshes: list[MappedResource]
    nested_effects: list[MappedResource]
    unresolved_tokens: list[str]
    structure: dict[str, Any]


@dataclass(slots=True)
class WZUEffectAttachment:
    code: str
    bone_index: int
    position: list[float]
    rotation: list[float]
    scale: list[float]
    apply_rotation: int
    offset: int


class EWZFormatError(ValueError):
    pass


class BinaryReader:
    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.offset = offset

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise EWZFormatError(f"读取越界：0x{self.offset:X} + {size}，文件长度 0x{len(self.data):X}")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def vec3(self) -> list[float]:
        return list(struct.unpack("<3f", self.take(12)))


def version_key(version: str) -> tuple[int, ...]:
    match = re.search(r"e(\d+)(?:\.(\d+))?", version)
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2) or 0)


def fourcc(value: int) -> str:
    raw = struct.pack("<I", value)
    return raw.decode("ascii", errors="replace").rstrip("\0")


def read_curve(reader: BinaryReader) -> dict[str, Any]:
    count = reader.u32()
    if count > 100_000:
        raise EWZFormatError(f"异常曲线关键帧数量 {count}，位置 0x{reader.offset - 4:X}")
    keys = [{"time": reader.u32(), "value": reader.f32()} for _ in range(count)]
    result: dict[str, Any] = {"key_count": count}
    if keys:
        result["first"] = keys[0]
        result["last"] = keys[-1]
        if count <= 16:
            result["keys"] = keys
    return result


def read_timed_vectors(reader: BinaryReader) -> dict[str, Any]:
    count = reader.u32()
    if count > 100_000:
        raise EWZFormatError(f"异常向量关键帧数量 {count}，位置 0x{reader.offset - 4:X}")
    keys = [{"time": reader.u32(), "value": reader.vec3()} for _ in range(count)]
    result: dict[str, Any] = {"key_count": count}
    if keys:
        result["first"] = keys[0]
        result["last"] = keys[-1]
        if count <= 16:
            result["keys"] = keys
    return result


def read_optional_ranges(reader: BinaryReader, count: int) -> list[dict[str, Any]]:
    result = []
    for _ in range(count):
        enabled = bool(reader.u8())
        item: dict[str, Any] = {"enabled": enabled}
        if enabled:
            item["minimum"] = reader.f32()
            item["maximum"] = reader.f32()
        result.append(item)
    return result


def parse_create(reader: BinaryReader, version: tuple[int, ...]) -> dict[str, Any]:
    start = reader.offset
    result: dict[str, Any] = {
        "duration_ms": reader.u32(),
        "amount": reader.u32(),
        "shape_type": reader.u32(),
    }
    if version >= (2, 93):
        result["v293_fields"] = [reader.u32(), reader.u32(), reader.u32()]

    shape = result["shape_type"]
    if shape == 0:
        result["point"] = reader.vec3()
    elif shape == 1:
        result["minimum"] = reader.vec3()
        result["maximum"] = reader.vec3()
    elif shape == 2:
        result["radius"] = reader.f32()
        result["direction"] = reader.vec3()
    elif shape == 3:
        result["control_points"] = read_timed_vectors(reader)
        if version > (1, 8):
            result["minimum"] = reader.vec3()
            result["maximum"] = reader.vec3()
    else:
        raise EWZFormatError(f"未知生成形状 {shape}，位置 0x{start:X}")

    sequence_count = reader.u32()
    if sequence_count > 100_000:
        raise EWZFormatError(f"异常生成序列数量 {sequence_count}，位置 0x{reader.offset - 4:X}")
    sequence = [reader.u32() for _ in range(sequence_count)]
    result["sequence_count"] = sequence_count
    if sequence_count <= 64:
        result["sequence"] = sequence
    result["active"] = bool(reader.u8())
    result["random_ranges"] = read_optional_ranges(reader, 5)
    result["offset"] = start
    result["size"] = reader.offset - start
    return result


def parse_move(reader: BinaryReader) -> dict[str, Any]:
    start = reader.offset
    result: dict[str, Any] = {
        "mode": reader.u8(),
        "rotation": [reader.f32(), reader.f32(), reader.f32()],
        "use_acceleration": bool(reader.u8()),
        "use_fixed_motion": bool(reader.u8()),
    }
    if result["use_fixed_motion"]:
        result["velocity"] = reader.vec3()
        result["speed"] = reader.f32()
        if result["use_acceleration"]:
            result["acceleration_direction"] = reader.vec3()
            result["acceleration"] = reader.f32()
        result["motion_scalar"] = reader.f32()
        result["motion_vector"] = reader.vec3()
        result["motion_mode"] = reader.u32()
        result["target"] = reader.vec3()
    else:
        result["path"] = read_timed_vectors(reader)
        result["target"] = reader.vec3()
        result["secondary_path"] = read_timed_vectors(reader)
    result["random_ranges"] = read_optional_ranges(reader, 7)
    result["offset"] = start
    result["size"] = reader.offset - start
    return result


def parse_change(reader: BinaryReader, version: tuple[int, ...]) -> dict[str, Any]:
    start = reader.offset
    channels = [read_curve(reader) for _ in range(16)]
    extra_curve = read_curve(reader) if version >= (2, 92) else None
    extra_mode = reader.u32() if version >= (2, 92) else None
    interpolation_modes = [reader.u32() for _ in range(16)]
    record_mode = reader.u32()
    result = {
        "offset": start,
        "size": reader.offset - start,
        "channels": channels,
        "interpolation_modes": interpolation_modes,
        "record_mode": record_mode,
    }
    if extra_curve is not None:
        result["v292_curve"] = extra_curve
        result["v292_mode"] = extra_mode
    return result


def parse_visual(reader: BinaryReader, version: tuple[int, ...]) -> dict[str, Any]:
    start = reader.offset
    result: dict[str, Any] = {"render_type": reader.u32(), "resource_type": reader.u32()}
    resource_type = result["resource_type"]
    if resource_type == 0:
        pass
    elif resource_type == 1:
        code = reader.u32()
        result["resource_code"] = fourcc(code)
        result["resource_scale"] = reader.f32()
        animated = bool(reader.u8())
        result["animated"] = animated
        if animated:
            result["animation_fields"] = [reader.u32() for _ in range(4)]
            result["animation_flag"] = bool(reader.u8())
            result["animation_rate"] = reader.f32()
            if version >= (2, 5):
                result["animation_mode"] = reader.u8()
        result["blend_mode"] = reader.u32()
        result["facing"] = reader.vec3()
        result["billboard_mode"] = reader.u8()
        result["quad_vertices"] = list(struct.unpack("<12f", reader.take(48)))
        result["pivot"] = [reader.f32(), reader.f32(), reader.f32()]
        result["uv_mode"] = reader.u8()
        if version >= (3, 0):
            result["v300_block_a"] = reader.take(28).hex()
            result["v300_block_b"] = reader.take(20).hex()
    elif resource_type == 2:
        result["resource_code"] = fourcc(reader.u32())
        result["frame_count"] = reader.u32()
        result["frame_time"] = reader.f32()
    elif resource_type == 3:
        result["resource_code"] = fourcc(reader.u32())
        result["axis_primary"] = reader.vec3()
        result["axis_secondary"] = reader.vec3()
        result["length"] = reader.f32()
        result["width"] = reader.f32()
        result["chain_mode"] = reader.u8()
    elif resource_type == 4:
        result["resource_code"] = fourcc(reader.u32())
        result["direction"] = reader.vec3()
        result["segment_count"] = reader.u32()
    elif resource_type == 5:
        result["resource_code"] = fourcc(reader.u32())
        result["mesh_mode"] = reader.u32()
        result["mesh_scale"] = reader.f32()
        result["mesh_flags"] = reader.u32()
        if version >= (3, 0):
            result["v300_flags"] = [reader.u8(), reader.u8()]
            result["v300_mesh_block"] = reader.take(20).hex()
    else:
        raise EWZFormatError(f"尚未映射的外观资源类型 {resource_type}，位置 0x{start:X}")

    result["random_size"] = bool(reader.u8())
    if result["random_size"]:
        result["minimum_size"] = reader.f32()
        result["maximum_size"] = reader.f32()
    result["offset"] = start
    result["size"] = reader.offset - start
    return result


def parse_ewz_structure(data: bytes, version_text: str) -> dict[str, Any]:
    version = version_key(version_text)
    if version < (2, 3):
        return {"supported": False, "reason": "结构解析当前覆盖 ver.e2.3 及更新版本"}
    reader = BinaryReader(data, 14)
    result: dict[str, Any] = {
        "supported": True,
        "effect_id": data[10:14].decode("ascii", errors="replace"),
        "global_scale": reader.f32(),
        "body_flag": reader.u8(),
        "playback_rate": reader.f32(),
    }
    try:
        element_count = reader.u16()
        elements = []
        for index in range(element_count):
            start = reader.offset
            element = {
                "index": index,
                "create_index": reader.u16(),
                "move_index": reader.u16(),
                "change_index": reader.u16(),
                "visual_index": reader.u16(),
                "flag_0": reader.u8(),
                "time_0": reader.u32(),
                "flag_1": reader.u8(),
                "time_1": reader.u32(),
                "flag_2": reader.u8(),
                "link_index": reader.u16(),
                "flag_3": reader.u8(),
                "offset": start,
                "size": reader.offset - start,
            }
            elements.append(element)
        result["elements"] = elements

        create_count = reader.u16()
        result["creates"] = [parse_create(reader, version) for _ in range(create_count)]
        move_count = reader.u16()
        result["moves"] = [parse_move(reader) for _ in range(move_count)]
        change_count = reader.u16()
        result["changes"] = [parse_change(reader, version) for _ in range(change_count)]
        visual_count = reader.u16()
        result["visuals"] = []
        for _ in range(visual_count):
            result["visuals"].append(parse_visual(reader, version))

        result["body_extent_scale"] = reader.f32()
        result["bounds_minimum"] = reader.vec3()
        result["bounds_maximum"] = reader.vec3()
        result["parsed_bytes"] = reader.offset
        result["trailing_bytes"] = len(data) - reader.offset
    except EWZFormatError as exc:
        result["supported"] = False
        result["parse_error"] = str(exc)
        result["parsed_bytes"] = reader.offset
        result["trailing_bytes"] = len(data) - reader.offset
    return result


class EffectCatalog:
    def __init__(self, root: Path = EFFECT_DIR):
        self.root = root
        self.effects = self._load_tables(RESOURCE_FILES)
        self.textures = self._load_tables(TEXTURE_FILES)
        self.meshes = self._load_tables(MESH_FILES)
        self._wzu_cache: dict[str, list[WZUEffectAttachment]] = {}
        self._files_by_name: dict[str, list[Path]] = {}
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    self._files_by_name.setdefault(path.name.casefold(), []).append(path)

    def _load_tables(self, names: tuple[str, ...]) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {}
        for name in names:
            path = self.root / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="cp949", errors="ignore").splitlines():
                match = ROW_RE.match(line)
                if match:
                    # Effect identifiers are case-sensitive: A003 is an EWZ
                    # effect while a003 is a DDS texture in the same install.
                    result.setdefault(match.group(1), []).append((match.group(2), name))
        return result

    def _locate(self, declared: str) -> Path | None:
        normal = declared.replace("/", "\\").lstrip("\\")
        direct = self.root.joinpath(*[part for part in normal.split("\\") if part])
        if direct.is_file():
            return direct
        candidates = self._files_by_name.get(Path(normal).name.casefold(), [])
        return candidates[0] if len(candidates) == 1 else None

    def _mapped(self, code: str, table: dict[str, list[tuple[str, str]]]) -> list[MappedResource]:
        rows: list[MappedResource] = []
        seen: set[tuple[str, str | None]] = set()
        for declared, source_table in table.get(code, []):
            actual = self._locate(declared)
            key = (declared.replace("/", "\\").casefold(), str(actual).casefold() if actual else None)
            if key in seen:
                continue
            seen.add(key)
            rows.append(MappedResource(code, declared, source_table, str(actual) if actual else None))
        return rows

    def resolve_effect(self, code: str) -> list[MappedResource]:
        return self._mapped(code, self.effects)

    def inspect_ewz(self, path: str | Path) -> EWZInspection:
        source = Path(path)
        data = source.read_bytes()
        version = data[:10].split(b"\0", 1)[0].decode("ascii", errors="replace")
        magic = data[10:14].decode("ascii", errors="replace")
        tokens = sorted({match.decode("ascii") for match in TOKEN_RE.findall(data)})
        texture_rows: list[MappedResource] = []
        mesh_rows: list[MappedResource] = []
        nested_rows: list[MappedResource] = []
        unresolved: list[str] = []
        for token in tokens:
            textures = self._mapped(token, self.textures)
            meshes = self._mapped(token, self.meshes)
            effects = self._mapped(token, self.effects)
            if textures or meshes or effects:
                texture_rows.extend(textures)
                mesh_rows.extend(meshes)
                nested_rows.extend(effects)
            elif token not in ("WZE0",):
                unresolved.append(token)
        return EWZInspection(
            source=str(source),
            version=version,
            magic=magic,
            textures=texture_rows,
            meshes=mesh_rows,
            nested_effects=nested_rows,
            unresolved_tokens=unresolved,
            structure=parse_ewz_structure(data, version),
        )

    def inspect_code(self, code: str) -> list[EWZInspection]:
        result: list[EWZInspection] = []
        for resource in self.resolve_effect(code):
            if resource.actual_path and Path(resource.actual_path).suffix.casefold() == ".ewz":
                result.append(self.inspect_ewz(resource.actual_path))
        return result

    def inspect_wzu(self, path: str | Path) -> list[WZUEffectAttachment]:
        """Read serialized type-2 WzUnit effect nodes.

        The client stores these records as type:u8, bone:u16, code:char[5],
        position:vec3, rotation:quat, scale:vec3, apply_rotation:u32.
        """
        source = Path(path)
        cache_key = str(source.resolve()).casefold()
        if cache_key in self._wzu_cache:
            return list(self._wzu_cache[cache_key])
        data = source.read_bytes()
        result: list[WZUEffectAttachment] = []
        for offset in range(0, max(len(data) - 51, 0)):
            if data[offset] != 2 or data[offset + 7] != 0:
                continue
            try:
                code = data[offset + 3:offset + 7].decode("ascii")
            except UnicodeDecodeError:
                continue
            if code not in self.effects:
                continue
            values = struct.unpack_from("<3f4f3fI", data, offset + 8)
            floats = values[:-1]
            if not all(math.isfinite(value) and abs(value) < 1_000_000 for value in floats):
                continue
            result.append(WZUEffectAttachment(
                code=code,
                bone_index=struct.unpack_from("<H", data, offset + 1)[0],
                position=list(values[0:3]),
                rotation=list(values[3:7]),
                scale=list(values[7:10]),
                apply_rotation=int(values[10]),
                offset=offset,
            ))
        self._wzu_cache[cache_key] = result
        return list(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="SUN EWZ 只读依赖检查器")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--code", help="effectResource 表中的四字符效果编号")
    group.add_argument("--file", type=Path, help="直接检查 EWZ 文件")
    parser.add_argument("--root", type=Path, default=EFFECT_DIR)
    args = parser.parse_args()
    catalog = EffectCatalog(args.root)
    if args.code:
        payload = {
            "code": args.code,
            "candidates": [asdict(row) for row in catalog.resolve_effect(args.code)],
            "inspections": [asdict(row) for row in catalog.inspect_code(args.code)],
        }
    else:
        payload = asdict(catalog.inspect_ewz(args.file))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

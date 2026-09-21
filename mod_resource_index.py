from __future__ import annotations

"""Build and cache a virtual filesystem for SUN client resources."""

import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Callable

from wzm_unpack import _texture_role, inspect_wzu_materials


TOOL_DIR = Path(__file__).resolve().parent
CACHE_FILE = TOOL_DIR / "mod_resource_index.json"
INDEX_VERSION = 5
INDEX_ROOTS = ("Armor", "Weapon", "Weapon1", "PC", "NPC", "Pet", "Riding")


def set_cache_file(path: Path) -> None:
    """Select a persistent cache location (important for one-file builds)."""
    global CACHE_FILE
    CACHE_FILE = Path(path).resolve()

CLASS_LABELS = {
    "berserker": "狂战士",
    "dragon": "龙骑士",
    "valkyrie": "圣射手",
    "magician": "魔法师",
    "elementalist": "魔法师",
    "shadow": "暗影",
    "mystic": "刺客",
    "helroid": "海洛伊德",
    "witchblade": "魔力刀锋",
    "berserker_w": "狂战士（女）",
    "dragon_w": "龙骑士（女）",
    "shadow_w": "暗影（女）",
    "valkyrie_m": "圣射手（男）",
    "vlakyrie_m": "圣射手（男）",
    "elementalist_m": "魔法师（男）",
}

PART_LABELS = {
    "chest": "铠甲",
    "wing": "护肩/翼",
    "protector": "护肩",
    "helm": "头盔",
    "head": "头部",
    "pants": "护腿",
    "boots": "靴子",
    "hand": "护手",
    "glove": "护手",
    "belt": "腰带",
    "shirt": "衬衣",
    "ax": "巨斧",
    "axe": "巨斧",
    "sword": "剑",
    "spear": "长矛",
    "dagger": "匕首",
    "whip": "魔剑",
    "crossbow": "弩",
    "staff": "法杖",
    "orb": "魔杖",
    "scythe": "镰刀",
    "blaster": "能源爆裂",
    "blade": "弧刃",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_label(value: str) -> str:
    return CLASS_LABELS.get(value.casefold(), value)


def _part_label(stem: str) -> str:
    low = stem.casefold()
    for token, label in PART_LABELS.items():
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", low):
            return label
    return "其他部件"


def classify(relative: Path) -> list[str]:
    parts = relative.parts
    top = parts[0].casefold() if parts else ""
    second = parts[1] if len(parts) > 1 else "未分组"
    tier = parts[2] if len(parts) > 2 else "通用"
    stem = relative.stem
    if top == "armor":
        low = stem.casefold()
        if "hair" in low:
            return ["装扮模型", "发型", _class_label(second), tier]
        if "costume" in low or "coustme" in low or low.startswith("c_"):
            return ["装扮模型", "时装", _class_label(second), tier, _part_label(stem)]
        if any(token in low for token in ("face", "mask", "hat", "cap")):
            return ["装扮模型", "头饰/面饰", _class_label(second), tier]
        return ["装备模型", "防具", _class_label(second), tier, _part_label(stem)]
    if top in ("weapon", "weapon1"):
        return ["装备模型", "武器", _class_label(second), tier, _part_label(stem)]
    if top == "pc":
        return ["角色模型", "玩家角色", _class_label(second), _part_label(stem)]
    if top == "npc":
        return ["角色模型", "NPC/怪物", second, _part_label(stem)]
    if top == "pet":
        return ["角色模型", "宠物", second, _part_label(stem)]
    if top == "riding":
        return ["角色模型", "坐骑", second, _part_label(stem)]
    if top == "map":
        area = second if len(parts) > 1 else "地图公共资源"
        section = parts[2] if len(parts) > 2 else "根目录"
        return ["场景模型", area, section]
    if top == "effect":
        return ["特效模型", second, _part_label(stem)]
    if top in ("item", "element", "arrow"):
        return ["道具模型", parts[0], second]
    if top == "interface":
        return ["界面模型", second]
    return ["其他模型", parts[0] if parts else "未分类", second]


def display_name(path: Path) -> str:
    name = path.stem
    replacements = {
        "chest": "铠甲", "helm": "头盔", "pants": "护腿", "boots": "靴子",
        "hand": "护手", "wing": "护肩/翼", "ax": "巨斧", "sword": "剑",
    }
    for token, label in replacements.items():
        name = re.sub(rf"(?:^|_){token}(?=_|$)", f"_{label}", name, flags=re.I)
    return name.strip("_")


def _find_case_insensitive(directory: Path, name: str) -> Path | None:
    target = Path(name).name.casefold()
    try:
        for path in directory.iterdir():
            if path.is_file() and path.name.casefold() == target:
                return path
    except OSError:
        return None
    return None


def _companion_wzu(model_path: Path, directory_files: dict[str, Path] | None = None) -> Path | None:
    files = directory_files or {
        path.name.casefold(): path for path in model_path.parent.iterdir() if path.is_file()
    }
    return files.get(f"{model_path.stem}.wzu".casefold())


def _texture_rows(model_path: Path, data: bytes, unit_path: Path | None, directory_files: dict[str, Path] | None = None) -> list[dict[str, str]]:
    files = directory_files or {
        path.name.casefold(): path for path in model_path.parent.iterdir() if path.is_file()
    }
    roles_by_name: dict[str, str] = {}
    if unit_path:
        try:
            for row in inspect_wzu_materials(unit_path):
                roles_by_name.setdefault(str(row["name"]).casefold(), str(row["role"]))
        except OSError:
            pass
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    names = [
        match.group().decode("ascii", errors="replace")
        for match in re.finditer(rb"[A-Za-z0-9_.-]{3,}\.(?:dds|tga|bmp)", data, re.I)
    ]
    for position, name in enumerate(dict.fromkeys(names)):
        if name.casefold() in seen:
            continue
        role = _texture_role(name)
        if role == "unknown":
            role = "diffuse" if position == 0 else "specular" if position == 1 else "unknown"
        seen.add(name.casefold())
        path = files.get(Path(name).name.casefold())
        rows.append({"name": name, "role": role, "path": str(path) if path else ""})
    for name_key, role in roles_by_name.items():
        if name_key in seen:
            continue
        path = files.get(Path(name_key).name.casefold())
        rows.append({"name": Path(name_key).name, "role": role, "path": str(path) if path else ""})
        seen.add(name_key)
    return rows


def build_index(
    root: Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    root = Path(root).resolve()
    paths = sorted(
        (path for folder in INDEX_ROOTS for path in (root / folder).rglob("*.wzm") if (root / folder).is_dir()),
        key=lambda path: str(path).casefold(),
    )
    models: list[dict] = []
    material_models: dict[str, set[str]] = defaultdict(set)
    material_files: dict[str, set[str]] = defaultdict(set)
    failures: list[dict[str, str]] = []
    directory_cache: dict[str, dict[str, Path]] = {}
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root)
        if progress and (index == 1 or index % 50 == 0 or index == len(paths)):
            progress(index, len(paths), str(relative))
        try:
            data = path.read_bytes()
            if data[:4] != b"WZMD" or len(data) < 6:
                raise ValueError("不是 WZMD 模型文件")
            version = struct.unpack_from("<H", data, 4)[0]
        except (OSError, ValueError) as exc:
            failures.append({"path": str(relative), "error": str(exc)})
            continue
        directory_key = str(path.parent).casefold()
        directory_files = directory_cache.get(directory_key)
        if directory_files is None:
            directory_files = {item.name.casefold(): item for item in path.parent.iterdir() if item.is_file()}
            directory_cache[directory_key] = directory_files
        unit = _companion_wzu(path, directory_files)
        textures = _texture_rows(path, data, unit, directory_files)
        model_key = str(relative).replace("/", "\\")
        for texture in textures:
            key = texture["name"].casefold()
            material_models[key].add(model_key)
            if texture["path"]:
                material_files[key].add(texture["path"])
        models.append({
            "path": str(path),
            "relative": model_key,
            "name": display_name(path),
            "category": classify(relative),
            "version": version,
            "bones": -1,
            "submeshes": -1,
            "triangles": -1,
            "wzu": str(unit) if unit else "",
            "materials": textures,
        })
    materials = {
        key: {
            "model_count": len(model_paths),
            "models": sorted(model_paths),
            "files": sorted(material_files.get(key, set())),
        }
        for key, model_paths in material_models.items()
    }
    payload = {
        "version": INDEX_VERSION,
        "root": str(root),
        "models": models,
        "materials": materials,
        "failures": failures,
    }
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return payload


def load_index(root: Path) -> dict | None:
    root = Path(root).resolve()
    if not CACHE_FILE.is_file():
        return None
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("version") != INDEX_VERSION or payload.get("root") != str(root):
        return None
    return payload


def index_is_stale(payload: dict, root: Path) -> bool:
    if not payload:
        return True
    root = Path(root).resolve()
    try:
        cache_time = CACHE_FILE.stat().st_mtime_ns
        newest = max(
            (
                path.stat().st_mtime_ns
                for folder in INDEX_ROOTS
                for pattern in ("*.wzm", "*.wzu")
                for path in (root / folder).rglob(pattern)
                if (root / folder).is_dir()
            ),
            default=0,
        )
        return newest > cache_time
    except OSError:
        return True

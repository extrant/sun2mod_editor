from __future__ import annotations

import json
import io
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import unquote

from PIL import Image, ImageTk

SOURCE_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False) or "__compiled__" in globals())
TOOL_DIR = Path(sys.argv[0]).resolve().parent if IS_FROZEN else SOURCE_DIR
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from mod_resource_index import INDEX_ROOTS, build_index, load_index, set_cache_file, sha256
from wzm_unpack import parse_wzm


APP_TITLE = "SUN2 MOD 资源编辑器 QQ群:221860548"
PREVIEW_DIR = TOOL_DIR / "mod_previews"
BACKUP_DIR = TOOL_DIR / "mod_backups"
SETTINGS_FILE = TOOL_DIR / "mod_editor_settings.json"
PREVIEW_MAPPINGS_FILE = TOOL_DIR / "preview_material_mappings.json"
PREVIEW_MAPPINGS_VERSION = 1
set_cache_file(TOOL_DIR / "mod_resource_index.json")
ICON_FILE = next(
    (
        path for path in (
            SOURCE_DIR / "favicon.ico",
            SOURCE_DIR / "解包和MOD工具" / "favicon.ico",
            TOOL_DIR / "favicon.ico",
        )
        if path.is_file()
    ),
    None,
)
ROLE_LABELS = {
    "diffuse": "基础颜色",
    "normal": "法线",
    "specular": "镜面/高光",
    "glow": "自发光",
    "diffuse_variant": "颜色变体",
    "internal": "内部材质",
    "effect_auxiliary": "特效辅助",
    "unknown": "其他",
}


BLENDER_EXPORT_SCRIPT = r'''
import bpy
import json
import re
import shutil
import sys
from pathlib import Path

blend_path, gltf_path, manifest_path, extracted_path = sys.argv[sys.argv.index('--') + 1:]
gltf_path = Path(gltf_path)
manifest_path = Path(manifest_path)
extracted_path = Path(extracted_path)
extracted_path.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend_path)

def is_exported(obj):
    return not any(collection.name.startswith('glTF_not_exported') for collection in obj.users_collection)

bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    obj.select_set(is_exported(obj))

face_counts = {}
for obj in bpy.context.scene.objects:
    if not is_exported(obj) or obj.type != 'MESH':
        continue
    for polygon in obj.data.polygons:
        if 0 <= polygon.material_index < len(obj.material_slots):
            material = obj.material_slots[polygon.material_index].material
            if material:
                face_counts[material.name] = face_counts.get(material.name, 0) + 1

def safe_stem(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
    return value or 'image'

def payload_suffix(payload):
    if payload.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if payload.startswith(b'DDS '):
        return '.dds'
    if payload[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if payload[:2] == b'BM':
        return '.bmp'
    return ''

extracted = {}
used_names = set()
blend_directory = Path(blend_path).resolve().parent

def find_nearby_source(image, missing_source):
    requested = {image.name.casefold()}
    if image.filepath:
        requested.add(Path(image.filepath).name.casefold())
    roots = []
    for root in (blend_directory, blend_directory.parent):
        if root.is_dir() and root not in roots:
            roots.append(root)
    matches = []
    seen = set()
    for root in roots:
        try:
            for candidate in root.rglob('*'):
                if not candidate.is_file() or candidate.name.casefold() not in requested:
                    continue
                key = str(candidate.resolve()).casefold()
                if key not in seen:
                    seen.add(key)
                    matches.append(candidate)
                if len(matches) >= 50:
                    break
        except OSError:
            continue
    if not matches:
        return None
    wanted_parts = {part.casefold() for part in missing_source.parts} if missing_source else set()
    matches.sort(key=lambda path: (
        -sum(part.casefold() in wanted_parts for part in path.parts),
        len(path.parts),
        str(path).casefold(),
    ))
    return matches[0]

def extract_image(image):
    key = str(image.as_pointer())
    if key in extracted:
        return extracted[key]
    source = Path(bpy.path.abspath(image.filepath, library=image.library)) if image.filepath else None
    packed = image.packed_file
    payload = b''
    packed_error = ''
    if packed:
        try:
            payload = bytes(packed.data)
        except Exception as exc:
            packed_error = str(exc)
    recovered_source = False
    if not payload and (source is None or not source.is_file()):
        nearby = find_nearby_source(image, source)
        if nearby is not None:
            source = nearby
            recovered_source = True
            try:
                image.filepath = str(source)
                image.reload()
            except Exception:
                pass
    suffix = payload_suffix(payload)
    if not suffix and source and source.suffix:
        suffix = source.suffix.lower()
    if suffix not in ('.dds', '.tga', '.bmp', '.png', '.jpg', '.jpeg'):
        suffix = '.png'
    stem = safe_stem(Path(image.name).stem)
    name = stem + suffix
    counter = 2
    while name.casefold() in used_names:
        name = f'{stem}_{counter}{suffix}'
        counter += 1
    used_names.add(name.casefold())
    destination = extracted_path / name
    try:
        if payload:
            destination.write_bytes(payload)
        elif source and source.is_file():
            shutil.copy2(source, destination)
        elif image.has_data:
            old_raw = image.filepath_raw
            old_format = image.file_format
            image.filepath_raw = str(destination.with_suffix('.png'))
            image.file_format = 'PNG'
            image.save_render(filepath=image.filepath_raw, scene=bpy.context.scene)
            destination = destination.with_suffix('.png')
            image.filepath_raw = old_raw
            image.file_format = old_format
        else:
            recorded = bpy.path.abspath(image.filepath, library=image.library) if image.filepath else '（空）'
            extra = f'；打包数据错误：{packed_error}' if packed_error else ''
            raise RuntimeError(f'已连接图片，但外部文件不存在且没有打包进 blend；记录路径：{recorded}{extra}')
    except Exception as exc:
        extracted[key] = {'image': image.name, 'error': str(exc)}
        return extracted[key]
    extracted[key] = {
        'image': image.name,
        'path': str(destination.resolve()),
        'packed': bool(packed),
        'source': str(source) if source else '',
        'recovered_source': recovered_source,
    }
    return extracted[key]

def linked_images(socket):
    if socket is None:
        return []
    queue = [(socket, 0)]
    seen_sockets = set()
    found = []
    while queue:
        current, depth = queue.pop(0)
        pointer = current.as_pointer()
        if pointer in seen_sockets or depth > 24:
            continue
        seen_sockets.add(pointer)
        for link in current.links:
            node = link.from_node
            if node.type == 'TEX_IMAGE' and node.image:
                found.append((depth, node, node.image))
                continue
            for upstream in node.inputs:
                if upstream.is_linked:
                    queue.append((upstream, depth + 1))
    found.sort(key=lambda item: (item[0], item[1].name.casefold()))
    unique = []
    seen_images = set()
    for depth, node, image in found:
        key = image.as_pointer()
        if key not in seen_images:
            seen_images.add(key)
            unique.append((depth, node, image))
    return unique

def shader_nodes(material):
    if not material.use_nodes or not material.node_tree:
        return []
    outputs = [node for node in material.node_tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
    if not outputs:
        outputs = [node for node in material.node_tree.nodes if node.type == 'OUTPUT_MATERIAL']
    queue = []
    for output in outputs:
        surface = output.inputs.get('Surface')
        if surface:
            queue.append((surface, 0))
    seen = set()
    result = []
    while queue:
        socket, depth = queue.pop(0)
        if socket.as_pointer() in seen or depth > 24:
            continue
        seen.add(socket.as_pointer())
        for link in socket.links:
            node = link.from_node
            if node.type in ('BSDF_PRINCIPLED', 'BSDF_DIFFUSE'):
                result.append((depth, node))
            else:
                for upstream in node.inputs:
                    if upstream.is_linked:
                        queue.append((upstream, depth + 1))
    result.sort(key=lambda item: (item[0], 0 if item[1].type == 'BSDF_PRINCIPLED' else 1, item[1].name.casefold()))
    return result

def input_by_names(node, names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None

material_rows = []
for material in bpy.data.materials:
    if face_counts.get(material.name, 0) <= 0:
        continue
    row = {
        'name': material.name, 'face_count': face_counts[material.name],
        'roles': {}, 'failed_roles': {}, 'warnings': [],
    }
    shaders = shader_nodes(material)
    if not shaders:
        row['warnings'].append('材质输出没有连接到 Principled/Diffuse BSDF')
        material_rows.append(row)
        continue
    if len(shaders) > 1:
        row['warnings'].append(f'材质输出包含 {len(shaders)} 个表面着色器，按最接近输出的着色器读取')
    shader = shaders[0][1]
    role_sockets = {
        'diffuse': input_by_names(shader, ('Base Color', 'Color')),
        'normal': input_by_names(shader, ('Normal',)),
        'specular': input_by_names(shader, ('Specular IOR Level', 'Specular')),
    }
    for role, socket in role_sockets.items():
        candidates = linked_images(socket)
        if not candidates:
            continue
        depth, node, image = candidates[0]
        saved = extract_image(image)
        if 'path' not in saved:
            row['failed_roles'][role] = {'image': image.name, 'error': saved.get('error', '未知错误')}
            row['warnings'].append(f'{role} 图片 {image.name} 提取失败：{saved.get("error", "未知错误")}')
            continue
        row['roles'][role] = {
            **saved,
            'node': node.name,
            'socket': socket.name,
            'candidate_count': len(candidates),
        }
        if saved.get('recovered_source'):
            row['warnings'].append(f'{role} 图片原路径失效，已在相邻模型目录找回：{saved.get("source", "")}')
        if len(candidates) > 1:
            row['warnings'].append(
                f'{role} 输入链包含 {len(candidates)} 张图片，自动选择离着色器最近的 {image.name}'
            )
    material_rows.append(row)

manifest_path.write_text(json.dumps({'materials': material_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
bpy.ops.export_scene.gltf(
    filepath=str(gltf_path), export_format='GLTF_SEPARATE', export_texture_dir='textures',
    export_skins=True, export_all_influences=True, export_tangents=True,
    export_morph=False, export_animations=False, export_extras=True,
    export_materials='EXPORT', use_selection=True,
)
'''


def find_client_data(selected: Path) -> tuple[Path, Path]:
    """Return (game directory, Data directory) for a user-selected folder."""
    selected = Path(selected).expanduser().resolve()
    candidates = (
        (selected, selected / "Data"),
        (selected / "客户端", selected / "客户端" / "Data"),
        (selected.parent, selected),
    )
    for game_path, data_path in candidates:
        if data_path.is_dir() and any((data_path / name).is_dir() for name in INDEX_ROOTS):
            return game_path.resolve(), data_path.resolve()
    raise ValueError(
        "所选目录中没有找到有效的客户端 Data 资源目录。\n\n"
        "可以选择游戏客户端目录、Data 目录，或包含“客户端\\Data”的上级目录。"
    )


def load_settings() -> dict:
    if not SETTINGS_FILE.is_file():
        return {}
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(game_path: Path, client_data: Path) -> None:
    payload = {
        "game_path": str(game_path),
        "client_data": str(client_data),
    }
    SETTINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_preview_mappings() -> dict:
    empty = {"version": PREVIEW_MAPPINGS_VERSION, "models": {}}
    if not PREVIEW_MAPPINGS_FILE.is_file():
        return empty
    try:
        payload = json.loads(PREVIEW_MAPPINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if payload.get("version") != PREVIEW_MAPPINGS_VERSION or not isinstance(payload.get("models"), dict):
        return empty
    return payload


def save_preview_mappings(payload: dict) -> None:
    payload["version"] = PREVIEW_MAPPINGS_VERSION
    payload.setdefault("models", {})
    temporary = PREVIEW_MAPPINGS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, PREVIEW_MAPPINGS_FILE)


def viewer_command(*arguments: object) -> list[str]:
    args = [str(value) for value in arguments]
    if IS_FROZEN:
        return [str(Path(sys.argv[0]).resolve()), "--viewer", *args]
    return [sys.executable, str(SOURCE_DIR / "wzm_viewer.py"), *args]


class ScrollImage(ttk.Frame):
    def __init__(self, master, placeholder: str):
        super().__init__(master)
        self.source: Path | None = None
        self.photo = None
        self.fit_mode = False
        self.canvas = tk.Canvas(self, bg="#15181d", highlightthickness=0)
        xbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.clear(placeholder)

    def clear(self, text: str) -> None:
        self.source = None
        self.photo = None
        self.canvas.delete("all")
        self.canvas.create_text(20, 20, text=text, fill="#cbd3dc", anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))

    def show(self, path: Path) -> None:
        self.source = path
        self.after_idle(self._render)

    def set_fit(self, enabled: bool) -> None:
        self.fit_mode = enabled
        if self.source:
            self._render()

    def _render(self) -> None:
        if not self.source or not self.source.is_file():
            return
        with Image.open(self.source) as opened:
            image = opened.convert("RGBA")
        if self.fit_mode:
            width = max(self.canvas.winfo_width() - 4, 1)
            height = max(self.canvas.winfo_height() - 4, 1)
            scale = min(width / image.width, height / image.height, 1.0)
            if scale < 1.0:
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.convert("RGBa").resize(size, Image.Resampling.BOX).convert("RGBA")
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)


class ModEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        if ICON_FILE is not None:
            try:
                self.iconbitmap(default=str(ICON_FILE))
            except tk.TclError:
                pass
        self.geometry("1480x880")
        self.minsize(1080, 680)
        self.option_add("*Font", ("Microsoft YaHei UI", 9))
        self.game_path: Path | None = None
        self.client_data: Path | None = None
        self.path_generation = 0
        self.index_data: dict | None = None
        self.filtered_models: list[dict] = []
        self.model_by_row: dict[str, dict] = {}
        self.category_by_node: dict[str, tuple[str, ...]] = {}
        self.current_model: dict | None = None
        self.current_material: dict | None = None
        self.material_by_row: dict[str, dict] = {}
        self.replacement_map: dict[str, Path] = {}
        self.preview_mappings = load_preview_mappings()
        self.preview_photo = None
        self.preview_generation = 0
        self.preview_process: subprocess.Popen | None = None
        self.search_var = tk.StringVar()
        self.game_path_var = tk.StringVar(value="游戏路径：尚未设置")
        self.status_var = tk.StringVar(value="正在载入资源索引")
        self.model_info_var = tk.StringVar(value="选择模型后显示信息")
        self.material_info_var = tk.StringVar(value="选择材质后显示信息")
        self.replacement_var = tk.StringVar()
        self._configure_style()
        self._build_ui()
        self.after(50, self._initialize_game_path)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=23)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 7))
        ttk.Label(header, text="SUN2 MOD 资源编辑器 QQ群:221860548", style="Title.TLabel").pack(side="left")
        self.rebuild_button = ttk.Button(header, text="重建索引", command=lambda: self._build_index_async(True))
        self.rebuild_button.pack(side="right")
        ttk.Button(header, text="设置游戏路径", command=self._select_game_path).pack(side="right", padx=(0, 6))
        ttk.Label(header, textvariable=self.game_path_var, width=55, anchor="e").pack(side="right", padx=(8, 12))
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.browser_tab = ttk.Frame(self.tabs)
        self.editor_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.browser_tab, text="资源浏览")
        self.tabs.add(self.editor_tab, text="材质编辑")
        self._build_browser()
        self._build_editor()
        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(6, 0))

    def _build_browser(self) -> None:
        pane = ttk.Panedwindow(self.browser_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=5)
        middle = ttk.Frame(pane, padding=5)
        right = ttk.Frame(pane, padding=5)
        pane.add(left, weight=2)
        pane.add(middle, weight=4)
        pane.add(right, weight=5)

        ttk.Label(left, text="资源分类", style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        self.category_tree = ttk.Treeview(left, show="tree", selectmode="browse")
        category_scroll = ttk.Scrollbar(left, orient="vertical", command=self.category_tree.yview)
        self.category_tree.configure(yscrollcommand=category_scroll.set)
        self.category_tree.pack(side="left", fill="both", expand=True)
        category_scroll.pack(side="right", fill="y")
        self.category_tree.bind("<<TreeviewSelect>>", self._category_selected)

        search_row = ttk.Frame(middle)
        search_row.pack(fill="x", pady=(0, 5))
        ttk.Entry(search_row, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        ttk.Button(search_row, text="搜索", command=self._refresh_models).pack(side="left", padx=(5, 0))
        self.search_var.trace_add("write", lambda *_: self.after_idle(self._refresh_models))
        columns = ("name", "version", "triangles", "materials")
        self.model_tree = ttk.Treeview(middle, columns=columns, show="headings", selectmode="browse")
        for key, label, width in (
            ("name", "模型", 250), ("version", "版本", 55),
            ("triangles", "三角形", 75), ("materials", "材质", 55),
        ):
            self.model_tree.heading(key, text=label)
            self.model_tree.column(key, width=width, stretch=key == "name")
        model_scroll = ttk.Scrollbar(middle, orient="vertical", command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=model_scroll.set)
        self.model_tree.pack(side="left", fill="both", expand=True)
        model_scroll.pack(side="right", fill="y")
        self.model_tree.bind("<<TreeviewSelect>>", self._model_selected)

        preview_box = tk.Frame(right, bg="#101318", width=560, height=520)
        preview_box.pack(fill="both", expand=True)
        preview_box.pack_propagate(False)
        self.preview_label = tk.Label(
            preview_box, text="选择模型后自动生成 3D 预览", bg="#101318", fg="#cbd3dc",
            anchor="center", justify="center",
        )
        self.preview_label.pack(fill="both", expand=True)
        ttk.Label(right, textvariable=self.model_info_var, wraplength=540, justify="left").pack(fill="x", pady=(7, 4))
        buttons = ttk.Frame(right)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="交互预览", command=self._open_interactive).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="打开目录", command=self._open_model_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(buttons, text="导出 Blender 包", command=self._export_blender).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(buttons, text="进入编辑页", command=self._open_editor).pack(side="left", fill="x", expand=True)

    def _build_editor(self) -> None:
        pane = ttk.Panedwindow(self.editor_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=7)
        right = ttk.Frame(pane, padding=7)
        pane.add(left, weight=3)
        pane.add(right, weight=5)
        ttk.Label(left, text="模型材质", style="Section.TLabel").pack(anchor="w")
        self.edit_model_label = ttk.Label(left, text="尚未选择模型", wraplength=380)
        self.edit_model_label.pack(fill="x", pady=(4, 7))
        self.material_tree = ttk.Treeview(left, columns=("role", "reuse", "preview", "replacement"), show="tree headings", selectmode="browse")
        self.material_tree.heading("#0", text="材质文件")
        self.material_tree.heading("role", text="作用")
        self.material_tree.heading("reuse", text="复用")
        self.material_tree.heading("preview", text="3D槽位")
        self.material_tree.heading("replacement", text="待替换")
        self.material_tree.column("#0", width=205)
        self.material_tree.column("role", width=90, anchor="center")
        self.material_tree.column("reuse", width=60, anchor="center")
        self.material_tree.column("preview", width=70, anchor="center")
        self.material_tree.column("replacement", width=120)
        self.material_tree.pack(fill="both", expand=True)
        self.material_tree.bind("<<TreeviewSelect>>", self._material_selected)
        ttk.Label(left, textvariable=self.material_info_var, wraplength=390, justify="left").pack(fill="x", pady=7)
        ttk.Button(left, text="查看全部引用模型", command=self._show_material_references).pack(fill="x", pady=(0, 4))

        images = ttk.Frame(right)
        images.pack(fill="both", expand=True)
        current = ttk.LabelFrame(images, text="当前材质", padding=5)
        replacement = ttk.LabelFrame(images, text="替换材质", padding=5)
        current.pack(side="left", fill="both", expand=True, padx=(0, 4))
        replacement.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.current_image = ScrollImage(current, "选择材质")
        self.current_image.pack(fill="both", expand=True)
        self.replacement_image = ScrollImage(replacement, "选择替换图片")
        self.replacement_image.pack(fill="both", expand=True)
        current_modes = ttk.Frame(current)
        current_modes.pack(fill="x", pady=(4, 0))
        ttk.Button(current_modes, text="原图 100%", command=lambda: self.current_image.set_fit(False)).pack(side="left", fill="x", expand=True)
        ttk.Button(current_modes, text="适应窗口", command=lambda: self.current_image.set_fit(True)).pack(side="left", fill="x", expand=True, padx=(4, 0))
        replacement_modes = ttk.Frame(replacement)
        replacement_modes.pack(fill="x", pady=(4, 0))
        ttk.Button(replacement_modes, text="原图 100%", command=lambda: self.replacement_image.set_fit(False)).pack(side="left", fill="x", expand=True)
        ttk.Button(replacement_modes, text="适应窗口", command=lambda: self.replacement_image.set_fit(True)).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(right, textvariable=self.replacement_var, wraplength=700).pack(fill="x", pady=(7, 4))
        preview_commands = ttk.LabelFrame(right, text="3D 预览贴图映射（只影响预览，不修改游戏文件）", padding=5)
        preview_commands.pack(fill="x", pady=(0, 5))
        ttk.Button(preview_commands, text="将当前材质映射到模型槽位", command=self._map_current_material_for_preview).pack(side="left")
        ttk.Button(preview_commands, text="从文件选择并映射", command=self._map_file_for_preview).pack(side="left", padx=5)
        ttk.Button(preview_commands, text="清除映射", command=self._clear_preview_mapping).pack(side="left")
        commands = ttk.Frame(right)
        commands.pack(fill="x")
        ttk.Button(commands, text="为所选材质选择图片", command=self._choose_replacement).pack(side="left")
        ttk.Button(commands, text="备份并应用全部", command=self._apply_replacement).pack(side="left", padx=5)
        ttk.Button(commands, text="清空替换列表", command=self._clear_replacements).pack(side="left")
        ttk.Button(commands, text="恢复最近备份", command=self._restore_latest).pack(side="left")
        ttk.Button(commands, text="导入 Blender 修改", command=self._import_blend).pack(side="left", padx=5)
        ttk.Button(commands, text="刷新 3D 预览", command=self._refresh_preview_after_edit).pack(side="right")

    def _initialize_game_path(self) -> None:
        configured = load_settings().get("game_path")
        if configured:
            try:
                game_path, client_data = find_client_data(Path(configured))
            except (OSError, ValueError):
                pass
            else:
                self._set_game_path(game_path, client_data)
                self._load_or_build_index()
                return
        self.status_var.set("首次启动：请选择 SUN 游戏客户端目录")
        self._select_game_path(first_run=True)

    def _select_game_path(self, first_run: bool = False) -> None:
        initial = str(self.game_path) if self.game_path else str(TOOL_DIR)
        value = filedialog.askdirectory(
            parent=self,
            title="选择 SUN 游戏客户端目录（也可选择 Data 目录）",
            initialdir=initial,
            mustexist=True,
        )
        if not value:
            if first_run:
                self.status_var.set("尚未设置游戏路径；请点击右上角“设置游戏路径”")
                messagebox.showinfo(APP_TITLE, "未选择游戏路径。稍后可点击右上角“设置游戏路径”。")
            return
        try:
            game_path, client_data = find_client_data(Path(value))
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        changed = client_data != self.client_data
        self._set_game_path(game_path, client_data)
        try:
            save_settings(game_path, client_data)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"保存设置失败：\n{exc}")
            return
        if changed:
            self._reset_index_view()
            self._load_or_build_index()
        else:
            self.status_var.set(f"游戏路径已保存：{game_path}")

    def _set_game_path(self, game_path: Path, client_data: Path) -> None:
        if client_data != self.client_data:
            self.path_generation += 1
        self.game_path = game_path
        self.client_data = client_data
        self.game_path_var.set(f"游戏路径：{game_path}")

    def _reset_index_view(self) -> None:
        self.preview_generation += 1
        self.index_data = None
        self.filtered_models.clear()
        self.model_by_row.clear()
        self.category_by_node.clear()
        self.current_model = None
        self.current_material = None
        self.replacement_map.clear()
        self.category_tree.delete(*self.category_tree.get_children())
        self.model_tree.delete(*self.model_tree.get_children())
        self.material_tree.delete(*self.material_tree.get_children())
        self.preview_label.configure(image="", text="选择模型后自动生成 3D 预览")
        self.preview_photo = None
        self.model_info_var.set("选择模型后显示信息")
        self.material_info_var.set("选择材质后显示信息")

    def _load_or_build_index(self) -> None:
        if self.client_data is None:
            self.status_var.set("请先设置游戏路径")
            return
        payload = load_index(self.client_data)
        if payload is None:
            self._build_index_async(False)
            return
        self._install_index(payload)

    def _build_index_async(self, confirm: bool) -> None:
        if self.client_data is None:
            messagebox.showinfo(APP_TITLE, "请先设置游戏路径。")
            return
        if confirm and not messagebox.askyesno(APP_TITLE, "重建索引会重新扫描全部模型，是否继续？"):
            return
        client_data = self.client_data
        generation = self.path_generation
        self.status_var.set(f"正在扫描客户端模型资源：{client_data}")
        def worker():
            try:
                payload = build_index(client_data, progress=lambda done, total, name: self.after(
                    0, self._set_index_progress, generation, done, total, name
                ))
                self.after(0, self._install_index_if_current, payload, generation)
            except Exception as exc:
                self.after(0, self._show_index_error, generation, exc)
        threading.Thread(target=worker, daemon=True).start()

    def _set_index_progress(self, generation: int, done: int, total: int, name: str) -> None:
        if generation == self.path_generation:
            self.status_var.set(f"正在建立索引 {done:,}/{total:,} · {name}")

    def _install_index_if_current(self, payload: dict, generation: int) -> None:
        if generation == self.path_generation:
            self._install_index(payload)

    def _show_index_error(self, generation: int, exc: Exception) -> None:
        if generation == self.path_generation:
            messagebox.showerror(APP_TITLE, f"建立索引失败：\n{exc}")

    def _install_index(self, payload: dict) -> None:
        self.index_data = payload
        self.category_tree.delete(*self.category_tree.get_children())
        self.category_by_node.clear()
        nodes: dict[tuple[str, ...], str] = {}
        counts: dict[tuple[str, ...], int] = {}
        for model in payload.get("models", []):
            category = tuple(model["category"])
            for depth in range(1, len(category) + 1):
                prefix = category[:depth]
                counts[prefix] = counts.get(prefix, 0) + 1
        for category in sorted(counts, key=lambda value: tuple(part.casefold() for part in value)):
            parent_prefix = category[:-1]
            parent = nodes.get(parent_prefix, "")
            node = self.category_tree.insert(parent, "end", text=f"{category[-1]}  ({counts[category]:,})", open=len(category) <= 1)
            nodes[category] = node
            self.category_by_node[node] = category
        self._refresh_models()
        self.status_var.set(
            f"已索引 {len(payload.get('models', [])):,} 个模型、"
            f"{len(payload.get('materials', {})):,} 种材质；"
            f"{len(payload.get('failures', [])):,} 个文件未识别"
        )

    def _selected_category(self) -> tuple[str, ...]:
        selection = self.category_tree.selection()
        return self.category_by_node.get(selection[0], ()) if selection else ()

    def _category_selected(self, _event=None) -> None:
        self._refresh_models()

    def _refresh_models(self) -> None:
        if not self.index_data:
            return
        prefix = self._selected_category()
        terms = [term.casefold() for term in self.search_var.get().split() if term]
        models = []
        for model in self.index_data.get("models", []):
            category = tuple(model["category"])
            if prefix and category[:len(prefix)] != prefix:
                continue
            haystack = " ".join((model["name"], model["relative"], *[row["name"] for row in model["materials"]])).casefold()
            if all(term in haystack for term in terms):
                models.append(model)
        self.filtered_models = models
        self.model_tree.delete(*self.model_tree.get_children())
        self.model_by_row.clear()
        for model in models[:5000]:
            row = self.model_tree.insert("", "end", values=(
                model["name"], model["version"], (f"{model['triangles']:,}" if model["triangles"] >= 0 else "按需"), len(model["materials"]),
            ))
            self.model_by_row[row] = model
        self.status_var.set(f"当前分类显示 {min(len(models), 5000):,}/{len(models):,} 个模型")

    def _model_selected(self, _event=None) -> None:
        selection = self.model_tree.selection()
        if not selection:
            return
        selected_model = self.model_by_row.get(selection[0])
        if self.current_model and selected_model and self.current_model.get("path") != selected_model.get("path"):
            self.replacement_map.clear()
            self.current_material = None
        self.current_model = selected_model
        if not self.current_model:
            return
        model = self.current_model
        self.model_info_var.set(
            f"{model['relative']}\nWZM {model['version']} · 骨骼 {model['bones']} · "
            f"材质 {len(model['materials'])} · 正在读取详细结构"
        )
        self._populate_materials()
        self._render_preview()
        self._load_model_details(model)

    def _load_model_details(self, model: dict) -> None:
        if model.get("triangles", -1) >= 0:
            self._show_model_details(model)
            return
        def worker():
            try:
                parsed = parse_wzm(model["path"])
                details = (len(parsed.bones), len(parsed.submeshes), sum(len(mesh.indices) // 3 for mesh in parsed.submeshes))
                self.after(0, self._install_model_details, model, details)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _install_model_details(self, model: dict, details: tuple[int, int, int]) -> None:
        model["bones"], model["submeshes"], model["triangles"] = details
        if self.current_model is model:
            self._show_model_details(model)

    def _show_model_details(self, model: dict) -> None:
        self.model_info_var.set(
            f"{model['relative']}\nWZM {model['version']} · 骨骼 {model['bones']} · "
            f"子网格 {model['submeshes']} · 三角形 {model['triangles']:,} · 材质 {len(model['materials'])}"
        )

    def _populate_materials(self) -> None:
        self.material_tree.delete(*self.material_tree.get_children())
        self.material_by_row.clear()
        if not self.current_model or not self.index_data:
            return
        mapped_paths: dict[str, list[int]] = {}
        for slot, path in self._preview_slot_paths(self.current_model, existing_only=False).items():
            mapped_paths.setdefault(str(path.resolve()).casefold(), []).append(slot)
        for row in self.current_model["materials"]:
            reuse = self.index_data["materials"].get(row["name"].casefold(), {}).get("model_count", 0)
            replacement = self.replacement_map.get(row.get("path", ""))
            path_text = row.get("path", "")
            slots = mapped_paths.get(str(Path(path_text).resolve()).casefold(), []) if path_text else []
            preview_slots = ", ".join(f"#{slot + 1}" for slot in slots)
            item = self.material_tree.insert("", "end", text=row["name"], values=(
                ROLE_LABELS.get(row["role"], row["role"]), reuse, preview_slots,
                replacement.name if replacement else "",
            ))
            self.material_tree.set(item, "role", ROLE_LABELS.get(row["role"], row["role"]))
            self.material_by_row[item] = row

    def _render_preview(self) -> None:
        if not self.current_model:
            return
        self.preview_generation += 1
        generation = self.preview_generation
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        destination = PREVIEW_DIR / f"preview_{uuid.uuid4().hex}.png"
        self.preview_label.configure(image="", text="正在生成材质化 3D 预览…")
        command = self._viewer_command(
            "--no-effect", "--hidden", "--screenshot", str(destination), "--close-after-frame", "5",
        )
        def worker():
            try:
                subprocess.run(command, cwd=str(TOOL_DIR), check=True, timeout=40, creationflags=0x08000000)
                self.after(0, self._install_preview, destination, generation)
            except Exception as exc:
                self.after(0, self.preview_label.configure, {"image": "", "text": f"预览生成失败\n{exc}"})
        threading.Thread(target=worker, daemon=True).start()

    def _install_preview(self, path: Path, generation: int) -> None:
        if generation != self.preview_generation or not path.is_file():
            return
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((650, 620), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def _open_interactive(self) -> None:
        if not self.current_model:
            return
        subprocess.Popen(self._viewer_command("--no-effect"), cwd=str(TOOL_DIR))

    def _open_model_dir(self) -> None:
        if self.current_model:
            os.startfile(Path(self.current_model["path"]).parent)

    @staticmethod
    def _find_blender() -> Path | None:
        candidates = [
            Path(r"E:\SteamLibrary\steamapps\common\Blender\blender.exe"),
            Path(r"C:\Program Files\Blender Foundation\Blender\blender.exe"),
        ]
        command = shutil.which("blender")
        if command:
            candidates.insert(0, Path(command))
        return next((path for path in candidates if path.is_file()), None)

    def _export_blender(self) -> None:
        if not self.current_model:
            return
        model_record = self.current_model
        material_overrides = self._preview_slot_paths(model_record)
        self.status_var.set(f"正在导出 Blender 包：{model_record['name']}")
        def worker():
            try:
                from wzm_unpack import export_wzm, parse_wzm
                model = parse_wzm(model_record["path"])
                output = export_wzm(
                    model,
                    TOOL_DIR / "blender_exports",
                    companion_unit=model_record.get("wzu") or None,
                    material_overrides=material_overrides,
                )
                gltf = output / f"{Path(model_record['path']).stem}.gltf"
                blend = output / f"{Path(model_record['path']).stem}.blend"
                blender = self._find_blender()
                if blender:
                    script = output / "import_to_blender.py"
                    script.write_text(
                        "import bpy, sys\n"
                        "from pathlib import Path\n"
                        "args = sys.argv[sys.argv.index('--') + 1:]\n"
                        "gltf, blend = map(str, args[:2])\n"
                        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
                        "bpy.ops.import_scene.gltf(filepath=gltf, import_pack_images=True)\n"
                        "for material in bpy.data.materials:\n"
                        "    material.use_nodes = True\n"
                        "bpy.ops.wm.save_as_mainfile(filepath=blend)\n",
                        encoding="utf-8",
                    )
                    subprocess.run(
                        [str(blender), "--background", "--python", str(script), "--", str(gltf), str(blend)],
                        check=True,
                        timeout=180,
                        creationflags=0x08000000,
                    )
                self.after(0, self._blender_export_done, output, blend if blend.is_file() else gltf)
            except Exception as exc:
                self.after(0, messagebox.showerror, APP_TITLE, f"Blender 导出失败：\n{exc}")
                self.after(0, self.status_var.set, "Blender 导出失败")
        threading.Thread(target=worker, daemon=True).start()

    def _blender_export_done(self, output: Path, primary: Path) -> None:
        self.status_var.set(f"Blender 包已生成：{primary}")
        os.startfile(output)

    def _open_editor(self) -> None:
        if not self.current_model:
            return
        self._populate_materials()
        self.edit_model_label.configure(text=self.current_model["relative"])
        self.tabs.select(self.editor_tab)

    def _material_selected(self, _event=None) -> None:
        selection = self.material_tree.selection()
        if not selection or not self.current_model or not self.index_data:
            return
        self.current_material = self.material_by_row.get(selection[0])
        if not self.current_material:
            return
        name = self.current_material["name"]
        path = Path(self.current_material["path"]) if self.current_material["path"] else None
        reuse = self.index_data["materials"].get(name.casefold(), {})
        dimensions = "文件未定位"
        if path and path.is_file():
            try:
                with Image.open(path) as image:
                    dimensions = f"{image.width}×{image.height} · {image.format}"
                self.current_image.show(path)
            except Exception as exc:
                self.current_image.clear(f"读取失败\n{exc}")
        self.material_info_var.set(
            f"作用：{ROLE_LABELS.get(self.current_material['role'], self.current_material['role'])}\n"
            f"文件：{path or name}\n{dimensions}\n"
            f"被 {reuse.get('model_count', 0):,} 个模型引用"
        )
        replacement = self.replacement_map.get(self.current_material.get("path", ""))
        if replacement and replacement.is_file():
            self.replacement_image.show(replacement)
            self.replacement_var.set(str(replacement))
        else:
            self.replacement_image.clear("为该材质选择替换图片")
            self.replacement_var.set("")

    def _show_material_references(self) -> None:
        if not self.current_material or not self.index_data:
            messagebox.showinfo(APP_TITLE, "请先选择一个材质。")
            return
        data = self.index_data["materials"].get(self.current_material["name"].casefold(), {})
        references = data.get("models", [])
        model_map = {model["relative"].casefold(): model for model in self.index_data["models"]}
        window = tk.Toplevel(self)
        window.title(f"引用模型 · {self.current_material['name']} · {len(references):,} 个")
        window.geometry("920x620")
        frame = ttk.Frame(window, padding=8)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=("category", "path"), show="headings")
        tree.heading("category", text="分类")
        tree.heading("path", text="模型路径")
        tree.column("category", width=250)
        tree.column("path", width=620)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        rows: dict[str, str] = {}
        for relative in references:
            model = model_map.get(relative.casefold())
            category = " / ".join(model["category"]) if model else "未分类"
            row = tree.insert("", "end", values=(category, relative))
            rows[row] = relative
        def jump(_event=None):
            selected = tree.selection()
            if selected:
                relative = rows[selected[0]]
                window.destroy()
                self._jump_to_model(relative)
        tree.bind("<Double-1>", jump)
        ttk.Label(window, text="双击模型可在资源浏览页中定位").pack(pady=(0, 7))

    def _jump_to_model(self, relative: str) -> None:
        self.category_tree.selection_remove(self.category_tree.selection())
        self.search_var.set(relative)
        self.tabs.select(self.browser_tab)
        self._refresh_models()
        for row, model in self.model_by_row.items():
            if model["relative"].casefold() == relative.casefold():
                self.model_tree.selection_set(row)
                self.model_tree.focus(row)
                self.model_tree.see(row)
                self._model_selected()
                break

    @staticmethod
    def _preview_mapping_key(model: dict) -> str:
        return str(model["relative"]).replace("/", "\\").casefold()

    def _preview_mapping_entry(self, model: dict, create: bool = False) -> dict | None:
        models = self.preview_mappings.setdefault("models", {})
        key = self._preview_mapping_key(model)
        entry = models.get(key)
        if entry is None and create:
            entry = {"relative": model["relative"], "slots": {}}
            models[key] = entry
        if create and isinstance(entry, dict) and not isinstance(entry.get("slots"), dict):
            entry["slots"] = {}
        return entry if isinstance(entry, dict) else None

    def _preview_slot_paths(self, model: dict, existing_only: bool = True) -> dict[int, Path]:
        entry = self._preview_mapping_entry(model)
        if not entry or not isinstance(entry.get("slots"), dict):
            return {}
        model_dir = Path(model["path"]).parent
        result: dict[int, Path] = {}
        for slot_text, value in entry["slots"].items():
            try:
                slot = int(slot_text)
                path = Path(str(value))
            except (TypeError, ValueError):
                continue
            if not path.is_absolute():
                path = model_dir / path
            if not existing_only or path.is_file():
                result[slot] = path
        return result

    def _viewer_command(self, *arguments: object) -> list[str]:
        if not self.current_model:
            return []
        command: list[object] = [self.current_model["path"], *arguments]
        for slot, path in sorted(self._preview_slot_paths(self.current_model).items()):
            command.extend(("--texture-override", f"{slot}={path}"))
        return viewer_command(*command)

    def _choose_preview_slot(self, model: dict, allowed_slots: list[int] | None = None) -> int | None:
        parsed = parse_wzm(model["path"])
        slots = allowed_slots if allowed_slots is not None else list(range(len(parsed.submeshes)))
        if not slots:
            return None
        if len(slots) == 1:
            return slots[0]

        current = self._preview_slot_paths(model, existing_only=False)
        result: dict[str, int | None] = {"slot": None}
        window = tk.Toplevel(self)
        window.title("选择模型材质槽位")
        window.geometry("760x390")
        window.transient(self)
        window.grab_set()
        frame = ttk.Frame(window, padding=8)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="请选择这张贴图对应的 WZM 材质槽位：").pack(anchor="w", pady=(0, 6))
        tree = ttk.Treeview(frame, columns=("source", "mapped"), show="tree headings", selectmode="browse")
        tree.heading("#0", text="槽位")
        tree.heading("source", text="WZM 原始材质名")
        tree.heading("mapped", text="当前手动映射")
        tree.column("#0", width=70, anchor="center")
        tree.column("source", width=300)
        tree.column("mapped", width=330)
        rows: dict[str, int] = {}
        for slot in slots:
            submesh = parsed.submeshes[slot]
            row = tree.insert(
                "", "end", text=f"#{slot + 1}",
                values=(submesh.diffuse or "（未命名）", str(current.get(slot, ""))),
            )
            rows[row] = slot
        tree.pack(fill="both", expand=True)
        first = next(iter(rows), None)
        if first:
            tree.selection_set(first)
            tree.focus(first)

        def accept(_event=None):
            selection = tree.selection()
            if selection:
                result["slot"] = rows[selection[0]]
                window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(buttons, text="取消", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="确定", command=accept).pack(side="right", padx=(0, 5))
        tree.bind("<Double-1>", accept)
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        self.wait_window(window)
        return result["slot"]

    def _save_preview_mapping(self, texture_path: Path) -> None:
        if not self.current_model:
            return
        try:
            with Image.open(texture_path) as image:
                image.verify()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"无法读取所选贴图：\n{exc}")
            return
        try:
            slot = self._choose_preview_slot(self.current_model)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"读取模型材质槽失败：\n{exc}")
            return
        if slot is None:
            return
        model_dir = Path(self.current_model["path"]).parent.resolve()
        resolved = texture_path.resolve()
        try:
            stored_path = str(resolved.relative_to(model_dir))
        except ValueError:
            stored_path = str(resolved)
        entry = self._preview_mapping_entry(self.current_model, create=True)
        assert entry is not None
        entry.setdefault("slots", {})[str(slot)] = stored_path
        try:
            save_preview_mappings(self.preview_mappings)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"保存 3D 贴图映射失败：\n{exc}")
            return
        self.status_var.set(f"已保存 3D 贴图映射：槽位 #{slot + 1} → {texture_path.name}")
        self._populate_materials()
        self._render_preview()

    def _map_current_material_for_preview(self) -> None:
        if not self.current_material:
            messagebox.showinfo(APP_TITLE, "请先在左侧选择一张材质。")
            return
        path_text = self.current_material.get("path", "")
        if not path_text or not Path(path_text).is_file():
            messagebox.showinfo(APP_TITLE, "该材质文件未定位，请使用“从文件选择并映射”。")
            return
        self._save_preview_mapping(Path(path_text))

    def _map_file_for_preview(self) -> None:
        if not self.current_model:
            messagebox.showinfo(APP_TITLE, "请先选择一个模型。")
            return
        value = filedialog.askopenfilename(
            title="选择仅用于 3D 预览的贴图",
            initialdir=str(Path(self.current_model["path"]).parent),
            filetypes=(("图片", "*.dds *.tga *.png *.bmp *.jpg *.jpeg"), ("所有文件", "*.*")),
        )
        if value:
            self._save_preview_mapping(Path(value))

    def _clear_preview_mapping(self) -> None:
        if not self.current_model:
            return
        entry = self._preview_mapping_entry(self.current_model)
        slots_data = entry.get("slots", {}) if entry else {}
        if not isinstance(slots_data, dict):
            slots_data = {}
        configured = sorted(int(slot) for slot in slots_data if str(slot).isdigit())
        if not configured:
            messagebox.showinfo(APP_TITLE, "当前模型没有手动 3D 贴图映射。")
            return
        try:
            slot = self._choose_preview_slot(self.current_model, configured)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"读取模型材质槽失败：\n{exc}")
            return
        if slot is None:
            return
        slots_data.pop(str(slot), None)
        if not slots_data:
            self.preview_mappings.get("models", {}).pop(self._preview_mapping_key(self.current_model), None)
        try:
            save_preview_mappings(self.preview_mappings)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"保存 3D 贴图映射失败：\n{exc}")
            return
        self.status_var.set(f"已清除槽位 #{slot + 1} 的 3D 贴图映射")
        self._populate_materials()
        self._render_preview()

    def _choose_replacement(self) -> None:
        if not self.current_material:
            messagebox.showinfo(APP_TITLE, "请先选择一个材质。")
            return
        value = filedialog.askopenfilename(
            title="选择替换图片",
            filetypes=(("图片", "*.dds *.tga *.png *.bmp"), ("所有文件", "*.*")),
        )
        if not value:
            return
        path = Path(value)
        try:
            self.replacement_image.show(path)
            with Image.open(path) as image:
                self.replacement_var.set(f"{path} · {image.width}×{image.height} · {image.format}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"读取替换图片失败：\n{exc}")
            return
        self.replacement_var.set(str(path))
        target = self.current_material.get("path", "")
        if target:
            self.replacement_map[target] = path
            selection = self.material_tree.selection()
            if selection:
                self.material_tree.set(selection[0], "replacement", path.name)

    def _clear_replacements(self) -> None:
        self.replacement_map.clear()
        self.replacement_var.set("")
        self.replacement_image.clear("选择替换图片")
        self._populate_materials()

    @staticmethod
    def _dds_format(path: Path) -> str:
        try:
            with Image.open(path) as image:
                for tile in image.tile:
                    args = tile.args
                    if isinstance(args, tuple) and len(args) > 1 and str(args[1]).startswith("DXT"):
                        return str(args[1])
        except Exception:
            pass
        return "DXT5"

    def _apply_replacement(self) -> None:
        if not self.replacement_map:
            messagebox.showinfo(APP_TITLE, "请先为一个或多个材质选择替换图片。")
            return
        pairs: list[tuple[Path, Path]] = []
        for target_text, source in self.replacement_map.items():
            target = Path(target_text)
            if not source.is_file() or not target.is_file():
                messagebox.showerror(APP_TITLE, f"文件不存在：\n{source}\n{target}")
                return
            try:
                with Image.open(source) as new_image:
                    width, height = new_image.size
                    if width > 8192 or height > 8192 or width < 4 or height < 4:
                        raise ValueError(f"{source.name} 尺寸需在 4×4 到 8192×8192 之间")
            except Exception as exc:
                messagebox.showerror(APP_TITLE, str(exc))
                return
            pairs.append((source, target))
        try:
            backup = self._apply_material_pairs(pairs, "批量材质替换")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"应用失败：\n{exc}")
            return
        self.status_var.set(f"已替换 {len(pairs)} 个材质；备份位于 {backup}")
        if self.current_material and self.current_material.get("path"):
            target = Path(self.current_material["path"])
            self.current_image.show(target)
        self.replacement_map.clear()
        self._populate_materials()
        self._render_preview()

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and value & (value - 1) == 0

    @classmethod
    def _save_dds_mipmapped(cls, source: Path, target: Path, pixel_format: str) -> None:
        with Image.open(source) as opened:
            current = opened.convert("RGBA")
        levels: list[bytes] = []
        header: bytearray | None = None
        while True:
            stream = io.BytesIO()
            current.save(stream, format="DDS", pixel_format=pixel_format)
            payload = stream.getvalue()
            if header is None:
                header = bytearray(payload[:128])
            levels.append(payload[128:])
            if current.size == (1, 1):
                break
            size = (max(1, current.width // 2), max(1, current.height // 2))
            current = current.convert("RGBa").resize(size, Image.Resampling.BOX).convert("RGBA")
        assert header is not None
        flags = struct.unpack_from("<I", header, 8)[0] | 0x20000
        struct.pack_into("<I", header, 8, flags)
        struct.pack_into("<I", header, 28, len(levels))
        caps = struct.unpack_from("<I", header, 108)[0] | 0x8 | 0x400000 | 0x1000
        struct.pack_into("<I", header, 108, caps)
        target.write_bytes(bytes(header) + b"".join(levels))

    def _write_material(self, source: Path, target: Path, temp: Path) -> None:
        if source.suffix.casefold() == target.suffix.casefold() == ".dds":
            shutil.copy2(source, temp)
        elif target.suffix.casefold() == ".dds":
            self._save_dds_mipmapped(source, temp, self._dds_format(target))
        elif source.suffix.casefold() == target.suffix.casefold():
            shutil.copy2(source, temp)
        else:
            with Image.open(source) as image:
                image.convert("RGBA").save(temp, format=target.suffix.lstrip(".").upper())
        with Image.open(temp) as check:
            width, height = check.size
            check.verify()
        if width > 8192 or height > 8192:
            raise ValueError(f"贴图尺寸过大：{width}×{height}")

    def _apply_material_pairs(self, pairs: list[tuple[Path, Path]], label: str) -> Path:
        if self.client_data is None:
            raise RuntimeError("请先设置游戏路径")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_root = BACKUP_DIR / timestamp
        records = []
        for source, target in pairs:
            relative = target.relative_to(self.client_data)
            saved = backup_root / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            records.append({
                "target": str(target), "relative": str(relative), "backup": str(saved),
                "original_sha256": sha256(target), "replacement": str(source),
                "replacement_sha256": sha256(source),
            })
        manifest = {"created": datetime.now().isoformat(timespec="seconds"), "label": label, "files": records}
        (backup_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        completed: list[dict] = []
        try:
            for record, (source, target) in zip(records, pairs):
                temp = target.with_name(target.name + ".modtmp")
                self._write_material(source, target, temp)
                os.replace(temp, target)
                completed.append(record)
        except Exception:
            for record in completed:
                shutil.copy2(record["backup"], record["target"])
            raise
        return backup_root

    def _apply_blend_resources(
        self,
        staged_wzm: Path,
        target_wzm: Path,
        material_pairs: list[tuple[Path, Path]],
        label: str,
        staged_wzu: Path | None = None,
        target_wzu: Path | None = None,
    ) -> Path:
        if self.client_data is None:
            raise RuntimeError("请先设置游戏路径")
        if not staged_wzm.is_file() or not target_wzm.is_file():
            raise FileNotFoundError(f"模型文件不存在：\n{staged_wzm}\n{target_wzm}")
        from wzm_unpack import parse_wzm
        staged_model = parse_wzm(staged_wzm)
        original_model = parse_wzm(target_wzm)
        if staged_model.version != original_model.version:
            raise ValueError("回流 WZM 版本与原模型不一致")
        if [bone.name for bone in staged_model.bones] != [bone.name for bone in original_model.bones]:
            raise ValueError("回流 WZM 骨骼结构与原模型不一致")

        unique_materials: dict[str, tuple[Path, Path]] = {}
        for source, target in material_pairs:
            if not source.is_file():
                raise FileNotFoundError(f"Blender 提取的贴图文件不存在：\n{source}")
            unique_materials[str(target.resolve()).casefold()] = (source, target)
        pairs = [(staged_wzm, target_wzm, "model")]
        if staged_wzu is not None or target_wzu is not None:
            if staged_wzu is None or target_wzu is None or not staged_wzu.is_file() or not target_wzu.is_file():
                raise FileNotFoundError(f"WZU 回流文件不存在：\n{staged_wzu}\n{target_wzu}")
            pairs.append((staged_wzu, target_wzu, "unit"))
        pairs.extend((source, target, "material") for source, target in unique_materials.values())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_root = BACKUP_DIR / timestamp
        records: list[dict] = []
        for source, target, kind in pairs:
            relative = target.resolve().relative_to(self.client_data.resolve())
            saved = backup_root / relative
            existed = target.is_file()
            if existed:
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
            records.append({
                "target": str(target), "relative": str(relative),
                "backup": str(saved) if existed else "", "existed": existed,
                "kind": kind, "original_sha256": sha256(target) if existed else "",
                "replacement": str(source), "replacement_sha256": sha256(source),
            })
        manifest = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "files": records,
        }
        (backup_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        completed: list[dict] = []
        temporary_files: list[Path] = []
        try:
            for record, (source, target, kind) in zip(records, pairs):
                temp = target.with_name(target.name + ".modtmp")
                temporary_files.append(temp)
                if kind in ("model", "unit"):
                    shutil.copy2(source, temp)
                    if kind == "model":
                        parse_wzm(temp)
                else:
                    self._write_material(source, target, temp)
                os.replace(temp, target)
                completed.append(record)
        except Exception:
            for temp in temporary_files:
                temp.unlink(missing_ok=True)
            for record in completed:
                if record.get("existed", True):
                    shutil.copy2(record["backup"], record["target"])
                else:
                    Path(record["target"]).unlink(missing_ok=True)
            raise
        return backup_root

    def _restore_latest(self) -> None:
        backups = sorted((path for path in BACKUP_DIR.iterdir() if path.is_dir()), reverse=True) if BACKUP_DIR.is_dir() else []
        if not backups:
            messagebox.showinfo(APP_TITLE, "没有可恢复的备份。")
            return
        manifest_path = backups[0] / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("files") or [manifest]
            for record in records:
                if record.get("existed", True):
                    shutil.copy2(record["backup"], record["target"])
                else:
                    Path(record["target"]).unlink(missing_ok=True)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"恢复失败：\n{exc}")
            return
        label = "、".join(record["relative"] for record in records[:3])
        self.status_var.set(f"已恢复 {label}")
        current_target = Path(self.current_material["path"]) if self.current_material and self.current_material.get("path") else None
        if current_target and any(Path(record["target"]) == current_target for record in records):
            self.current_image.show(current_target)
        if self.current_model and any(Path(record["target"]) == Path(self.current_model["path"]) for record in records):
            self.current_model["bones"] = -1
            self.current_model["submeshes"] = -1
            self.current_model["triangles"] = -1
            self._refresh_current_model_materials()
            self._load_model_details(self.current_model)
        self._render_preview()

    @staticmethod
    def _pixel_digest(path: Path) -> tuple[tuple[int, int], str]:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            return rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest()

    def _refresh_current_model_materials(self) -> None:
        if not self.current_model:
            return
        from mod_resource_index import _texture_rows
        model_path = Path(self.current_model["path"])
        files = {path.name.casefold(): path for path in model_path.parent.iterdir() if path.is_file()}
        wzu_text = self.current_model.get("wzu", "")
        wzu_path = Path(wzu_text) if wzu_text and Path(wzu_text).is_file() else None
        self.current_model["materials"] = _texture_rows(
            model_path, model_path.read_bytes(), wzu_path, files,
        )
        self.current_material = None
        self._populate_materials()

    def _import_blend(self) -> None:
        if not self.current_model:
            messagebox.showinfo(APP_TITLE, "请先选择一个模型。")
            return
        blend = filedialog.askopenfilename(title="选择编辑后的 Blender 文件", filetypes=(("Blender", "*.blend"),))
        if not blend:
            return
        blender = self._find_blender()
        if blender is None:
            messagebox.showerror(APP_TITLE, "未找到 Blender。")
            return
        model_record = self.current_model
        stage = TOOL_DIR / "mod_staging" / uuid.uuid4().hex
        stage.mkdir(parents=True, exist_ok=True)
        script = stage / "export_blend.py"
        script.write_text(BLENDER_EXPORT_SCRIPT, encoding="utf-8")
        gltf = stage / "edited.gltf"
        shader_manifest = stage / "shader_materials.json"
        extracted = stage / "shader_images"
        self.status_var.set("正在读取 Blender 修改并计算差异")
        def worker():
            try:
                subprocess.run(
                    [
                        str(blender), "--background", "--python", str(script), "--",
                        blend, str(gltf), str(shader_manifest), str(extracted),
                    ],
                    check=True, timeout=180, creationflags=0x08000000,
                )
                report = self._compare_blend_export(model_record, gltf, shader_manifest)
                self.after(0, self._show_blend_diff, Path(blend), stage, report)
            except Exception as exc:
                self.after(0, messagebox.showerror, APP_TITLE, f"读取 Blender 文件失败：\n{exc}")
                self.after(0, self.status_var.set, "Blender 差异分析失败")
        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _clean_material_name(value: str) -> str:
        value = re.sub(r"\.\d{3}$", "", value.casefold())
        return re.sub(r"[^a-z0-9]+", "", value)

    @staticmethod
    def _game_texture_name(
        model_stem: str,
        slot: int,
        role: str,
        image_name: str,
        used: set[str],
    ) -> str:
        model_part = re.sub(r"[^A-Za-z0-9_-]+", "_", model_stem).strip("_") or "model"
        image_part = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(image_name).stem).strip("_") or "image"
        base = f"{model_part}_{slot + 1}_{role}_{image_part}"[:220].rstrip("_")
        candidate = f"{base}.dds"
        counter = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{counter}.dds"
            counter += 1
        used.add(candidate.casefold())
        return candidate

    def _compare_blend_export(
        self,
        model_record: dict,
        gltf_path: Path,
        shader_manifest_path: Path,
    ) -> dict:
        from wzm_unpack import parse_wzm
        from wzm_repack import repack_wzm_from_gltf, rewrite_wzu_texture_names
        original = parse_wzm(model_record["path"])
        edited = json.loads(gltf_path.read_text(encoding="utf-8"))
        shader_document = json.loads(shader_manifest_path.read_text(encoding="utf-8"))
        shader_materials = [row for row in shader_document.get("materials", []) if isinstance(row, dict)]
        shader_by_name = {
            self._clean_material_name(str(row.get("name", ""))): row
            for row in shader_materials
            if self._clean_material_name(str(row.get("name", "")))
        }
        gltf_materials = edited.get("materials", [])
        used_material_names: list[str] = []
        for mesh in edited.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                material_index = primitive.get("material")
                if isinstance(material_index, int) and 0 <= material_index < len(gltf_materials):
                    name = str(gltf_materials[material_index].get("name", ""))
                    if name and name not in used_material_names:
                        used_material_names.append(name)
        used_shader_materials = [
            shader_by_name[key]
            for key in (self._clean_material_name(name) for name in used_material_names)
            if key in shader_by_name
        ]

        shader_bindings: list[dict | None] = []
        shader_warnings: list[str] = []
        if len(original.submeshes) == 1:
            candidates = sorted(
                used_shader_materials or shader_materials,
                key=lambda row: int(row.get("face_count", 0)), reverse=True,
            )
            shader_bindings = [candidates[0] if candidates else None]
            textured = [row for row in candidates if row.get("roles", {}).get("diffuse")]
            if len(textured) > 1:
                shader_warnings.append(
                    f"原 WZM 只有一个材质槽，但 Blender 有 {len(textured)} 个带基础色图片的材质；"
                    f"已按使用面数选择 {candidates[0].get('name', '第一个材质')}。若要同时保留多套材质，请先在 Blender 烘焙为一张贴图。"
                )
        else:
            ordered = used_shader_materials or shader_materials
            for index, submesh in enumerate(original.submeshes):
                key = self._clean_material_name(submesh.diffuse)
                binding = shader_by_name.get(key)
                if binding is None and index < len(ordered):
                    binding = ordered[index]
                    shader_warnings.append(
                        f"材质槽 #{index + 1} 未按名称匹配，已按顺序使用 Blender 材质 {binding.get('name', '')}"
                    )
                shader_bindings.append(binding)

        material_names: dict[int, dict[str, str]] = {}
        texture_diffs: list[dict] = []
        used_texture_names: set[str] = set()
        model_dir = Path(model_record["path"]).parent
        role_targets = {
            role: [
                row for row in model_record.get("materials", [])
                if row.get("role") == role and row.get("name")
            ]
            for role in ("diffuse", "normal", "specular")
        }
        wzu_path = Path(model_record["wzu"]) if model_record.get("wzu") else None
        wzu_replacements: dict[str, str] = {}

        def neutral_texture(role: str) -> dict:
            neutral_dir = gltf_path.parent / "shader_images"
            neutral_dir.mkdir(parents=True, exist_ok=True)
            if role == "normal":
                path = neutral_dir / "__neutral_normal.png"
                color = (128, 128, 255, 255)
                label = "Blender 未连接法线（中性法线）"
            else:
                path = neutral_dir / "__neutral_specular.png"
                color = (0, 0, 0, 255)
                label = "Blender 未连接高光（黑色屏蔽）"
            if not path.is_file():
                Image.new("RGBA", (4, 4), color).save(path)
            return {"path": str(path), "image": label, "generated_neutral": True}

        def add_texture(
            slot: int,
            role: str,
            source_info: dict,
            target_name: str,
            required_by_wzm: bool = False,
        ) -> bool:
            source = Path(str(source_info["path"]))
            if not source.is_file():
                shader_warnings.append(f"Blender 图片提取文件不存在：{source}")
                return False
            target = model_dir / Path(target_name).name
            try:
                new_size, new_hash = self._pixel_digest(source)
            except Exception as exc:
                shader_warnings.append(f"无法读取 Blender 图片 {source.name}：{exc}")
                return False
            if target.is_file():
                old_size, old_hash = self._pixel_digest(target)
                original_text = f"{old_size[0]}×{old_size[1]}"
                changed = old_size != new_size or old_hash != new_hash
                status = "从 Blender 着色器提取（将替换）" if changed else "图片内容相同"
            else:
                original_text = "将新建"
                changed = True
                status = "从 Blender 着色器提取（将新建）"
            if source_info.get("generated_neutral"):
                status = "未连接该层；自动写入中性图" if changed else "中性图已存在"
            texture_diffs.append({
                "name": f"槽位 #{slot + 1} · {ROLE_LABELS.get(role, role)} · {source_info.get('image', source.name)}",
                "role": role,
                "target": target,
                "source": source if changed else None,
                "original": original_text,
                "edited": f"{new_size[0]}×{new_size[1]}",
                "status": status,
                "required_by_wzm": required_by_wzm,
            })
            return True

        for slot, (submesh, binding) in enumerate(zip(original.submeshes, shader_bindings)):
            if not binding:
                shader_warnings.append(f"材质槽 #{slot + 1} 没有找到可读取的 Blender 着色器")
                continue
            shader_warnings.extend(
                f"{binding.get('name', f'槽位 #{slot + 1}')}：{warning}"
                for warning in binding.get("warnings", [])
            )
            roles = binding.get("roles", {})
            failed_roles = binding.get("failed_roles", {})
            diffuse = roles.get("diffuse")
            if diffuse:
                name = self._game_texture_name(
                    Path(model_record["path"]).stem, slot, "di", str(diffuse.get("image", "diffuse")),
                    used_texture_names,
                )
                if add_texture(slot, "diffuse", diffuse, name, required_by_wzm=True):
                    material_names.setdefault(slot, {})["diffuse"] = name
                    if wzu_path and role_targets["diffuse"]:
                        old = role_targets["diffuse"][min(slot, len(role_targets["diffuse"]) - 1)]["name"]
                        wzu_replacements[str(old)] = name
            elif "diffuse" not in failed_roles:
                shader_warnings.append(
                    f"{binding.get('name', f'槽位 #{slot + 1}')} 的基础色没有连接图片，保留原 WZM 基础贴图引用"
                )
            specular = roles.get("specular")
            if specular:
                name = self._game_texture_name(
                    Path(model_record["path"]).stem, slot, "sp", str(specular.get("image", "specular")),
                    used_texture_names,
                )
                if add_texture(slot, "specular", specular, name, required_by_wzm=True):
                    material_names.setdefault(slot, {})["specular"] = name
                    if wzu_path and role_targets["specular"]:
                        old = role_targets["specular"][min(slot, len(role_targets["specular"]) - 1)]["name"]
                        wzu_replacements[str(old)] = name
            elif "specular" in failed_roles:
                shader_warnings.append("高光图片已连接但提取失败，因此保留原高光引用")
            else:
                material_names.setdefault(slot, {})["specular"] = ""
                if wzu_path and role_targets["specular"]:
                    old = role_targets["specular"][min(slot, len(role_targets["specular"]) - 1)]["name"]
                    name = self._game_texture_name(
                        Path(model_record["path"]).stem, slot, "sp_off", "neutral",
                        used_texture_names,
                    )
                    if add_texture(slot, "specular", neutral_texture("specular"), name, required_by_wzm=True):
                        wzu_replacements[str(old)] = name
            normal = roles.get("normal")
            if normal:
                if role_targets["normal"]:
                    normal_target = role_targets["normal"][min(slot, len(role_targets["normal"]) - 1)]
                    if wzu_path:
                        name = self._game_texture_name(
                            Path(model_record["path"]).stem, slot, "no", str(normal.get("image", "normal")),
                            used_texture_names,
                        )
                        if add_texture(slot, "normal", normal, name, required_by_wzm=True):
                            wzu_replacements[str(normal_target["name"])] = name
                    else:
                        add_texture(slot, "normal", normal, str(normal_target["name"]))
                else:
                    shader_warnings.append(
                        f"{binding.get('name', f'槽位 #{slot + 1}')} 含法线图片，但原 WZU 没有法线引用；"
                        "当前只回流模型与 WZM 材质，因此未写入这张法线图"
                    )
            elif "normal" in failed_roles:
                shader_warnings.append("法线图片已连接但提取失败，因此保留原法线引用")
            elif wzu_path and role_targets["normal"]:
                old = role_targets["normal"][min(slot, len(role_targets["normal"]) - 1)]["name"]
                name = self._game_texture_name(
                    Path(model_record["path"]).stem, slot, "no_off", "neutral",
                    used_texture_names,
                )
                if add_texture(slot, "normal", neutral_texture("normal"), name, required_by_wzm=True):
                    wzu_replacements[str(old)] = name

        staged_wzm = gltf_path.with_name("edited.WZM")
        repack = repack_wzm_from_gltf(
            model_record["path"], gltf_path, staged_wzm, material_names=material_names,
        )
        staged_wzu = None
        if wzu_path and wzu_path.is_file() and wzu_replacements:
            staged_wzu = gltf_path.with_name("edited.WZU")
            replaced_count = rewrite_wzu_texture_names(wzu_path, staged_wzu, wzu_replacements)
            shader_warnings.append(f"已按 Blender 着色器更新 WZU 中的 {replaced_count} 个贴图引用")
        accessors = edited.get("accessors", [])
        edited_vertices = 0
        edited_indices = 0
        edited_submeshes = 0
        edited_minimum: list[float] | None = None
        edited_maximum: list[float] | None = None
        has_uv = False
        has_weights = False
        for mesh in edited.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                edited_submeshes += 1
                attrs = primitive.get("attributes", {})
                if "POSITION" in attrs:
                    position_accessor = accessors[attrs["POSITION"]]
                    edited_vertices += position_accessor["count"]
                    minimum = position_accessor.get("min")
                    maximum = position_accessor.get("max")
                    if minimum and maximum:
                        edited_minimum = minimum if edited_minimum is None else [min(a, b) for a, b in zip(edited_minimum, minimum)]
                        edited_maximum = maximum if edited_maximum is None else [max(a, b) for a, b in zip(edited_maximum, maximum)]
                if "indices" in primitive:
                    edited_indices += accessors[primitive["indices"]]["count"]
                has_uv |= "TEXCOORD_0" in attrs
                has_weights |= "JOINTS_0" in attrs and "WEIGHTS_0" in attrs
        original_vertices = sum(len(mesh.vertices) for mesh in original.submeshes)
        original_triangles = sum(len(mesh.indices) // 3 for mesh in original.submeshes)
        original_points = [
            (original.positions[vertex.position_index][0] * 0.01,
             original.positions[vertex.position_index][2] * 0.01,
             -original.positions[vertex.position_index][1] * 0.01)
            for mesh in original.submeshes for vertex in mesh.vertices
        ]
        original_minimum = [min(point[axis] for point in original_points) for axis in range(3)]
        original_maximum = [max(point[axis] for point in original_points) for axis in range(3)]
        bounds_same = bool(edited_minimum and edited_maximum) and all(
            abs(a - b) < 1e-4
            for a, b in zip((*original_minimum, *original_maximum), (*edited_minimum, *edited_maximum))
        )
        edited_bones = []
        nodes = edited.get("nodes", [])
        for skin in edited.get("skins", []):
            edited_bones.extend(nodes[index].get("name", f"bone_{index}") for index in skin.get("joints", []))
        original_bones = [bone.name for bone in original.bones]
        triangles_same = original_triangles == edited_indices // 3
        geometry_same = triangles_same and bounds_same
        vertex_status = "相同" if original_vertices == edited_vertices else "Blender 重排" if geometry_same else "变化"
        material_slot_status = (
            "相同" if len(original.submeshes) == edited_submeshes
            else "自动合并" if len(original.submeshes) == 1 and edited_submeshes > 0
            else "变化"
        )

        return {
            "summary": [
                ("顶点", str(original_vertices), str(edited_vertices), vertex_status),
                ("三角形", str(original_triangles), str(edited_indices // 3), "相同" if triangles_same else "变化"),
                ("空间边界", " / ".join(f"{v:.3f}" for v in (*original_minimum, *original_maximum)), " / ".join(f"{v:.3f}" for v in (*(edited_minimum or []), *(edited_maximum or []))), "相同" if bounds_same else "变化"),
                ("骨骼", str(len(original_bones)), str(len(edited_bones)), "相同" if set(original_bones) == set(edited_bones) else "变化"),
                ("法线", str(original_vertices), str(repack.normal_count), "将写回"),
                ("UV", "存在", "存在" if has_uv else "缺失", "相同" if has_uv else "变化"),
                ("权重", "存在", "存在" if has_weights else "缺失", "相同" if has_weights else "变化"),
                ("材质槽", str(len(original.submeshes)), str(edited_submeshes), material_slot_status),
            ],
            "textures": texture_diffs,
            "staged_wzm": staged_wzm,
            "target_wzm": Path(model_record["path"]),
            "target_wzm_sha256": sha256(Path(model_record["path"])),
            "staged_wzu": staged_wzu,
            "target_wzu": wzu_path,
            "target_wzu_sha256": sha256(wzu_path) if wzu_path and wzu_path.is_file() else None,
            "repack_warnings": [*repack.warnings, *shader_warnings],
        }

    def _show_blend_diff(self, blend: Path, stage: Path, report: dict) -> None:
        self.status_var.set(f"Blender 差异分析完成：{blend.name}")
        window = tk.Toplevel(self)
        window.title(f"Blender 文件差异 · {blend.name}")
        window.geometry("980x700")
        frame = ttk.Frame(window, padding=8)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=("original", "edited", "status"), show="tree headings")
        tree.heading("#0", text="项目")
        tree.heading("original", text="原始资源")
        tree.heading("edited", text="Blender 修改")
        tree.heading("status", text="状态")
        tree.column("#0", width=260)
        tree.column("original", width=170)
        tree.column("edited", width=170)
        tree.column("status", width=140, anchor="center")
        geometry = tree.insert("", "end", text="模型结构", open=True)
        for name, old, new, status in report["summary"]:
            tree.insert(geometry, "end", text=name, values=(old, new, status))
        textures = tree.insert("", "end", text="材质贴图", open=True)
        texture_by_item: dict[str, dict] = {}
        for row in report["textures"]:
            item = tree.insert(
                textures, "end", text=row["name"],
                values=(row["original"], row["edited"], row["status"]),
            )
            texture_by_item[item] = row
        if report.get("repack_warnings"):
            warnings = tree.insert("", "end", text="回流提示", open=True)
            for warning in report["repack_warnings"]:
                tree.insert(warnings, "end", text=warning, values=("", "", "提示"))
        tree.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(buttons, text="打开临时导出目录", command=lambda: os.startfile(stage)).pack(side="left")

        def choose_texture() -> None:
            selection = tree.selection()
            row = texture_by_item.get(selection[0]) if selection else None
            if row is None:
                messagebox.showinfo(APP_TITLE, "请先在“材质贴图”下选择一项。", parent=window)
                return
            value = filedialog.askopenfilename(
                parent=window,
                title=f"为 {row['name']} 选择导入贴图",
                initialdir=str(stage),
                filetypes=(("图片", "*.dds *.tga *.png *.bmp *.jpg *.jpeg"), ("所有文件", "*.*")),
            )
            if not value:
                return
            source = Path(value)
            try:
                with Image.open(source) as image:
                    width, height = image.size
                    image.verify()
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"无法读取贴图：\n{exc}", parent=window)
                return
            row["source"] = source
            row["edited"] = f"{width}×{height}"
            row["status"] = "手动选择（将导入）"
            tree.item(selection[0], values=(row["original"], row["edited"], row["status"]))
            refresh_apply_button()

        def ignore_texture() -> None:
            selection = tree.selection()
            row = texture_by_item.get(selection[0]) if selection else None
            if row is None:
                return
            if row.get("required_by_wzm") and not Path(row["target"]).is_file():
                messagebox.showinfo(
                    APP_TITLE,
                    "这张贴图使用了新的文件名，并已写入待应用的 WZM。\n"
                    "若不创建对应图片，游戏会找不到材质；可以改选另一张图片，但不能直接跳过。",
                    parent=window,
                )
                return
            row["source"] = None
            row["status"] = "不导入"
            tree.item(selection[0], values=(row["original"], row["edited"], row["status"]))
            refresh_apply_button()

        ttk.Button(buttons, text="替换自动提取结果", command=choose_texture).pack(side="left", padx=(5, 0))
        ttk.Button(buttons, text="不导入选中贴图", command=ignore_texture).pack(side="left", padx=(5, 0))
        apply_button = ttk.Button(
            buttons,
            command=lambda: self._apply_blend_changes(window, blend, report),
        )
        apply_button.pack(side="right")

        def refresh_apply_button() -> None:
            count = sum(1 for row in report["textures"] if row.get("source"))
            apply_button.configure(text=f"备份并应用模型/法线/权重 + {count} 张着色器贴图")

        refresh_apply_button()

    def _apply_blend_changes(self, window: tk.Toplevel, blend: Path, report: dict) -> None:
        target_wzm = Path(report["target_wzm"])
        if not target_wzm.is_file() or sha256(target_wzm) != report.get("target_wzm_sha256"):
            messagebox.showerror(
                APP_TITLE,
                "原 WZM 在差异分析后已发生变化。为避免覆盖其他修改，请关闭窗口后重新导入 Blender 文件。",
                parent=window,
            )
            return
        target_wzu = Path(report["target_wzu"]) if report.get("target_wzu") else None
        if target_wzu and (
            not target_wzu.is_file() or sha256(target_wzu) != report.get("target_wzu_sha256")
        ):
            messagebox.showerror(
                APP_TITLE,
                "原 WZU 在差异分析后已发生变化。为避免覆盖其他修改，请关闭窗口后重新导入 Blender 文件。",
                parent=window,
            )
            return
        material_pairs = [
            (Path(row["source"]), Path(row["target"]))
            for row in report["textures"] if row.get("source")
        ]
        try:
            backup = self._apply_blend_resources(
                Path(report["staged_wzm"]), target_wzm, material_pairs,
                f"Blender 模型回流：{blend.name}",
                staged_wzu=Path(report["staged_wzu"]) if report.get("staged_wzu") else None,
                target_wzu=target_wzu,
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"应用 Blender 修改失败：\n{exc}", parent=window)
            return
        window.destroy()
        self.status_var.set(f"已应用 Blender 模型与 {len(material_pairs)} 张贴图；备份位于 {backup}")
        if self.current_model:
            self.current_model["bones"] = -1
            self.current_model["submeshes"] = -1
            self.current_model["triangles"] = -1
            self._refresh_current_model_materials()
            self._load_model_details(self.current_model)
        self._render_preview()

    def _refresh_preview_after_edit(self) -> None:
        self.tabs.select(self.browser_tab)
        self._render_preview()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--viewer":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from wzm_viewer import main as viewer_main
        raise SystemExit(viewer_main())
    app = ModEditor()
    app.mainloop()


if __name__ == "__main__":
    main()

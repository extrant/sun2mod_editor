from __future__ import annotations

import json
import io
import hashlib
import os
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
        self.material_tree = ttk.Treeview(left, columns=("role", "reuse", "replacement"), show="tree headings", selectmode="browse")
        self.material_tree.heading("#0", text="材质文件")
        self.material_tree.heading("role", text="作用")
        self.material_tree.heading("reuse", text="复用")
        self.material_tree.heading("replacement", text="待替换")
        self.material_tree.column("#0", width=235)
        self.material_tree.column("role", width=100, anchor="center")
        self.material_tree.column("reuse", width=60, anchor="center")
        self.material_tree.column("replacement", width=150)
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
        for row in self.current_model["materials"]:
            reuse = self.index_data["materials"].get(row["name"].casefold(), {}).get("model_count", 0)
            replacement = self.replacement_map.get(row.get("path", ""))
            item = self.material_tree.insert("", "end", text=row["name"], values=(
                ROLE_LABELS.get(row["role"], row["role"]), reuse, replacement.name if replacement else "",
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
        command = viewer_command(
            self.current_model["path"],
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
        subprocess.Popen(viewer_command(self.current_model["path"], "--no-effect"), cwd=str(TOOL_DIR))

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
        self.status_var.set(f"正在导出 Blender 包：{model_record['name']}")
        def worker():
            try:
                from wzm_unpack import export_wzm, parse_wzm
                model = parse_wzm(model_record["path"])
                output = export_wzm(
                    model,
                    TOOL_DIR / "blender_exports",
                    companion_unit=model_record.get("wzu") or None,
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
                shutil.copy2(record["backup"], record["target"])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"恢复失败：\n{exc}")
            return
        label = "、".join(record["relative"] for record in records[:3])
        self.status_var.set(f"已恢复 {label}")
        current_target = Path(self.current_material["path"]) if self.current_material and self.current_material.get("path") else None
        if current_target and any(Path(record["target"]) == current_target for record in records):
            self.current_image.show(current_target)
        self._render_preview()

    @staticmethod
    def _pixel_digest(path: Path) -> tuple[tuple[int, int], str]:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            return rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest()

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
        script.write_text(
            "import bpy, sys\n"
            "blend, gltf = sys.argv[sys.argv.index('--') + 1:]\n"
            "bpy.ops.wm.open_mainfile(filepath=blend)\n"
            "bpy.ops.export_scene.gltf(filepath=gltf, export_format='GLTF_SEPARATE', "
            "export_texture_dir='textures', export_skins=True, export_morph=False, export_materials='EXPORT')\n",
            encoding="utf-8",
        )
        gltf = stage / "edited.gltf"
        self.status_var.set("正在读取 Blender 修改并计算差异")
        def worker():
            try:
                subprocess.run(
                    [str(blender), "--background", "--python", str(script), "--", blend, str(gltf)],
                    check=True, timeout=180, creationflags=0x08000000,
                )
                report = self._compare_blend_export(model_record, gltf)
                self.after(0, self._show_blend_diff, Path(blend), stage, report)
            except Exception as exc:
                self.after(0, messagebox.showerror, APP_TITLE, f"读取 Blender 文件失败：\n{exc}")
                self.after(0, self.status_var.set, "Blender 差异分析失败")
        threading.Thread(target=worker, daemon=True).start()

    def _compare_blend_export(self, model_record: dict, gltf_path: Path) -> dict:
        from wzm_unpack import parse_wzm
        original = parse_wzm(model_record["path"])
        edited = json.loads(gltf_path.read_text(encoding="utf-8"))
        accessors = edited.get("accessors", [])
        edited_vertices = 0
        edited_indices = 0
        edited_minimum: list[float] | None = None
        edited_maximum: list[float] | None = None
        has_uv = False
        has_weights = False
        for mesh in edited.get("meshes", []):
            for primitive in mesh.get("primitives", []):
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

        images: dict[str, Path] = {}
        for image in edited.get("images", []):
            uri = unquote(image.get("uri", ""))
            if uri and not uri.startswith("data:"):
                path = (gltf_path.parent / uri).resolve()
                if path.is_file():
                    images[Path(uri).stem.casefold()] = path
                    if image.get("name"):
                        images[str(image["name"]).casefold()] = path
        material_pairs: list[tuple[Path, Path]] = []
        texture_diffs = []
        for material in model_record.get("materials", []):
            if not material.get("path"):
                continue
            target = Path(material["path"])
            source = images.get(Path(material["name"]).stem.casefold())
            if source is None:
                texture_diffs.append((material["name"], "存在", "未导出", "未匹配"))
                continue
            old_size, old_hash = self._pixel_digest(target)
            new_size, new_hash = self._pixel_digest(source)
            changed = old_size != new_size or old_hash != new_hash
            texture_diffs.append((material["name"], f"{old_size[0]}×{old_size[1]}", f"{new_size[0]}×{new_size[1]}", "已修改" if changed else "相同"))
            if changed:
                material_pairs.append((source, target))
        return {
            "summary": [
                ("顶点", str(original_vertices), str(edited_vertices), vertex_status),
                ("三角形", str(original_triangles), str(edited_indices // 3), "相同" if triangles_same else "变化"),
                ("空间边界", " / ".join(f"{v:.3f}" for v in (*original_minimum, *original_maximum)), " / ".join(f"{v:.3f}" for v in (*(edited_minimum or []), *(edited_maximum or []))), "相同" if bounds_same else "变化"),
                ("骨骼", str(len(original_bones)), str(len(edited_bones)), "相同" if set(original_bones) == set(edited_bones) else "变化"),
                ("UV", "存在", "存在" if has_uv else "缺失", "相同" if has_uv else "变化"),
                ("权重", "存在", "存在" if has_weights else "缺失", "相同" if has_weights else "变化"),
                ("材质槽", str(len(original.submeshes)), str(len(edited.get("materials", []))), "变化" if len(original.submeshes) != len(edited.get("materials", [])) else "相同"),
            ],
            "textures": texture_diffs,
            "material_pairs": material_pairs,
        }

    def _show_blend_diff(self, blend: Path, stage: Path, report: dict) -> None:
        self.status_var.set(f"Blender 差异分析完成：{blend.name}")
        window = tk.Toplevel(self)
        window.title(f"Blender 文件差异 · {blend.name}")
        window.geometry("900x650")
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
        tree.column("status", width=100, anchor="center")
        geometry = tree.insert("", "end", text="模型结构", open=True)
        for name, old, new, status in report["summary"]:
            tree.insert(geometry, "end", text=name, values=(old, new, status))
        textures = tree.insert("", "end", text="材质贴图", open=True)
        for name, old, new, status in report["textures"]:
            tree.insert(textures, "end", text=name, values=(old, new, status))
        tree.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(buttons, text="打开临时导出目录", command=lambda: os.startfile(stage)).pack(side="left")
        apply_button = ttk.Button(
            buttons, text=f"备份并应用 {len(report['material_pairs'])} 个材质差异",
            command=lambda: self._apply_blend_materials(window, blend, report["material_pairs"]),
        )
        apply_button.pack(side="right")
        if not report["material_pairs"]:
            apply_button.state(["disabled"])

    def _apply_blend_materials(self, window: tk.Toplevel, blend: Path, pairs: list[tuple[Path, Path]]) -> None:
        try:
            backup = self._apply_material_pairs(pairs, f"Blender 回流：{blend.name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"应用 Blender 材质失败：\n{exc}")
            return
        window.destroy()
        self.status_var.set(f"已应用 {len(pairs)} 个 Blender 材质差异；备份位于 {backup}")
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

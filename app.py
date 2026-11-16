"""Friendly, local-only admissions document intake and roster application.

Install the requirements in this folder and start with ``python app.py``.
For an offline PyInstaller build, collect RapidOCR/ONNX Runtime and model
resources, then verify the built app on a clean offline computer.
"""

from __future__ import annotations

import atexit
import os
import queue
import subprocess
import sys
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageOps
from tkinter import TclError, filedialog, messagebox

from excel_manager import ExcelManager, ImportedStudent
from local_receiver import LocalPhotoReceiver
from ocr_engine import (
    DOCUMENT_TYPES,
    DOC_FIELDS,
    ApplicationOcrEngine,
    configured_ocr_backend,
)
from security import (
    cleanup_cache_folder,
    create_private_cache,
    sanitize_text,
    shred_file,
    stage_private_copy,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
    DND_MIXIN = TkinterDnD.DnDWrapper
except ImportError:
    DND_FILES = None
    DND_AVAILABLE = False

    class DND_MIXIN:
        """Fallback base when native drag-and-drop support is unavailable."""


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    Path.home() / "Documents" / "SchoolApplicationOCR"
    if getattr(sys, "frozen", False)
    else APP_DIR / "output"
)
DEFAULT_OUTPUT_PATH = (
    OUTPUT_DIR / f"入学願書受付名簿_{date.today():%Y%m%d}.xlsx"
)
GREEN = "#167A4A"
GREEN_HOVER = "#11633B"
GRAY = "#68727C"
IMAGE_TYPES = [
    ("画像・PDF", "*.png *.jpg *.jpeg *.pdf"),
    ("すべてのファイル", "*.*"),
]

APP_FIELD_LABELS = {
    "name": "氏名（漢字）",
    "furigana": "フリガナ",
    "birth_date_year": "生年月日 / 年度",
    "age": "年齢",
    "school_name": "出身校",
    "academic_record": "評定・資格・出欠・活動内容",
    "reason": "志願理由・推薦所見",
    "notes": "特記事項",
}


class AdmissionsOCRApp(ctk.CTk, DND_MIXIN):
    def __init__(self) -> None:
        super().__init__()
        self.title("入試関係書類の読取・名簿")
        self.geometry("1180x850")
        self.minsize(850, 680)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("green")

        self.cache_dir = create_private_cache()
        self.ocr_engine = ApplicationOcrEngine(backend=configured_ocr_backend())
        self.excel_manager = ExcelManager()
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.pending_cleanup_paths: set[Path] = set()
        self.current_image: Path | None = None
        self.current_original: Path | None = None
        self.receiver = LocalPhotoReceiver(
            cache_dir=self.cache_dir,
            on_upload=lambda path: self.event_queue.put(("phone", path)),
        )
        atexit.register(self._emergency_cleanup)
        self.preview_source: Image.Image | None = None
        self.preview_ctk_image: ctk.CTkImage | None = None
        self.output_path = DEFAULT_OUTPUT_PATH
        self.imported_students: list[ImportedStudent] = []
        self.current_doc_type = DOCUMENT_TYPES[0]
        self.field_widgets: dict[str, Any] = {}
        self.field_labels: dict[str, Any] = {}
        self.field_confidence: dict[str, float] = {}
        self.field_review_overrides: set[str] = set()
        self.last_saved_student = ""
        self._ui_scale = 1.0
        self._camera_window: ctk.CTkToplevel | None = None
        self._camera: cv2.VideoCapture | None = None
        self._camera_after_id: str | None = None
        self._closing = False
        self._read_generation = 0
        self._deletion_notice_acknowledged = False
        self._event_after_id: str | None = None

        self.grid_columnconfigure((0, 1), weight=1, uniform="columns")
        self.grid_rowconfigure(2, weight=1)
        self._build_header()
        self._build_image_panel()
        self._build_form_panel()
        self._build_footer()
        self.bind("<Configure>", self._on_resize)
        self.protocol("WM_DELETE_WINDOW", self._close_app)
        self._enable_drag_and_drop()
        self._event_after_id = self.after(120, self._poll_events)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        header.grid_columnconfigure(2, weight=0)
        self.reassurance_label = ctk.CTkLabel(
            header,
            text=(
                "Ollamaを使う設定です。読取にはOllamaと画像モデルの起動が必要です。"
                if self.ocr_engine.backend == "ollama"
                else "✅ 準備ができました。書類を入れると文字を自動で読み取ります。"
            ),
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=("#145C3C", "#9FE0BD"),
            fg_color=("#E8F5EC", "#163D2D"),
            corner_radius=8,
        )
        self.reassurance_label.grid(
            row=0, column=0, columnspan=2, padx=14, pady=(9, 5), sticky="ew"
        )
        ctk.CTkLabel(
            header,
            text="入試関係書類の読取・名簿",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).grid(row=1, column=0, padx=18, pady=(7, 3), sticky="w")
        ctk.CTkLabel(
            header,
            text="🔍 文字サイズ",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=1, column=1, padx=(4, 4), pady=(7, 3), sticky="e")
        self.text_size_switch = ctk.CTkSegmentedButton(
            header,
            values=["標準", "大きめ"],
            command=self._change_text_size,
            width=150,
        )
        self.text_size_switch.set("標準")
        self.text_size_switch.grid(
            row=1, column=2, padx=(4, 16), pady=(7, 3), sticky="e"
        )
        steps = ctk.CTkFrame(header, fg_color=("gray92", "gray20"))
        steps.grid(row=2, column=0, columnspan=3, padx=16, pady=(4, 10), sticky="ew")
        for column, text in enumerate(
            ("① 書類を入れる", "② 写真から文字を自動読取・確認", "③ 名簿に保存")
        ):
            steps.grid_columnconfigure(column, weight=1)
            ctk.CTkLabel(
                steps,
                text=text,
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="center",
            ).grid(row=0, column=column, padx=6, pady=10, sticky="ew")

    def _build_image_panel(self) -> None:
        panel = ctk.CTkFrame(self)
        panel.grid(row=2, column=0, padx=(14, 7), pady=12, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        controls.grid_columnconfigure(0, weight=1)
        self.choose_image_button = ctk.CTkButton(
            controls,
            text="書類の写真・PDFを選ぶ",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._choose_document,
        )
        self.choose_image_button.grid(row=0, column=0, padx=(0, 7), sticky="ew")
        self.phone_button = ctk.CTkButton(
            controls,
            text="📸 スマホで撮影して送る",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#285A9F",
            hover_color="#1E477E",
            command=self._show_phone_qr,
        )
        self.phone_button.grid(row=0, column=1, sticky="ew")
        self.preview_label = ctk.CTkLabel(
            panel,
            text="書類を選ぶか、ここにドラッグしてください",
            fg_color=("gray92", "gray18"),
            corner_radius=8,
            font=ctk.CTkFont(size=15),
        )
        self.preview_label.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="nsew")
        self.image_name = ctk.CTkLabel(panel, text="書類未選択", anchor="w")
        self.image_name.grid(row=2, column=0, padx=12, pady=(0, 4), sticky="ew")
        self.camera_button = ctk.CTkButton(
            panel,
            text="📷 パソコンのカメラで撮影",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._open_camera,
            fg_color="#285A9F",
            hover_color="#1E477E",
            state="disabled",
        )
        self.camera_button.configure(state="normal")
        self.camera_button.grid(row=3, column=0, padx=10, pady=(3, 5), sticky="ew")
        self.reading_label = ctk.CTkLabel(
            panel,
            text="書類を入れると自動で読み取ります",
            anchor="center",
            font=ctk.CTkFont(size=13),
        )
        self.reading_label.grid(row=4, column=0, padx=10, pady=(2, 0), sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(panel, mode="indeterminate")
        self.progress_bar.grid(row=5, column=0, padx=10, pady=(3, 10), sticky="ew")
        self.progress_bar.set(0)

    def _build_form_panel(self) -> None:
        panel = ctk.CTkFrame(self)
        panel.grid(row=2, column=1, padx=(7, 14), pady=12, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            panel,
            text="読み取り結果を確認・修正",
            font=ctk.CTkFont(size=17, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="ew")
        self.doc_type_menu = ctk.CTkOptionMenu(
            panel,
            values=list(DOCUMENT_TYPES),
            command=self._change_document_type,
            font=ctk.CTkFont(size=14),
        )
        self.doc_type_menu.set(self.current_doc_type)
        self.doc_type_menu.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        self.form_scroll = ctk.CTkScrollableFrame(panel)
        self.form_scroll.grid(row=2, column=0, padx=8, pady=5, sticky="nsew")
        self.form_scroll.grid_columnconfigure(0, weight=1)
        self._render_fields()

        destination = ctk.CTkFrame(panel)
        destination.grid(row=3, column=0, padx=8, pady=(6, 9), sticky="ew")
        destination.grid_columnconfigure(0, weight=1)
        self.output_label = ctk.CTkLabel(
            destination, text="", anchor="w", wraplength=500
        )
        self.output_label.grid(row=0, column=0, padx=9, pady=(8, 3), sticky="ew")
        controls = ctk.CTkFrame(destination, fg_color="transparent")
        controls.grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        controls.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            controls,
            text="保存先を選ぶ",
            command=self._select_output,
        ).grid(row=0, column=0, padx=4, sticky="ew")
        ctk.CTkButton(
            controls,
            text="既存の名簿を開く",
            command=self._import_workbook,
        ).grid(row=0, column=1, padx=4, sticky="ew")
        ctk.CTkButton(
            controls,
            text="新しい名簿を作る",
            command=self._create_workbook,
        ).grid(row=1, column=0, padx=4, pady=(5, 0), sticky="ew")
        ctk.CTkButton(
            controls,
            text="入力を消す",
            command=self._clear_fields,
            fg_color=GRAY,
            hover_color="#535C65",
        ).grid(row=1, column=1, padx=4, pady=(5, 0), sticky="ew")
        self.student_menu = ctk.CTkOptionMenu(
            destination,
            values=["名簿から選ぶ（未読込）"],
            command=self._select_imported_student,
            state="disabled",
        )
        self.student_menu.grid(row=2, column=0, padx=8, pady=4, sticky="ew")
        self.save_button = ctk.CTkButton(
            destination,
            text="名簿に保存して写真を消去",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=GREEN,
            hover_color=GREEN_HOVER,
            command=self._save_record,
        )
        self.save_button.grid(row=3, column=0, padx=8, pady=(4, 9), sticky="ew")
        self._update_output_label()

    def _render_fields(self) -> None:
        for child in self.form_scroll.winfo_children():
            child.destroy()
        self.field_widgets.clear()
        self.field_labels.clear()
        for row, field in enumerate(DOC_FIELDS[self.current_doc_type]):
            label = APP_FIELD_LABELS[field]
            label_widget = ctk.CTkLabel(
                self.form_scroll,
                text=f"{label}　⚠️ 要確認",
                anchor="w",
                font=ctk.CTkFont(size=14),
            )
            label_widget.grid(row=row * 2, column=0, padx=4, pady=(6, 1), sticky="ew")
            self.field_labels[field] = label_widget
            if field in {"reason", "notes", "academic_record"}:
                widget = ctk.CTkTextbox(
                    self.form_scroll, height=74, wrap="word", fg_color="#FEF08A"
                )
            else:
                widget = ctk.CTkEntry(
                    self.form_scroll,
                    height=34,
                    fg_color="#FEF08A",
                    text_color="#111827",
                )
            widget.grid(row=row * 2 + 1, column=0, padx=4, pady=(0, 3), sticky="ew")
            self.field_widgets[field] = widget
            widget.bind("<KeyRelease>", lambda _event, key=field: self._field_edited(key))
            widget.bind("<FocusOut>", lambda _event, key=field: self._field_edited(key))
        self._refresh_field_review()

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=0)
        self.status_label = ctk.CTkLabel(footer, text="準備ができました", anchor="w")
        self.status_label.grid(row=0, column=0, pady=(0, 4), sticky="ew")
        self.ocr_backend_menu = ctk.CTkOptionMenu(
            footer,
            values=["標準OCR（初回から使用可）", "Ollama（別途準備）"],
            command=self._change_ocr_backend,
            width=205,
        )
        self.ocr_backend_menu.set(
            "Ollama（別途準備）"
            if self.ocr_engine.backend == "ollama"
            else "標準OCR（初回から使用可）"
        )
        self.ocr_backend_menu.grid(row=0, column=1, padx=(8, 0), pady=(0, 4))
        self.privacy_badge = ctk.CTkLabel(
            footer,
            text=(
                "🔒 個人情報保護：元写真と作業コピーは差替え・登録・終了時に複数回上書き削除を試みます。"
                "記憶装置・クラウド同期・バックアップは完全消去を保証できません。スマホ本体の写真は残ります。"
            ),
            anchor="w",
            wraplength=1120,
            text_color=("#145C3C", "#9FE0BD"),
            font=ctk.CTkFont(size=12),
        )
        self.privacy_badge.grid(row=1, column=0, pady=(0, 3), sticky="ew")
        dnd_note = (
            "写真・PDF・名簿Excelを画面へドラッグできます。"
            if DND_AVAILABLE
            else "写真・PDF・名簿Excelのドラッグ操作は未設定です。上の選択ボタンをご利用ください。"
        )
        ctk.CTkLabel(
            footer,
            text=(
                dnd_note
                + " このPCとスマホは信頼できる同じWi-Fiにつないでください。"
                "配布用アプリでは写真読取に必要な部品の同梱確認が必要です。"
            ),
            anchor="w",
            wraplength=1120,
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=0, sticky="ew")

    def _enable_drag_and_drop(self) -> None:
        if not DND_AVAILABLE:
            return
        try:
            TkinterDnD._require(self)
            pending = [self]
            failures = 0
            while pending:
                widget = pending.pop()
                try:
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
                except (AttributeError, TclError):
                    failures += 1
                pending.extend(widget.winfo_children())
            if failures:
                self._set_status("一部の欄ではドロップできません。書類選択ボタンもご利用ください。")
        except Exception as error:
            self._set_status(f"ドラッグ操作を使えません。選択ボタンをご利用ください。({error})")

    def _on_drop(self, event: Any) -> None:
        try:
            paths = self.tk.splitlist(event.data)
            if not paths:
                return
            path = Path(paths[0])
            if path.suffix.lower() == ".xlsx":
                self._load_workbook(path)
            elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                self._stage_and_use_image(path)
            else:
                messagebox.showwarning(
                    "このファイルは使えません",
                    "写真、PDF、または .xlsx 名簿をドロップしてください。",
                    parent=self,
                )
        except Exception as error:
            messagebox.showerror("ファイルを開けません", str(error), parent=self)

    def _choose_document(self) -> None:
        selected = filedialog.askopenfilename(
            title="書類の写真またはPDFを選択", filetypes=IMAGE_TYPES
        )
        if selected:
            self._stage_and_use_image(Path(selected))

    def _stage_and_use_image(self, source: Path) -> None:
        if not self._deletion_notice_acknowledged:
            accepted = messagebox.askyesno(
                "元写真の自動削除について",
                "選んだ元写真は、新しい書類への差替え・名簿への保存・アプリ終了時に"
                "複数回の上書き後、削除を試みます。\n"
                "SSD、同期履歴、バックアップ等では完全消去を保証できません。\n"
                "この取り扱いを理解して続けますか？",
                parent=self,
            )
            if not accepted:
                return
            self._deletion_notice_acknowledged = True
        try:
            staged = stage_private_copy(source, self.cache_dir)
            self._accept_private_image(staged, original_path=source)
        except Exception as error:
            messagebox.showerror("書類を開けません", str(error), parent=self)

    def _accept_private_image(
        self, path: Path, *, original_path: Path | None = None
    ) -> None:
        self._stop_camera()
        try:
            if path.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                document = pdfium.PdfDocument(str(path))
                try:
                    if len(document) == 0:
                        raise ValueError("PDFにページがありません。")
                    page = document[0]
                    try:
                        width, height = page.get_size()
                        scale = min(1.5, 1800 / max(width, height))
                        bitmap = page.render(scale=max(0.5, scale))
                        try:
                            array = bitmap.to_numpy()
                            preview = Image.fromarray(np.asarray(array)).convert("RGB")
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
                finally:
                    document.close()
            else:
                with Image.open(path) as image:
                    if image.width * image.height > 50_000_000:
                        raise ValueError("画像の大きさが上限（5000万画素）を超えています。")
                    preview = ImageOps.exif_transpose(image).convert("RGB")
            self._clear_fields()
            deleted, detail = self._discard_current_image()
            self.current_image = path
            self.current_original = original_path
            self.preview_source = preview
            self.image_name.configure(text=path.name)
            self.camera_button.configure(state="normal")
            self._render_preview()
            status = "前の入力を消去しました。写真から文字を読み取っています。"
            if not deleted:
                status += f" 前の一時コピーを削除できませんでした: {detail}"
            self._set_status(status)
            self._start_reading()
        except Exception as error:
            if path.parent == self.cache_dir and path.exists():
                shred_file(path)
            messagebox.showerror("写真・PDFを表示できません", str(error), parent=self)

    def _discard_current_image(self) -> tuple[bool, str]:
        previous, self.current_image = self.current_image, None
        original, self.current_original = self.current_original, None
        failures: list[str] = []
        if previous and previous.exists():
            ok, message = shred_file(previous)
            if not ok:
                failures.append(message)
        if original and original.exists() and original != previous:
            ok, message = shred_file(original)
            if not ok:
                failures.append(message)
                self.pending_cleanup_paths.add(original)
            else:
                self.pending_cleanup_paths.discard(original)
        if previous:
            if previous.exists():
                self.pending_cleanup_paths.add(previous)
            else:
                self.pending_cleanup_paths.discard(previous)
        if failures:
            return False, "\n".join(failures)
        return True, "元写真と作業用コピーを上書き後に削除しました。"

    def _render_preview(self) -> None:
        if self.preview_source is None or self._closing:
            return
        try:
            if not self.preview_label.winfo_exists():
                return
            width = max(160, self.preview_label.winfo_width() - 24)
            height = max(160, self.preview_label.winfo_height() - 24)
        except TclError:
            return
        preview = self.preview_source.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        preview_image = ctk.CTkImage(
            light_image=preview, dark_image=preview, size=preview.size
        )
        self.preview_label.configure(image=preview_image, text="")
        self.preview_ctk_image = preview_image

    def _on_resize(self, _event: object) -> None:
        if self.preview_source is not None and not self._closing:
            self.after_idle(self._render_preview)

    def _change_document_type(self, value: str) -> None:
        if value not in DOCUMENT_TYPES or value == self.current_doc_type:
            return
        self.current_doc_type = value
        self._render_fields()
        self._set_status("書類の種類を変更しました。入力欄を確認してください。")
        if self.current_image:
            self._start_reading()

    def _start_reading(self) -> None:
        if self.current_image is None:
            return
        self.camera_button.configure(state="disabled")
        self.choose_image_button.configure(state="disabled")
        self.phone_button.configure(state="disabled")
        self.reading_label.configure(
            text=(
                "OllamaがPC内で画像を読み取っています。数十秒から数分かかる場合があります。"
                if self.ocr_engine.backend == "ollama"
                else "RapidOCRが文字を読み取っています…（書類枚数やPCにより変わります）"
            )
        )
        self.progress_bar.start()
        image = self.current_image
        doc_type = self.current_doc_type
        self._read_generation += 1
        generation = self._read_generation

        def worker() -> None:
            try:
                result = self.ocr_engine.recognize_document(image, doc_type)
            except Exception as error:
                result = {"_error": str(error)}
            self.event_queue.put(("reading", (generation, result)))

        threading.Thread(target=worker, daemon=True, name="document-reading").start()

    def _poll_events(self) -> None:
        if self._closing:
            return
        try:
            while True:
                event_name, payload = self.event_queue.get_nowait()
                if event_name == "phone":
                    self._accept_private_image(Path(payload))
                elif event_name == "reading":
                    generation, result = payload
                    if generation != self._read_generation:
                        continue
                    self.progress_bar.stop()
                    self.progress_bar.set(0)
                    self.choose_image_button.configure(state="normal")
                    self.phone_button.configure(state="normal")
                    self.camera_button.configure(state="normal")
                    self.reading_label.configure(text="読み取りが終わりました。黄色の項目をご確認ください。")
                    if not isinstance(result, dict):
                        self._set_status("文字を読み取れませんでした。")
                        continue
                    if result.get("_error"):
                        self._set_status("文字を読み取れませんでした。")
                        messagebox.showerror("読み取りエラー", result["_error"], parent=self)
                    else:
                        self._set_fields(result)
                        count = sum(
                            bool(result.get(field))
                            for field in DOC_FIELDS[self.current_doc_type]
                        )
                        self._set_status(
                            f"{count}項目を読み取りました。内容をご確認ください。"
                            if count
                            else "項目を読み取れませんでした。内容を手入力してください。"
                        )
                        self._play_completion_sound()
        except queue.Empty:
            pass
        self._event_after_id = self.after(120, self._poll_events)

    def _set_fields(self, values: dict[str, object]) -> None:
        confidences = values.get("field_confidence", {})
        self.field_confidence = (
            {key: float(value) for key, value in confidences.items()}
            if isinstance(confidences, dict)
            else {}
        )
        self.field_review_overrides.clear()
        for field, widget in self.field_widgets.items():
            value = sanitize_text(values.get(field, ""))
            if isinstance(widget, ctk.CTkTextbox):
                widget.delete("1.0", "end")
                widget.insert("1.0", value)
            else:
                widget.delete(0, "end")
                widget.insert(0, value)
        self._refresh_field_review()

    def _value_from_widget(self, widget: Any) -> str:
        if isinstance(widget, ctk.CTkTextbox):
            return sanitize_text(widget.get("1.0", "end-1c"))
        return sanitize_text(widget.get())

    def _field_edited(self, field: str) -> None:
        widget = self.field_widgets.get(field)
        if widget and self._value_from_widget(widget):
            self.field_review_overrides.add(field)
        self._refresh_field_review()

    def _refresh_field_review(self) -> None:
        if not hasattr(self, "field_widgets"):
            return
        for field, widget in self.field_widgets.items():
            value = self._value_from_widget(widget)
            confidence = self.field_confidence.get(field)
            needs_review = (
                not value
                or (
                    confidence is not None
                    and confidence < 0.8
                    and field not in self.field_review_overrides
                )
            )
            if needs_review:
                widget.configure(fg_color="#FEF08A", text_color="#111827")
                self.field_labels[field].configure(
                    text=f"{APP_FIELD_LABELS[field]}　⚠️ 要確認",
                    text_color="#854D0E",
                )
            else:
                widget.configure(
                    fg_color=("white", "#343638"),
                    text_color=("gray10", "gray90"),
                )
                self.field_labels[field].configure(
                    text=APP_FIELD_LABELS[field],
                    text_color=("gray10", "gray90"),
                )

    @staticmethod
    def _play_completion_sound() -> None:
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except (ImportError, RuntimeError):
            try:
                print("\a", end="", flush=True)
            except OSError:
                pass

    def _get_fields(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for field, widget in self.field_widgets.items():
            value = (
                widget.get("1.0", "end-1c")
                if isinstance(widget, ctk.CTkTextbox)
                else widget.get()
            )
            values[field] = sanitize_text(value)
        return values

    def _clear_fields(self) -> None:
        self._set_fields({})

    def _select_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="名簿の保存先を選ぶ",
            initialdir=str(self.output_path.parent),
            initialfile=self.output_path.name,
            defaultextension=".xlsx",
            filetypes=[("Excel名簿", "*.xlsx")],
        )
        if selected:
            self.output_path = Path(selected)
            self._update_output_label()
            self._set_status("名簿の保存先を変更しました。")

    def _import_workbook(self) -> None:
        selected = filedialog.askopenfilename(
            title="既存の名簿を開く",
            filetypes=[("Excel名簿", "*.xlsx")],
        )
        if selected:
            self._load_workbook(Path(selected))

    def _load_workbook(self, path: Path) -> None:
        try:
            records = self.excel_manager.load_students(path)
            self.output_path = path
            self.imported_students = records
            self._update_student_menu()
            self._update_output_label()
            self._set_status(
                f"既存の名簿を開きました（{len(records)}件）。この名簿へ追記します。"
            )
        except Exception as error:
            messagebox.showerror("名簿を開けません", str(error), parent=self)

    def _create_workbook(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="新しい名簿を作る",
            initialdir=str(self.output_path.parent),
            initialfile=self.output_path.name,
            defaultextension=".xlsx",
            filetypes=[("Excel名簿", "*.xlsx")],
        )
        if not selected:
            return
        try:
            self.output_path = self.excel_manager.create_template(selected)
            self.imported_students = []
            self._update_student_menu()
            self._update_output_label()
            self._set_status("新しい名簿を作りました。")
        except Exception as error:
            messagebox.showerror("名簿を作れません", str(error), parent=self)

    def _update_student_menu(self) -> None:
        labels = [student.display_label for student in self.imported_students]
        self.student_menu.configure(
            values=labels or ["名簿から選ぶ（データなし）"],
            state="normal" if labels else "disabled",
        )
        if labels:
            self.student_menu.set(labels[0])

    def _select_imported_student(self, label: str) -> None:
        student = next(
            (item for item in self.imported_students if item.display_label == label),
            None,
        )
        if student:
            imported = {
                "name": student.fields.get("氏名", ""),
                "furigana": student.fields.get("フリガナ", ""),
                "birth_date_year": student.fields.get("生年月日/年度", ""),
                "age": student.fields.get("年齢", ""),
                "school_name": student.fields.get("出身校", ""),
                "academic_record": student.fields.get("評定/資格/出欠", ""),
                "reason": student.fields.get("志願理由/推薦所見/特記事項", ""),
            }
            document_type = student.fields.get("書類種別", "")
            if document_type in DOCUMENT_TYPES and document_type != self.current_doc_type:
                self.current_doc_type = document_type
                self.doc_type_menu.set(document_type)
                self._render_fields()
            self._set_fields(imported)
            self._set_status(f"名簿の{student.row_number}行目を表示しました。")

    def _save_record(self) -> None:
        if self.current_image and not messagebox.askyesno(
            "OCR結果の確認",
            "OCRは誤字や読み違いをすることがあります。\n"
            "黄色の項目だけでなく、入力されたすべての項目を元の書類と見比べましたか？\n"
            "確認していない場合は「いいえ」を選び、修正してから保存してください。",
            parent=self,
        ):
            return
        fields = self._get_fields()
        if not any(fields.values()):
            messagebox.showwarning("入力がありません", "保存する内容を入力してください。", parent=self)
            return
        student_name = fields.get("name", "").strip()
        if not student_name:
            student_name = "お名前未入力"
        record = {"書類種別": self.current_doc_type, **fields}
        try:
            self.excel_manager.append_student(self.output_path, record)
        except Exception as error:
            self._set_status("名簿に保存できませんでした。写真は保持しています。")
            messagebox.showerror("保存エラー", str(error), parent=self)
            return
        deleted, detail = self._discard_current_image()
        self.preview_source = None
        self.preview_label.configure(image="", text="書類を選ぶか、ここにドラッグしてください")
        self.preview_ctk_image = None
        self.image_name.configure(text="元写真と作業用コピーの削除を試みました")
        self.camera_button.configure(state="normal")
        self._clear_fields()
        self.last_saved_student = student_name
        self.reassurance_label.configure(
            text=f"✅ 【{student_name}さん】のデータを名簿に記録しました。続けて次の書類を入れてください。"
        )
        if deleted:
            self._set_status("名簿に保存し、元写真と作業用コピーを上書き削除しました。")
            cleanup_note = (
                "元写真とPC上の作業コピーを上書きして削除しました。"
                "記憶装置の仕様により完全消去は保証できません。"
            )
        else:
            self._set_status("名簿には保存しましたが、元写真または一時コピーの削除を完了できませんでした。")
            cleanup_note = f"名簿には保存しましたが、写真の削除に失敗しました。\n{detail}"
        self._show_save_success(student_name, cleanup_note)

    def _show_save_success(self, student_name: str, cleanup_note: str) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("名簿に保存しました")
        dialog.geometry("490x300")
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            dialog,
            text=f"✅ {student_name}さんの書類を保存しました",
            font=ctk.CTkFont(size=19, weight="bold"),
            wraplength=440,
        ).grid(row=0, column=0, padx=20, pady=(22, 8), sticky="ew")
        ctk.CTkLabel(
            dialog,
            text=f"保存先：{self.output_path}\n{cleanup_note}",
            font=ctk.CTkFont(size=13),
            wraplength=440,
            justify="left",
        ).grid(row=1, column=0, padx=20, pady=8, sticky="ew")
        ctk.CTkButton(
            dialog,
            text="📂 今保存したExcelファイルを開く",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=GREEN,
            hover_color=GREEN_HOVER,
            command=self._open_saved_workbook,
        ).grid(row=2, column=0, padx=24, pady=(12, 8), sticky="ew")
        ctk.CTkButton(
            dialog, text="閉じる", command=dialog.destroy, fg_color=GRAY
        ).grid(row=3, column=0, padx=24, pady=(4, 18), sticky="ew")

    def _open_saved_workbook(self) -> None:
        try:
            if not self.output_path.is_file():
                raise FileNotFoundError(f"名簿ファイルが見つかりません: {self.output_path}")
            if hasattr(os, "startfile"):
                os.startfile(str(self.output_path))
            elif os.name == "posix":
                command = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([command, str(self.output_path)])
            else:
                raise OSError("このPCでExcelファイルを開く方法が見つかりません。")
        except (OSError, FileNotFoundError) as error:
            messagebox.showerror("Excelを開けません", str(error), parent=self)

    def _change_text_size(self, value: str) -> None:
        self._ui_scale = 1.2 if value == "大きめ" else 1.0
        ctk.set_widget_scaling(self._ui_scale)
        self._set_status(
            "文字を大きくしました。" if self._ui_scale > 1 else "文字サイズを標準に戻しました。"
        )

    def _open_camera(self) -> None:
        if self._camera_window and self._camera_window.winfo_exists():
            self._camera_window.lift()
            return
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "カメラを開けません",
                "カメラが見つからないか、ほかのアプリが使用中です。"
                "カメラの使用許可を確認するか、写真選択・スマホ送信をご利用ください。",
                parent=self,
            )
            return
        self._camera = camera
        window = ctk.CTkToplevel(self)
        self._camera_window = window
        window.title("パソコンのカメラで撮影")
        window.geometry("720x620")
        window.transient(self)
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(0, weight=1)
        preview = ctk.CTkLabel(
            window,
            text="カメラ映像を準備しています",
            fg_color=("gray90", "gray18"),
        )
        preview.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        ctk.CTkLabel(
            window,
            text="書類全体が画面に入り、明るく見える位置に置いてください。",
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        controls = ctk.CTkFrame(window, fg_color="transparent")
        controls.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        controls.grid_columnconfigure((0, 1), weight=1)

        def finish_camera() -> None:
            self._stop_camera()

        def show_frame() -> None:
            if not window.winfo_exists() or not self._camera:
                return
            ok, frame = self._camera.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                picture = Image.fromarray(rgb)
                picture.thumbnail((680, 460), Image.Resampling.LANCZOS)
                camera_image = ctk.CTkImage(
                    light_image=picture, dark_image=picture, size=picture.size
                )
                preview.configure(image=camera_image, text="")
                preview.image = camera_image
            else:
                preview.configure(text="カメラ映像を取得できません。")
            self._camera_after_id = window.after(40, show_frame)

        def take_picture() -> None:
            if not self._camera:
                return
            ok, frame = self._camera.read()
            if not ok:
                messagebox.showerror("撮影できません", "カメラから画像を取得できませんでした。", parent=window)
                return
            captured = self.cache_dir / f"camera-{uuid.uuid4().hex}.jpg"
            if not cv2.imwrite(str(captured), frame):
                messagebox.showerror("保存できません", "撮影した写真を一時保存できませんでした。", parent=window)
                return
            finish_camera()
            self._accept_private_image(captured)

        ctk.CTkButton(
            controls,
            text="この写真を使う",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=GREEN,
            hover_color=GREEN_HOVER,
            command=take_picture,
        ).grid(row=0, column=0, padx=5, sticky="ew")
        ctk.CTkButton(
            controls, text="閉じる", command=finish_camera, fg_color=GRAY
        ).grid(row=0, column=1, padx=5, sticky="ew")
        window.protocol("WM_DELETE_WINDOW", finish_camera)
        show_frame()

    def _stop_camera(self) -> None:
        window = self._camera_window
        if self._camera_after_id and window:
            try:
                window.after_cancel(self._camera_after_id)
            except TclError:
                pass
        self._camera_after_id = None
        if self._camera:
            self._camera.release()
            self._camera = None
        if window:
            try:
                if window.winfo_exists():
                    window.destroy()
            except TclError:
                pass
        self._camera_window = None

    def _show_phone_qr(self) -> None:
        try:
            url = self.receiver.start()
            qr = self.receiver.qr_image().convert("RGB")
        except Exception as error:
            messagebox.showerror("スマホ受信を開始できません", str(error), parent=self)
            return
        popup = ctk.CTkToplevel(self)
        popup.title("スマホで写真を送る")
        popup.geometry("420x520")
        popup.transient(self)
        popup.grab_set()
        ctk.CTkLabel(
            popup,
            text="スマホで読み取って写真を送信",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(padx=16, pady=(18, 8))
        qr_image = ctk.CTkImage(light_image=qr, dark_image=qr, size=(280, 280))
        qr_label = ctk.CTkLabel(popup, image=qr_image, text="")
        qr_label.image = qr_image
        qr_label.pack(padx=12, pady=8)
        ctk.CTkLabel(
            popup,
            text=(
                f"{url}\n信頼できる同じWi-Fiで使用してください。"
                "QRを知る人は送信できます。通信は暗号化されません。20MBまで。"
            ),
            wraplength=370,
            font=ctk.CTkFont(size=12),
        ).pack(padx=16, pady=8)

        def close_popup() -> None:
            self.receiver.stop()
            popup.destroy()

        ctk.CTkButton(
            popup, text="閉じる", command=close_popup, fg_color=GRAY
        ).pack(padx=18, pady=(4, 16))
        popup.protocol("WM_DELETE_WINDOW", close_popup)
        self._set_status("スマホからの写真を待っています。同じWi-Fiを使用してください。")

    def _update_output_label(self) -> None:
        self.output_label.configure(text=f"保存先: {self.output_path}")

    def _change_ocr_backend(self, selection: str) -> None:
        backend = "ollama" if selection == "Ollama（別途準備）" else "rapidocr"
        if backend == self.ocr_engine.backend:
            return
        if self.current_image and any(self._get_fields().values()):
            if not messagebox.askyesno(
                "読み取り方式を変更しますか",
                "現在の入力欄を新しい読み取り結果で置き換えます。続けますか？",
                parent=self,
            ):
                self.ocr_backend_menu.set(
                    "Ollama（別途準備）"
                    if self.ocr_engine.backend == "ollama"
                    else "標準OCR（初回から使用可）"
                )
                return

        self.ocr_engine = ApplicationOcrEngine(backend=backend)
        if hasattr(self, "reassurance_label"):
            self.reassurance_label.configure(
                text=(
                    "Ollamaを選択中です。Ollamaと画像モデルが必要です。"
                    if backend == "ollama"
                    else "標準OCRを選択中です。追加のOCRソフトは不要です。"
                )
            )
        self._set_status(
            "読み取り方式をOllamaに変更しました。"
            if backend == "ollama"
            else "読み取り方式を標準OCRに変更しました。"
        )
        if self.current_image:
            self._start_reading()

    def _set_status(self, text: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.configure(text=text)

    def _cleanup_cache(self) -> list[str]:
        errors = cleanup_cache_folder(self.cache_dir)
        if errors:
            for error in errors:
                print(f"一時コピーの削除に失敗しました: {error}")
        return errors

    def _retry_failed_source_cleanup(self) -> list[str]:
        errors: list[str] = []
        for path in tuple(self.pending_cleanup_paths):
            if not path.exists():
                self.pending_cleanup_paths.discard(path)
                continue
            ok, message = shred_file(path)
            if ok:
                self.pending_cleanup_paths.discard(path)
            else:
                errors.append(message)
        return errors

    def _emergency_cleanup(self) -> None:
        try:
            self.receiver.stop()
        except Exception as error:
            print(f"写真受信を停止できませんでした: {error}")
        deleted, detail = self._discard_current_image()
        if not deleted:
            print(f"写真の削除に失敗しました: {detail}")
        for error in self._retry_failed_source_cleanup():
            print(f"元写真の再削除に失敗しました: {error}")
        self._cleanup_cache()

    def _close_app(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._read_generation += 1
        if self._event_after_id:
            try:
                self.after_cancel(self._event_after_id)
            except TclError:
                pass
            self._event_after_id = None
        errors: list[str] = []
        self._stop_camera()
        try:
            self.receiver.stop()
        except Exception as error:
            errors.append(f"写真受信を停止できませんでした: {error}")
        deleted, detail = self._discard_current_image()
        if not deleted:
            errors.append(detail)
        errors.extend(self._retry_failed_source_cleanup())
        errors.extend(self._cleanup_cache())
        self.preview_source = None
        self.preview_label.configure(image="")
        self.preview_ctk_image = None
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        if errors:
            messagebox.showwarning(
                "終了時の処理に注意が必要です",
                "\n".join(errors[:5]),
                parent=self,
            )
        self.destroy()


if __name__ == "__main__":
    AdmissionsOCRApp().mainloop()

"""Import, format, and append records for multiple admissions document types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

from security import safe_excel_value, sanitize_text

HEADERS = [
    "受付番号",
    "書類種別",
    "登録日時",
    "氏名",
    "フリガナ",
    "生年月日/年度",
    "出身校",
    "評定/資格/出欠",
    "志願理由/推薦所見/特記事項",
]
STUDENT_FIELDS = HEADERS[3:]
HEADER_ALIASES = {
    "受付番号": ("受付番号", "受付No", "番号"),
    "書類種別": ("書類種別", "資料種別", "文書種別"),
    "登録日時": ("登録日時", "登録日", "受付日時"),
    "氏名": ("氏名", "受験者氏名", "志願者氏名", "名前"),
    "フリガナ": ("フリガナ", "ふりがな", "カナ"),
    "生年月日/年度": ("生年月日/年度", "生年月日", "生年月日・年度", "年度"),
    "年齢": ("年齢", "年令"),
    "出身校": ("出身校", "中学校名", "高校名", "学校名", "中学校", "高校", "在籍校", "卒業校"),
    "評定/資格/出欠": (
        "評定/資格/出欠", "評定", "成績", "資格", "出欠", "活動内容", "諸活動"
    ),
    "志願理由/推薦所見/特記事項": (
        "志願理由/推薦所見/特記事項",
        "志望理由/推薦所見/特記事項",
        "入学理由・志望動機",
        "志望理由",
        "入学理由",
        "志望動機",
        "推薦所見",
        "特記事項",
        "備考",
    ),
}
FIELD_KEYS = {
    "受付番号": ("受付番号", "reception_number"),
    "書類種別": ("書類種別", "doc_type"),
    "登録日時": ("登録日時", "registered_at"),
    "氏名": ("氏名", "name"),
    "フリガナ": ("フリガナ", "furigana"),
    "生年月日/年度": (
        "生年月日/年度",
        "birth_date_year",
        "生年月日",
        "birth_date",
        "academic_year",
    ),
    "出身校": ("出身校", "school_name"),
    "評定/資格/出欠": (
        "評定/資格/出欠",
        "academic_record",
        "grade_record",
        "qualifications",
        "activities",
        "attendance",
    ),
    "志願理由/推薦所見/特記事項": (
        "志願理由/推薦所見/特記事項",
        "reason",
        "recommendation",
        "notes",
        "special_notes",
    ),
}
HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F6B4E")
HEADER_FONT = Font(color="FFFFFF", bold=True)
HEADER_ALIGNMENT = Alignment(horizontal="center", vertical="center", wrap_text=True)
CELL_BORDER = Border(
    left=Side(style="thin", color="B7C9D6"),
    right=Side(style="thin", color="B7C9D6"),
    top=Side(style="thin", color="B7C9D6"),
    bottom=Side(style="thin", color="B7C9D6"),
)


@dataclass(frozen=True)
class ImportedStudent:
    row_number: int
    fields: dict[str, str]

    @property
    def display_label(self) -> str:
        name = self.fields.get("氏名", "").strip() or "（氏名なし）"
        reception = self.fields.get("受付番号", "").strip()
        return f"{reception + ': ' if reception else ''}{name}（{self.row_number}行目）"


def _normalize_header(value: object) -> str:
    return "".join(sanitize_text(value).lower().split()).replace("　", "")


def _header_key(value: object) -> str | None:
    normalized = _normalize_header(value)
    for key, aliases in HEADER_ALIASES.items():
        if normalized in {_normalize_header(alias) for alias in aliases}:
            return key
    return None


def _find_header_row(worksheet: Worksheet) -> tuple[int, dict[str, int]]:
    for row_number in range(1, min(worksheet.max_row, 12) + 1):
        row_values = next(
            worksheet.iter_rows(
                min_row=row_number, max_row=row_number, values_only=True
            )
        )
        mapping: dict[str, int] = {}
        for column_number, value in enumerate(row_values, start=1):
            key = _header_key(value)
            if key and key not in mapping:
                mapping[key] = column_number
        if any(key in mapping for key in ("氏名", "フリガナ", "出身校", "生年月日/年度")):
            return row_number, mapping
    if worksheet.max_row == 1 and worksheet.max_column == 1:
        if worksheet.cell(row=1, column=1).value is None:
            return 1, {}
    raise ValueError(
        "先頭12行に名簿の見出しが見つかりません。対応する .xlsx を選んでください。"
    )


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return sanitize_text(value)


def _student_value(
    student_data: Mapping[str, object], header: str, *, include_age: bool = True
) -> str:
    value: object = ""
    for key in FIELD_KEYS[header]:
        if key in student_data and student_data[key] not in (None, ""):
            value = student_data[key]
            break
    text = sanitize_text(value)
    if header == "生年月日/年度" and include_age:
        age = sanitize_text(student_data.get("age", student_data.get("年齢", "")))
        if age and age not in text:
            text = f"{text}（{age}）" if text else age
    return text


def _style_header(worksheet: Worksheet, row: int, column: int) -> None:
    cell = worksheet.cell(row=row, column=column)
    cell.fill = HEADER_FILL
    cell.font = HEADER_FONT
    cell.alignment = HEADER_ALIGNMENT
    cell.border = CELL_BORDER


class ExcelManager:
    """Keep all admissions-related records in one categorized workbook."""

    def create_template(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        if path.suffix.lower() != ".xlsx":
            raise ValueError("名簿ファイルには .xlsx 拡張子を指定してください。")
        if path.exists():
            raise FileExistsError(f"既存ファイルを上書きしません: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "入試関係書類"
        worksheet.append(HEADERS)
        for column, header in enumerate(HEADERS, start=1):
            _style_header(worksheet, 1, column)
            worksheet.column_dimensions[
                worksheet.cell(row=1, column=column).column_letter
            ].width = min(max(len(header) * 2 + 4, 14), 38)
        worksheet.freeze_panes = "A2"
        try:
            workbook.save(path)
        finally:
            workbook.close()
        return path

    def load_students(self, file_path: str | Path) -> list[ImportedStudent]:
        path = Path(file_path)
        if path.suffix.lower() != ".xlsx":
            raise ValueError("Excel取込は .xlsx ファイルに対応しています。")
        if not path.is_file():
            raise FileNotFoundError(f"名簿ファイルが見つかりません: {path}")
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            worksheet = workbook.active
            header_row, mapping = _find_header_row(worksheet)
            records: list[ImportedStudent] = []
            for row_number, row in enumerate(
                worksheet.iter_rows(min_row=header_row + 1, values_only=True),
                start=header_row + 1,
            ):
                fields = {
                    header: _cell_text(row[column - 1])
                    if column <= len(row)
                    else ""
                    for header, column in mapping.items()
                }
                if any(fields.get(field, "") for field in STUDENT_FIELDS):
                    records.append(ImportedStudent(row_number, fields))
            return records
        finally:
            workbook.close()

    def append_student(self, file_path: str | Path, student_data: dict) -> bool:
        path = Path(file_path)
        if path.suffix.lower() != ".xlsx":
            raise ValueError("名簿ファイルには .xlsx 拡張子を指定してください。")
        if not path.exists():
            self.create_template(path)
        if not path.is_file():
            raise FileNotFoundError(f"名簿ファイルを開けません: {path}")

        workbook = load_workbook(path)
        try:
            worksheet = workbook.active
            header_row, mapping = _find_header_row(worksheet)
            if not mapping:
                mapping = {}
                for column, header in enumerate(HEADERS, start=1):
                    worksheet.cell(row=header_row, column=column, value=header)
                    _style_header(worksheet, header_row, column)
                    mapping[header] = column
            for header in HEADERS:
                if header not in mapping:
                    column = worksheet.max_column + 1
                    worksheet.cell(row=header_row, column=column, value=header)
                    _style_header(worksheet, header_row, column)
                    worksheet.column_dimensions[
                        worksheet.cell(row=header_row, column=column).column_letter
                    ].width = min(max(len(header) * 2 + 4, 14), 38)
                    mapping[header] = column

            reception_column = mapping["受付番号"]
            last_number = 0
            for row in range(header_row + 1, worksheet.max_row + 1):
                try:
                    last_number = max(
                        last_number,
                        int(worksheet.cell(row=row, column=reception_column).value),
                    )
                except (TypeError, ValueError):
                    continue
            values = {
                "受付番号": last_number + 1,
                "書類種別": sanitize_text(
                    student_data.get("書類種別", student_data.get("doc_type", "その他入試関係書類"))
                ),
                "登録日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            values.update(
                {
                    header: _student_value(
                        student_data,
                        header,
                        include_age="年齢" not in mapping,
                    )
                    for header in STUDENT_FIELDS
                }
            )
            target_row = max(worksheet.max_row + 1, header_row + 1)
            for header in HEADERS:
                value: object = values[header]
                if header not in {"受付番号", "登録日時"}:
                    value = safe_excel_value(value)
                worksheet.cell(
                    row=target_row, column=mapping[header], value=value
                ).border = CELL_BORDER
            if "年齢" in mapping:
                age = sanitize_text(student_data.get("age", student_data.get("年齢", "")))
                worksheet.cell(
                    row=target_row,
                    column=mapping["年齢"],
                    value=safe_excel_value(age),
                ).border = CELL_BORDER
            workbook.save(path)
            return True
        finally:
            workbook.close()

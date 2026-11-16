"""Local OCR, camera-photo cleanup, and admissions-document field extraction."""

from __future__ import annotations

import base64
import json
import os
import re
import statistics
import threading
import unicodedata
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

from security import sanitize_text

MAX_IMAGE_PIXELS = 50_000_000
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 240
OLLAMA_MAX_IMAGE_SIDE = 2048
OLLAMA_MAX_RESPONSE_CHARS = 16_384
DOCUMENT_TYPES = (
    "入学願書",
    "調査書・成績証明書",
    "推薦書・志望理由書",
    "諸活動の記録・報告書",
    "その他入試関係書類",
)


def configured_ocr_backend() -> str:
    """Return the selected OCR backend, keeping the no-extra-install default."""
    return os.environ.get("SCHOOL_APPLICATION_OCR_BACKEND", "rapidocr").strip().lower()


FIELD_LABELS = {
    "name": "氏名",
    "furigana": "フリガナ",
    "birth_date_year": "生年月日/年度",
    "age": "年齢",
    "school_name": "出身校",
    "academic_record": "評定/資格/出欠",
    "reason": "志願理由/推薦所見/特記事項",
    "notes": "特記事項",
}

BASE_ALIASES = {
    "name": ("氏名", "受験者氏名", "志願者氏名", "生徒氏名", "名前"),
    "furigana": ("フリガナ", "ふりがな", "カナ"),
    "birth_date_year": (
        "生年月日/年度",
        "生年月日",
        "生年月日・年度",
        "入学年度",
        "卒業年度",
        "年度",
    ),
    "age": ("年齢", "年令"),
    "school_name": (
        "出身校",
        "出身中学校",
        "出身高校",
        "中学校名",
        "高校名",
        "在籍校",
        "卒業校",
        "志望校",
        "学校名",
    ),
    "academic_record": (
        "評定/資格/出欠",
        "評定",
        "成績",
        "資格",
        "検定",
        "出欠",
        "欠席日数",
        "活動内容",
        "諸活動",
        "活動記録",
        "受賞歴",
        "部活動",
        "委員会",
    ),
    "reason": (
        "志願理由",
        "志望理由",
        "志望動機",
        "志望校理由",
        "入学理由",
        "推薦所見",
        "推薦理由",
        "志願者所見",
        "入学後の抱負",
        "志望",
    ),
    "notes": ("特記事項", "備考", "所見"),
}

DOC_FIELDS = {
    "入学願書": ("name", "furigana", "birth_date_year", "age", "school_name", "reason"),
    "調査書・成績証明書": (
        "name",
        "birth_date_year",
        "school_name",
        "academic_record",
        "notes",
    ),
    "推薦書・志望理由書": (
        "name",
        "furigana",
        "school_name",
        "reason",
        "notes",
    ),
    "諸活動の記録・報告書": (
        "name",
        "school_name",
        "academic_record",
        "notes",
    ),
    "その他入試関係書類": (
        "name",
        "furigana",
        "birth_date_year",
        "school_name",
        "academic_record",
        "reason",
        "notes",
    ),
}

ALIASES_BY_TYPE = {
    "入学願書": ("name", "furigana", "birth_date_year", "age", "school_name", "reason"),
    "調査書・成績証明書": (
        "name", "birth_date_year", "school_name", "academic_record", "notes"
    ),
    "推薦書・志望理由書": ("name", "furigana", "school_name", "reason", "notes"),
    "諸活動の記録・報告書": ("name", "school_name", "academic_record", "notes"),
    "その他入試関係書類": tuple(BASE_ALIASES),
}


@dataclass(frozen=True)
class OCRLine:
    text: str
    box: tuple[tuple[float, float], ...]
    confidence: float

    @property
    def center_y(self) -> float:
        return sum(point[1] for point in self.box) / len(self.box)

    @property
    def left(self) -> float:
        return min(point[0] for point in self.box)

    @property
    def height(self) -> float:
        return max(point[1] for point in self.box) - min(point[1] for point in self.box)


def _read_image(image_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_path, np.ndarray):
        image = image_path.copy()
    else:
        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source)
            if oriented.width * oriented.height > MAX_IMAGE_PIXELS:
                raise ValueError("画像の大きさが上限（5000万画素）を超えています。")
            rgb = np.asarray(oriented.convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("対応していない画像形式です。")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _rectify_document(image: np.ndarray) -> np.ndarray | None:
    """Return a perspective-corrected page when a complete page edge is visible."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    sample = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if sample.ndim == 3 else sample
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sample_area = sample.shape[0] * sample.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < sample_area * 0.18 or area > sample_area * 0.97:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidates.append((area, approx.reshape(4, 2).astype(np.float32) / scale))
    if not candidates:
        return None

    corners = _ordered_quad(max(candidates, key=lambda item: item[0])[1])
    top_left, top_right, bottom_right, bottom_left = corners
    target_width = int(
        max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    )
    target_height = int(
        max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    )
    if target_width < 200 or target_height < 200:
        return None
    aspect = target_width / target_height
    if not 0.45 <= aspect <= 1.8:
        return None
    target_width = min(target_width, 3600)
    target_height = min(target_height, 3600)
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _enhanced_variants(image: np.ndarray) -> list[np.ndarray]:
    height, width = image.shape[:2]
    if not height or not width:
        raise ValueError("画像の大きさが正しくありません。")
    largest = max(height, width)
    if largest > 3600:
        scale = 3600 / largest
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    elif largest < 2200:
        scale = min(2200 / largest, 2.0)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(
        enhanced, None, h=6, templateWindowSize=7, searchWindowSize=21
    )
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.45, blurred, -0.45, 0)
    block_size = max(31, min(61, (min(sharpened.shape[:2]) // 30) | 1))
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        9,
    )
    return [sharpened, binary]


def preprocess_image_variants(
    image_path: str | Path | np.ndarray,
) -> list[np.ndarray]:
    """Prepare grayscale and binarized views, plus a rectified page when possible."""
    image = _read_image(image_path)
    rectified = _rectify_document(image)
    sources = [image]
    if rectified is not None:
        sources.append(rectified)
    variants: list[np.ndarray] = []
    for source in sources:
        variants.extend(_enhanced_variants(source))
    return variants


def preprocess_image(image_path: str | Path | np.ndarray) -> np.ndarray:
    """Backward-compatible single image preprocessor for existing callers."""
    return preprocess_image_variants(image_path)[0]


def _box_points(raw_box: Any) -> tuple[tuple[float, float], ...]:
    try:
        return tuple((float(point[0]), float(point[1])) for point in raw_box)
    except (TypeError, ValueError, IndexError):
        return ()


def _group_rows(lines: Iterable[OCRLine]) -> list[list[OCRLine]]:
    ordered = sorted(lines, key=lambda line: (line.center_y, line.left))
    heights = [line.height for line in ordered if line.height > 0]
    tolerance = max(5.0, (statistics.median(heights) if heights else 12.0) * 0.65)
    rows: list[list[OCRLine]] = []
    centers: list[float] = []
    for line in ordered:
        nearest = min(
            range(len(centers)),
            key=lambda index: abs(centers[index] - line.center_y),
            default=-1,
        )
        if nearest >= 0 and abs(centers[nearest] - line.center_y) <= tolerance:
            rows[nearest].append(line)
            centers[nearest] = sum(item.center_y for item in rows[nearest]) / len(
                rows[nearest]
            )
        else:
            rows.append([line])
            centers.append(line.center_y)
    return [sorted(row, key=lambda line: line.left) for row in rows]


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _label_matches(text: str, doc_type: str) -> list[tuple[int, int, str]]:
    field_keys = ALIASES_BY_TYPE.get(doc_type, ALIASES_BY_TYPE["その他入試関係書類"])
    aliases = sorted(
        (
            (alias, field)
            for field in field_keys
            for alias in BASE_ALIASES[field]
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    found: list[tuple[int, int, str]] = []
    compact_text = _compact(text)
    for alias, field in aliases:
        offset = 0
        alias = _compact(alias)
        while (position := compact_text.find(alias, offset)) >= 0:
            end = position + len(alias)
            if not any(position < right and end > left for left, right, _ in found):
                found.append((position, end, field))
            offset = end
    return sorted(found)


def _clean_value(value: str) -> str:
    return sanitize_text(re.sub(r"^[\s:：|｜・、,，]+", "", value).strip(" \t:：|｜"))


def _age_from_birth_date(value: str) -> str:
    text = value.strip()
    date_match = re.search(
        r"(?P<year>(?:19|20)\d{2})\s*[年./-]\s*(?P<month>\d{1,2})"
        r"\s*[月./-]\s*(?P<day>\d{1,2})",
        text,
    )
    if date_match:
        year, month, day = (
            int(date_match.group("year")),
            int(date_match.group("month")),
            int(date_match.group("day")),
        )
    else:
        era_match = re.search(
            r"(令和|平成|昭和)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
            text,
        )
        if not era_match:
            return ""
        era, era_year, month, day = era_match.groups()
        year = {"令和": 2018, "平成": 1988, "昭和": 1925}[era] + int(era_year)
        month, day = int(month), int(day)
    try:
        birth = date(year, month, day)
    except ValueError:
        return ""
    today = date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    return f"{age}歳" if 0 <= age <= 130 else ""


def _furigana_from_layout(
    rows: list[list[OCRLine]], row_texts: list[str], doc_type: str
) -> str:
    if "furigana" not in DOC_FIELDS.get(doc_type, ()):
        return ""
    name_rows = [
        index
        for index, text in enumerate(row_texts)
        if any(field == "name" for _, _, field in _label_matches(text, doc_type))
    ]
    birth_rows = [
        index
        for index, text in enumerate(row_texts)
        if any(
            field == "birth_date_year"
            for _, _, field in _label_matches(text, doc_type)
        )
    ]
    if not name_rows or not birth_rows or name_rows[0] >= birth_rows[0]:
        return ""

    name_label_left = min(line.left for line in rows[name_rows[0]])
    candidates: list[tuple[float, int, float, str]] = []
    for row in rows[name_rows[0] + 1 : birth_rows[0]]:
        for line in row:
            if line.left <= name_label_left:
                continue
            value = _clean_value(line.text)
            compact = _compact(value)
            if len(compact) < 2:
                continue
            kana_count = sum(
                "\u3040" <= character <= "\u30ff" for character in compact
            )
            kana_ratio = kana_count / len(compact)
            if kana_ratio >= 0.7:
                candidates.append(
                    (kana_ratio, len(compact), line.confidence, value)
                )
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[:3])[3]


def extract_document_fields(
    lines: Iterable[OCRLine], doc_type: str
) -> tuple[dict[str, str], str]:
    """Extract type-relevant values using text and reading positions."""
    if doc_type not in DOC_FIELDS:
        doc_type = "その他入試関係書類"
    rows = _group_rows(lines)
    row_texts = [_compact("".join(line.text for line in row)) for row in rows]
    result = {field: "" for field in DOC_FIELDS[doc_type]}

    for row_index, row_text in enumerate(row_texts):
        labels = _label_matches(row_text, doc_type)
        for label_index, (_start, end, field) in enumerate(labels):
            if field not in result or result[field]:
                continue
            stop = labels[label_index + 1][0] if label_index + 1 < len(labels) else len(row_text)
            value = _clean_value(row_text[end:stop])
            if not value:
                following: list[str] = []
                for next_row in row_texts[row_index + 1 :]:
                    if _label_matches(next_row, doc_type):
                        break
                    if next_row:
                        following.append(next_row)
                        if field not in {"reason", "academic_record", "notes"}:
                            break
                        if len(following) >= 5:
                            break
                value = _clean_value(" ".join(following))
            result[field] = value

    if "furigana" in result and not result["furigana"]:
        result["furigana"] = _furigana_from_layout(rows, row_texts, doc_type)
    if "age" in result and not result["age"]:
        result["age"] = _age_from_birth_date(result.get("birth_date_year", ""))
    if "academic_record" in result and not result["academic_record"]:
        evidence_terms = (
            "国語", "数学", "英語", "理科", "社会", "評定", "欠席", "出席",
            "遅刻", "資格", "検定", "活動", "受賞", "大会", "委員",
        )
        evidence = [
            row
            for row in row_texts
            if any(term in row for term in evidence_terms)
            and not _label_matches(row, doc_type)
        ]
        result["academic_record"] = _clean_value(" / ".join(evidence[:12]))
    return result, "\n".join(row_texts)


def extract_field_confidences(
    lines: Iterable[OCRLine], doc_type: str, fields: dict[str, str]
) -> dict[str, float]:
    """Estimate per-field confidence from OCR boxes; missing values are 0."""
    line_list = list(lines)
    if doc_type not in DOC_FIELDS:
        doc_type = "その他入試関係書類"
    confidences: dict[str, float] = {}
    for field in DOC_FIELDS[doc_type]:
        value = _compact(fields.get(field, ""))
        if not value:
            confidences[field] = 0.0
            continue
        aliases = BASE_ALIASES.get(field, ())
        label_scores = [
            line.confidence
            for line in line_list
            if any(alias in _compact(line.text) for alias in aliases)
        ]
        value_scores = [
            line.confidence
            for line in line_list
            if _compact(line.text) and _compact(line.text) in value
        ]
        if value_scores and label_scores:
            score = min(max(value_scores), max(label_scores))
        elif value_scores:
            score = max(value_scores)
        elif label_scores:
            score = max(label_scores)
        else:
            score = 0.0
        confidences[field] = max(0.0, min(1.0, float(score)))

    if "age" in confidences and not fields.get("age"):
        confidences["age"] = confidences.get("birth_date_year", 0.0)
    return confidences


def _lines_from_output(output: Any) -> list[OCRLine]:
    detections = output[0] if isinstance(output, tuple) and output else output
    lines: list[OCRLine] = []
    for item in detections or []:
        try:
            box, text, confidence = item
            points = _box_points(box)
            clean_text = sanitize_text(text)
            if points and clean_text:
                lines.append(OCRLine(clean_text, points, float(confidence)))
        except (TypeError, ValueError, IndexError):
            continue
    return lines


def _recognition_rank(
    lines: list[OCRLine], doc_type: str
) -> tuple[int, float, int, float]:
    fields, _ = extract_document_fields(lines, doc_type)
    confidences = extract_field_confidences(lines, doc_type, fields)
    reviewable_fields = sum(
        bool(value) and confidences.get(field, 0.0) >= 0.8
        for field, value in fields.items()
    )
    confidence_sum = sum(
        confidences.get(field, 0.0)
        for field, value in fields.items()
        if value
    )
    combined_text = _compact("".join(line.text for line in lines))
    label_count = len(_label_matches(combined_text, doc_type))
    confidence = (
        sum(max(0.0, min(1.0, line.confidence)) for line in lines) / len(lines)
        if lines
        else 0.0
    )
    return reviewable_fields, confidence_sum, label_count, confidence


def _iter_pdf_pages(path: Path) -> Iterable[np.ndarray]:
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        raise RuntimeError(
            "PDFを開くには pypdfium2 が必要です。requirements.txt をインストールしてください。"
        ) from error
    document = pdfium.PdfDocument(str(path))
    try:
        if len(document) > 100:
            raise ValueError("PDFのページ数が上限（100ページ）を超えています。")
        for page_index in range(len(document)):
            page = document[page_index]
            try:
                page_width, page_height = page.get_size()
                scale = min(1.8, 3200 / max(page_width, page_height))
                bitmap = page.render(scale=max(0.5, scale))
                try:
                    yield np.asarray(bitmap.to_numpy()).copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        document.close()


def _pdf_pages(path: Path) -> list[np.ndarray]:
    return list(_iter_pdf_pages(path))


class ApplicationOcrEngine:
    """Perform local OCR and document-type-aware Japanese field extraction."""

    def __init__(
        self,
        backend: str = "rapidocr",
        ollama_model: str | None = None,
    ) -> None:
        if backend not in {"rapidocr", "ollama"}:
            raise ValueError(f"未対応のOCRエンジンです: {backend}")
        self.backend = backend
        self._configured_ollama_model = (
            ollama_model or os.environ.get("SCHOOL_APPLICATION_OCR_OLLAMA_MODEL", "")
        ).strip()
        self._resolved_ollama_model: str | None = None
        self._engine: Any = None
        self._initialization_error = ""
        self._engine_lock = threading.Lock()

    def _ollama_model(self) -> str:
        if self._resolved_ollama_model:
            return self._resolved_ollama_model
        request = Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Ollamaに接続できません。Ollamaを起動し、PC内のAPI "
                "（127.0.0.1:11434）が利用できることを確認してください。"
            ) from error

        models = data.get("models", []) if isinstance(data, dict) else []
        if self._configured_ollama_model:
            installed = next(
                (
                    model
                    for model in models
                    if model.get("name") == self._configured_ollama_model
                    or model.get("model") == self._configured_ollama_model
                ),
                None,
            )
            if installed is None:
                raise RuntimeError(
                    "指定されたOllamaモデルが見つかりません。"
                    "SCHOOL_APPLICATION_OCR_OLLAMA_MODELの設定と"
                    "ollama listの結果を確認してください。"
                )
            capabilities = installed.get("capabilities")
            if isinstance(capabilities, list) and "vision" not in capabilities:
                raise RuntimeError("指定されたOllamaモデルは画像入力に対応していません。")
            self._resolved_ollama_model = self._configured_ollama_model
            return self._resolved_ollama_model

        vision_models = [
            model
            for model in models
            if isinstance(model.get("capabilities"), list)
            and "vision" in model["capabilities"]
            and isinstance(model.get("name"), str)
        ]
        if not vision_models:
            raise RuntimeError(
                "Ollamaに画像対応モデルがありません。画像対応モデルを用意するか、"
                "SCHOOL_APPLICATION_OCR_BACKEND=rapidocrで従来方式を選んでください。"
            )
        preferred = ("gemma4:e2b", "gemma4:e4b", "gemma4:12b", "gemma4")
        for preferred_name in preferred:
            selected = next(
                (model for model in vision_models if model["name"] == preferred_name),
                None,
            )
            if selected:
                self._resolved_ollama_model = selected["name"]
                return self._resolved_ollama_model
        self._resolved_ollama_model = vision_models[0]["name"]
        return self._resolved_ollama_model

    @staticmethod
    def _ollama_image(image: str | Path | np.ndarray) -> str:
        bgr = _read_image(image)
        if bgr.ndim == 2:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail(
            (OLLAMA_MAX_IMAGE_SIDE, OLLAMA_MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=92, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _parse_ollama_content(content: Any) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("応答本文が文字列ではありません")
        content = content.strip()
        if not content:
            raise ValueError("応答本文が空です")
        if len(content) > OLLAMA_MAX_RESPONSE_CHARS:
            raise ValueError("応答本文が上限を超えています")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as original_error:
            fenced_blocks = re.findall(
                r"```json\s*(.*?)\s*```",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if len(fenced_blocks) != 1:
                raise ValueError(
                    f"JSONとして解析できません（行{original_error.lineno}、"
                    f"列{original_error.colno}）"
                ) from original_error
            try:
                parsed = json.loads(fenced_blocks[0])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONとして解析できません（行{error.lineno}、列{error.colno}）"
                ) from error
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("JSONの最上位が項目オブジェクトではありません")

    def _recognize_ollama_page(
        self, image: str | Path | np.ndarray, doc_type: str
    ) -> dict[str, str]:
        model = self._ollama_model()
        fields = DOC_FIELDS[doc_type]
        aliases = {
            "name": "氏名（漢字）",
            "furigana": "フリガナ",
            "birth_date_year": "生年月日または年度",
            "age": "年齢",
            "school_name": "出身校・在籍校",
            "academic_record": "評定・資格・出欠・活動内容",
            "reason": "志願理由・志望理由・推薦所見",
            "notes": "特記事項・備考",
        }
        properties = {field: {"type": "string"} for field in fields}
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(fields),
            "additionalProperties": False,
        }
        field_instructions = "\n".join(
            f"- {field}: {aliases[field]}" for field in fields
        )
        prompt = (
            "あなたは日本語の入試書類を文字起こしする補助者です。"
            "画像に実際に見えている文字だけを、表記を直さず正確に転記してください。"
            "推測、補完、一般常識による修正をしてはいけません。"
            "読めない・確信がない・画像にない項目は空文字にしてください。"
            "氏名や学校名を創作してはいけません。"
            "JSON値は説明文を加えず、用紙に書かれた値だけにしてください。\n"
            f"書類種別: {doc_type}\n対象項目:\n{field_instructions}"
        )
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "keep_alive": "5m",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [self._ollama_image(image)],
                }
            ],
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 1024,
            },
        }
        request = Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read(2048).decode("utf-8", "replace")
            if "unknown model architecture" in detail:
                raise RuntimeError(
                    "Ollamaがこの画像モデルを実行できません。Ollamaを最新版へ更新し、"
                    "画像対応モデルを選び直してください。"
                ) from error
            raise RuntimeError(
                f"Ollamaの画像読取に失敗しました（HTTP {error.code}）。"
                "Ollamaのバージョンと画像対応モデルを確認してください。"
            ) from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Ollamaの画像読取が時間内に完了しませんでした。"
                "PCの空きメモリ、Ollamaモデル、画像サイズを確認してください。"
            ) from error

        message = result.get("message")
        message_keys = (
            ",".join(
                key
                for key in ("role", "content", "thinking", "images", "tool_calls")
                if key in message
            )
            if isinstance(message, dict)
            else "なし"
        )
        finish_reason = result.get("done_reason", "")
        if finish_reason not in {"stop", "length", "tool_calls"}:
            finish_reason = "その他" if finish_reason else "不明"
        content = message.get("content") if isinstance(message, dict) else None
        try:
            extracted = self._parse_ollama_content(content)
        except (ValueError, TypeError) as error:
            if finish_reason == "length":
                guidance = (
                    "Ollamaの回答が途中で切れた可能性があります。1ページずつ読み取るか、"
                    "画像を見やすく撮り直してください。"
                )
            elif str(error) == "応答本文が空です":
                guidance = (
                    "Ollamaから空の回答が返りました。Ollamaと画像モデルが起動しているか"
                    "確認してから再試行してください。"
                )
            else:
                guidance = (
                    "Ollamaの回答形式を読み取れませんでした。書類を変えずに再試行し、"
                    "続く場合は表示された診断情報をお知らせください。"
                )
            raise RuntimeError(
                f"{guidance}（{error}、終了理由={finish_reason}、"
                f"応答キー={message_keys}）"
            ) from error
        return {
            field: sanitize_text(extracted[field])
            if isinstance(extracted.get(field), str)
            else ""
            for field in fields
        }

    def _recognize_ollama_document(
        self, path: Path, doc_type: str
    ) -> dict[str, Any]:
        pages: Iterable[Path | np.ndarray]
        pages = _iter_pdf_pages(path) if path.suffix.lower() == ".pdf" else [path]
        merged = {field: "" for field in DOC_FIELDS[doc_type]}
        page_count = 0
        for page in pages:
            page_count += 1
            if page_count > 100:
                raise ValueError("書類のページ数が上限（100ページ）を超えています。")
            page_fields = self._recognize_ollama_page(page, doc_type)
            for field, value in page_fields.items():
                if value and not merged[field]:
                    merged[field] = value
        if not page_count:
            raise ValueError("書類に読み取れるページがありません。")
        if "age" in merged and not merged["age"]:
            merged["age"] = _age_from_birth_date(merged.get("birth_date_year", ""))
        merged["raw_text"] = ""
        merged["boxes"] = []
        merged["field_confidence"] = {
            field: 0.35 if merged[field] else 0.0 for field in DOC_FIELDS[doc_type]
        }
        merged["ocr_backend"] = f"Ollama: {self._resolved_ollama_model}"
        return merged

    def _recognize_page(self, image: np.ndarray) -> Any:
        with self._engine_lock:
            if self._engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    self._engine = RapidOCR()
                except Exception as error:
                    self._initialization_error = str(error)
                    raise RuntimeError(
                        "文字読取機能を起動できません。依存ライブラリとOCRモデルを確認してください。"
                    ) from error
            return self._engine(image)

    def recognize_document(
        self, image_path: str | Path, doc_type: str
    ) -> dict[str, Any]:
        if doc_type not in DOC_FIELDS:
            return {"_error": f"未対応の書類種別です: {doc_type}"}
        path = Path(image_path)
        if not path.is_file():
            return {"_error": f"画像ファイルが見つかりません: {path}"}
        try:
            if self.backend == "ollama":
                return self._recognize_ollama_document(path, doc_type)
            raw_texts: list[str] = []
            evidence: list[dict[str, Any]] = []
            page_results: list[dict[str, str]] = []
            page_confidences: list[dict[str, float]] = []
            pages: Iterable[Path | np.ndarray]
            pages = _iter_pdf_pages(path) if path.suffix.lower() == ".pdf" else [path]
            page_count = 0
            for page in pages:
                page_count += 1
                if page_count > 100:
                    raise ValueError("書類のページ数が上限（100ページ）を超えています。")
                best_lines: list[OCRLine] = []
                best_rank: tuple[int, float, int, float] | None = None
                recognition_errors: list[Exception] = []
                variants = preprocess_image_variants(page)
                for variant_index, variant in enumerate(variants):
                    try:
                        output = self._recognize_page(variant)
                    except Exception as error:
                        recognition_errors.append(error)
                        continue
                    candidate_lines = _lines_from_output(output)
                    candidate_rank = _recognition_rank(candidate_lines, doc_type)
                    if best_rank is None or candidate_rank > best_rank:
                        best_lines = candidate_lines
                        best_rank = candidate_rank
                    if (
                        variant_index == 1
                        and len(variants) > 2
                        and best_rank[0] >= 2
                        and best_rank[1] >= 1.6
                        and best_rank[2] >= 2
                    ):
                        break
                if best_rank is None:
                    detail = str(recognition_errors[-1]) if recognition_errors else "OCRを実行できませんでした。"
                    raise RuntimeError(f"文字読取に失敗しました: {detail}")
                page_lines = best_lines
                for line in page_lines:
                    evidence.append(
                        {
                            "text": line.text,
                            "box": [[x, y] for x, y in line.box],
                            "confidence": line.confidence,
                        }
                    )
                fields, raw = extract_document_fields(page_lines, doc_type)
                page_results.append(fields)
                page_confidences.append(
                    extract_field_confidences(page_lines, doc_type, fields)
                )
                raw_texts.append(raw)
            merged: dict[str, str] = {field: "" for field in DOC_FIELDS[doc_type]}
            for page_fields in page_results:
                for field, value in page_fields.items():
                    if value and not merged.get(field):
                        merged[field] = value
            merged_confidences: dict[str, float] = {}
            for page_confidence in page_confidences:
                for field, confidence in page_confidence.items():
                    if confidence and field not in merged_confidences:
                        merged_confidences[field] = confidence
            for field in merged:
                if merged.get(field) and field not in merged_confidences:
                    merged_confidences[field] = 0.0
            if "age" in merged and not merged["age"]:
                merged["age"] = _age_from_birth_date(merged.get("birth_date_year", ""))
            if "age" in merged and merged["age"] and not merged_confidences.get("age"):
                merged_confidences["age"] = merged_confidences.get("birth_date_year", 0.0)
            merged["raw_text"] = "\n".join(raw_texts)
            merged["boxes"] = evidence
            merged["field_confidence"] = merged_confidences
            return merged
        except Exception as error:
            return {"_error": f"写真を読み取れませんでした: {error}"}

    def recognize(self, image_path: str | Path) -> dict[str, Any]:
        return self.recognize_document(image_path, "入学願書")

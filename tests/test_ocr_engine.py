import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from ocr_engine import (
    ApplicationOcrEngine,
    OCRLine,
    _read_image,
    configured_ocr_backend,
    extract_document_fields,
    preprocess_image_variants,
)


class ImagePreprocessingTests(unittest.TestCase):
    def test_preprocess_returns_enhanced_and_binary_views(self) -> None:
        image = np.full((700, 500, 3), 245, dtype=np.uint8)
        cv2.putText(image, "TEST FORM", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)

        variants = preprocess_image_variants(image)

        self.assertEqual(len(variants), 2)
        self.assertTrue(all(variant.dtype == np.uint8 for variant in variants))
        self.assertTrue(all(variant.ndim == 2 for variant in variants))
        self.assertGreater(variants[0].shape[0], image.shape[0])

    def test_preprocess_adds_a_rectified_view_for_a_perspective_page(self) -> None:
        image = np.full((700, 900, 3), 100, dtype=np.uint8)
        page = np.array([[180, 60], [710, 90], [770, 620], [120, 590]], dtype=np.int32)
        cv2.fillConvexPoly(image, page, (240, 240, 240))
        cv2.polylines(image, [page], True, (15, 15, 15), 5)
        cv2.putText(image, "FORM", (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)

        variants = preprocess_image_variants(image)

        self.assertEqual(len(variants), 4)
        self.assertNotEqual(variants[2].shape, variants[0].shape)

    def test_exif_orientation_is_applied_when_loading_a_photo(self) -> None:
        image = Image.new("RGB", (80, 40), "white")
        exif = Image.Exif()
        exif[274] = 6
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotated.jpg"
            image.save(path, exif=exif)

            loaded = _read_image(path)

        self.assertEqual(loaded.shape[:2], (80, 40))


class RecognitionSelectionTests(unittest.TestCase):
    def test_ocr_backend_defaults_to_rapidocr_and_allows_ollama_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(configured_ocr_backend(), "rapidocr")

        with patch.dict("os.environ", {"SCHOOL_APPLICATION_OCR_BACKEND": "OLLAMA"}):
            self.assertEqual(configured_ocr_backend(), "ollama")

    def test_furigana_can_be_recovered_from_the_application_row_layout(self) -> None:
        lines = [
            OCRLine("氏名", ((0, 0), (40, 0), (40, 12), (0, 12)), 0.9),
            OCRLine("山田太郎", ((50, 0), (120, 0), (120, 12), (50, 12)), 0.9),
            OCRLine("タロウ", ((50, 20), (100, 20), (100, 32), (50, 32)), 0.55),
            OCRLine("生年月日", ((0, 40), (60, 40), (60, 52), (0, 52)), 0.9),
            OCRLine("2000年1月1日", ((70, 40), (150, 40), (150, 52), (70, 52)), 0.9),
        ]

        fields, _ = extract_document_fields(lines, "入学願書")

        self.assertEqual(fields["furigana"], "タロウ")

    def test_furigana_layout_fallback_does_not_copy_kanji_text(self) -> None:
        lines = [
            OCRLine("氏名", ((0, 0), (40, 0), (40, 12), (0, 12)), 0.9),
            OCRLine("山田太郎", ((50, 0), (120, 0), (120, 12), (50, 12)), 0.9),
            OCRLine("陽子", ((50, 20), (100, 20), (100, 32), (50, 32)), 0.55),
            OCRLine("生年月日", ((0, 40), (60, 40), (60, 52), (0, 52)), 0.9),
            OCRLine("2000年1月1日", ((70, 40), (150, 40), (150, 52), (70, 52)), 0.9),
        ]

        fields, _ = extract_document_fields(lines, "入学願書")

        self.assertEqual(fields["furigana"], "")

    def test_normalization_is_supported_but_misspelled_labels_are_not_guessed(self) -> None:
        lines = [
            OCRLine("ﾌﾘｶﾞﾅ", ((0, 0), (50, 0), (50, 12), (0, 12)), 0.6),
            OCRLine("ヤマダ", ((60, 0), (110, 0), (110, 12), (60, 12)), 0.8),
            OCRLine("出身枝", ((0, 30), (50, 30), (50, 42), (0, 42)), 0.6),
            OCRLine("高校", ((60, 30), (110, 30), (110, 42), (60, 42)), 0.8),
        ]

        fields, _ = extract_document_fields(lines, "入学願書")

        self.assertEqual(fields["furigana"], "ヤマダ")
        self.assertEqual(fields["school_name"], "")

    def test_multiple_fields_on_one_row_are_kept_separate(self) -> None:
        values = [
            ("氏名", 0, 45),
            ("山田花子", 70, 130),
            ("フリガナ", 180, 240),
            ("ヤマダハナコ", 270, 370),
            ("生年月日", 430, 500),
            ("2000年1月1日", 520, 620),
            ("出身校", 690, 740),
            ("青葉中学校", 760, 850),
        ]
        lines = [
            OCRLine(
                text,
                ((left, 10), (right, 10), (right, 24), (left, 24)),
                0.95,
            )
            for text, left, right in values
        ]

        fields, _ = extract_document_fields(lines, "入学願書")

        self.assertEqual(fields["name"], "山田花子")
        self.assertEqual(fields["furigana"], "ヤマダハナコ")
        self.assertEqual(fields["birth_date_year"], "2000年1月1日")
        self.assertEqual(fields["school_name"], "青葉中学校")

    def test_best_preprocessing_view_is_selected_for_field_extraction(self) -> None:
        views = [np.zeros((20, 20), dtype=np.uint8), np.ones((20, 20), dtype=np.uint8)]
        good_text = (
            "氏名", "山田花子", "フリガナ", "ヤマダハナコ",
            "生年月日", "2000年1月1日", "出身校", "青葉中学校",
            "志望理由", "学びたいです",
        )
        good_output = []
        for row, (label, value) in enumerate(zip(good_text[::2], good_text[1::2])):
            y = row * 30
            good_output.extend(
                [
                    ([[0, y], [45, y], [45, y + 12], [0, y + 12]], label, 0.95),
                    ([[70, y], [155, y], [155, y + 12], [70, y + 12]], value, 0.95),
                ]
            )
        weak_output = [([[0, 0], [20, 0], [20, 10], [0, 10]], "?", 0.2)]
        engine = ApplicationOcrEngine()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            path.touch()
            with patch("ocr_engine.preprocess_image_variants", return_value=views):
                with patch.object(engine, "_recognize_page", side_effect=[weak_output, good_output]):
                    result = engine.recognize_document(path, "入学願書")

        self.assertNotIn("_error", result)
        self.assertEqual(result["name"], "山田花子")
        self.assertEqual(result["furigana"], "ヤマダハナコ")
        self.assertEqual(result["school_name"], "青葉中学校")
        self.assertEqual(result["reason"], "学びたいです")


class OllamaRecognitionTests(unittest.TestCase):
    def test_selects_an_installed_vision_model(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama")
        tags = {
            "models": [
                {"name": "text-only", "capabilities": ["completion"]},
                {"name": "gemma4:12b", "capabilities": ["completion", "vision"]},
                {"name": "gemma4:e2b", "capabilities": ["completion", "vision"]},
                {"name": "vision-local", "capabilities": ["completion", "vision"]},
            ]
        }

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(tags).encode("utf-8")),
        ) as open_url:
            self.assertEqual(engine._ollama_model(), "gemma4:e2b")

        self.assertIn("127.0.0.1:11434/api/tags", open_url.call_args.args[0].full_url)

    def test_ollama_page_uses_loopback_and_marks_results_for_review(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        response = {
            "message": {
                "content": json.dumps(
                    {
                        "name": "山田花子",
                        "furigana": "ヤマダハナコ",
                        "birth_date_year": "",
                        "age": "",
                        "school_name": "青葉中学校",
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            }
        }
        image = np.full((120, 160, 3), 255, dtype=np.uint8)

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(response).encode("utf-8")),
        ) as open_url:
            fields = engine._recognize_ollama_page(image, "入学願書")

        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["model"], "vision-local")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(len(payload["messages"][0]["images"]), 1)
        self.assertEqual(fields["name"], "山田花子")
        self.assertEqual(fields["school_name"], "青葉中学校")
        self.assertEqual(fields["birth_date_year"], "")

    def test_ollama_page_accepts_a_json_code_fence(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        extracted = {
            "name": "山田花子",
            "furigana": "ヤマダハナコ",
            "birth_date_year": "",
            "age": "",
            "school_name": "青葉中学校",
            "reason": "",
        }
        response = {
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": (
                    "読み取り結果です。\n```json\n"
                    + json.dumps(extracted, ensure_ascii=False)
                    + "\n```"
                ),
            },
        }

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8")),
        ):
            fields = engine._recognize_ollama_page(
                np.full((120, 160, 3), 255, dtype=np.uint8), "入学願書"
            )

        self.assertEqual(fields["name"], "山田花子")
        self.assertEqual(fields["furigana"], "ヤマダハナコ")
        self.assertEqual(fields["school_name"], "青葉中学校")

    def test_invalid_ollama_content_reports_safe_response_diagnostics(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        private_like_text = "氏名は山田花子です"
        response = {
            "done_reason": "length",
            "message": {
                "role": "assistant",
                "content": private_like_text,
                "thinking": "この値をエラー表示に含めない",
            },
        }

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(response, ensure_ascii=False).encode("utf-8")),
        ):
            with self.assertRaisesRegex(RuntimeError, "終了理由=length") as raised:
                engine._recognize_ollama_page(
                    np.full((120, 160, 3), 255, dtype=np.uint8), "入学願書"
                )

        self.assertIn("応答キー=role,content,thinking", str(raised.exception))
        self.assertIn("JSONとして解析できません", str(raised.exception))
        self.assertNotIn(private_like_text, str(raised.exception))

    def test_empty_ollama_content_is_distinguished_without_exposing_text(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        response = {
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "   "},
        }

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(response).encode("utf-8")),
        ):
            with self.assertRaisesRegex(RuntimeError, "応答本文が空です") as raised:
                engine._recognize_ollama_page(
                    np.full((120, 160, 3), 255, dtype=np.uint8), "入学願書"
                )

        self.assertIn("終了理由=stop", str(raised.exception))
        self.assertIn("応答キー=role,content", str(raised.exception))

    def test_configured_model_must_be_installed(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="missing-model")
        tags = {"models": [{"name": "other", "capabilities": ["vision"]}]}

        with patch(
            "ocr_engine.urlopen",
            return_value=BytesIO(json.dumps(tags).encode("utf-8")),
        ):
            with self.assertRaisesRegex(RuntimeError, "指定されたOllamaモデルが見つかりません"):
                engine._ollama_model()

    def test_unknown_ollama_model_architecture_has_actionable_error(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            500,
            "server error",
            {},
            BytesIO(b'{"error":"unknown model architecture: mllama"}'),
        )

        with patch("ocr_engine.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Ollamaを最新版へ更新"):
                engine._recognize_ollama_page(
                    np.full((30, 30, 3), 255, dtype=np.uint8), "入学願書"
                )

    def test_ollama_http_failure_reports_status_without_server_details(self) -> None:
        engine = ApplicationOcrEngine(backend="ollama", ollama_model="vision-local")
        engine._resolved_ollama_model = "vision-local"
        error = HTTPError(
            "http://127.0.0.1:11434/api/chat",
            500,
            "server error",
            {},
            BytesIO(b"internal model details"),
        )

        with patch("ocr_engine.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500") as raised:
                engine._recognize_ollama_page(
                    np.full((30, 30, 3), 255, dtype=np.uint8), "入学願書"
                )

        self.assertNotIn("internal model details", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

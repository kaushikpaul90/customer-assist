import io
import unittest
from fastapi.testclient import TestClient

import app.main as main_module


class DummyRateLimiter:
    def consume(self, n=1):
        return True


class DummyCache:
    def __init__(self):
        self.store = {}
    def get(self, k):
        return None
    def set(self, k, v):
        self.store[k] = v


class FastAPIEndpointIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Patch rate limiter and cache to avoid side effects
        main_module.GLOBAL_RATE_LIMITER = DummyRateLimiter()
        main_module.RESPONSE_CACHE = DummyCache()

        # Patch model functions imported into main_module
        main_module.answer_question = lambda ctx, q: {"question": q, "answer": "mocked answer", "token_count": 5}
        main_module.summarize_text = lambda t: ("mocked summary", "mocked prompt", "v1", 10)
        main_module.translate_text = lambda text, src_lang, target_lang: ("mocked translation", "mocked prompt", "v1", 7)
        main_module.detect_defect = lambda b: {"is_defective": True, "predicted_label": "broken", "eligible_for_return": True}
        main_module.transcribe_audio = lambda b: ({"transcription": "hello world"}, 4)
        main_module.transcribe_and_translate_audio = lambda b: ("translated audio text", 6)
        main_module.auto_grade_response = lambda endpoint, out: (0.9, "ok")
        main_module.get_prompt_template = lambda key: ("template", "v1")

        self.client = TestClient(main_module.app)

    def test_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("message", r.json())

    def test_question_answering(self):
        payload = {"context": "some context", "question": "what?"}
        r = self.client.post("/question-answering", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("answer", data)
        self.assertIn("transaction_id", data)

    def test_summarize_text(self):
        payload = {"text": "long conversation here"}
        r = self.client.post("/summarize-text", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("summary", data)
        self.assertIn("transaction_id", data)

    def test_translate_text(self):
        payload = {"text": "hello", "src_lang": "hi", "target_lang": "eng"}
        r = self.client.post("/translate-text", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("translation", data)
        self.assertIn("transaction_id", data)

    def test_check_item_return_eligibility(self):
        # send a small PNG-like bytes payload
        file_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 100
        files = {"file": ("test.png", io.BytesIO(file_bytes), "image/png")}
        r = self.client.post("/check-item-return-eligibility", files=files)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("predicted_label", data)
        self.assertTrue(data.get("eligible_for_return"))

    def test_audio_transcribe(self):
        file_bytes = b"RIFF" + b"0" * 100
        files = {"file": ("audio.wav", io.BytesIO(file_bytes), "audio/wav")}
        r = self.client.post("/audio-transcribe", files=files)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("transcription", data)
        self.assertIn("transaction_id", data)

    def test_audio_transcribe_translate(self):
        file_bytes = b"RIFF" + b"0" * 100
        files = {"file": ("audio.wav", io.BytesIO(file_bytes), "audio/wav")}
        r = self.client.post("/audio-transcribe-translate", files=files)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("translation", data)
        self.assertIn("transaction_id", data)


if __name__ == "__main__":
    unittest.main()

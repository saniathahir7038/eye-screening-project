from io import BytesIO
import unittest
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ScreeningAppTests(unittest.TestCase):
    def test_empty_state_renders_without_errors(self):
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Pterygium screening prototype")
        self.assertEqual(len(app.file_uploader), 1)
        self.assertIn("not a diagnosis", app.info[0].value)
        self.assertTrue(any("Hold the phone level" in block.value for block in app.markdown))

    def test_failed_quality_gate_stops_before_model_inference(self):
        stream = BytesIO()
        Image.new("RGB", (100, 100), (4, 4, 4)).save(stream, format="PNG")
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        app.file_uploader[0].upload("dark.png", stream.getvalue(), "image/png").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertIn("Image quality insufficient", app.error[0].value)
        self.assertFalse(any(header.value == "Screening result" for header in app.subheader))


if __name__ == "__main__":
    unittest.main()

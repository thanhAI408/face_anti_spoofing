import tempfile
import unittest
from pathlib import Path

from face_antispoofing.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config" / "default.toml")
        self.assertTrue(config.challenge.enabled)
        self.assertEqual(config.texture.input_size, 224)
        self.assertEqual(config.challenge.actions[0], "blink")

    def test_unknown_section_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text("[unknown]\nvalue=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_fusion_thresholds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(
                "[fusion]\nspoof_threshold=0.8\nlive_threshold=0.7\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

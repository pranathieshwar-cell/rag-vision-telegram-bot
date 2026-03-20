from __future__ import annotations

from pathlib import Path

from utils import extract_tags


class VisionError(RuntimeError):
    pass


class VisionCaptioner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise VisionError(
                "transformers is not installed. Install requirements to enable vision mode."
            ) from exc

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)

    def caption(self, image_path: Path) -> tuple[str, list[str]]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise VisionError("Pillow is not installed.") from exc

        self._load()

        if self._processor is None or self._model is None:
            raise VisionError("Vision model failed to load.")

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(image, return_tensors="pt")
        output_ids = self._model.generate(**inputs, max_new_tokens=30)
        caption = self._processor.decode(output_ids[0], skip_special_tokens=True).strip()
        tags = extract_tags(caption, k=3)
        return caption, tags

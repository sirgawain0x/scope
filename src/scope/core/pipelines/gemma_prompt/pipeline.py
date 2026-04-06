import logging
import os
import threading
from typing import TYPE_CHECKING

import torch
from PIL import Image

from ..interface import Pipeline, Requirements
from .schema import DEFAULT_SYSTEM_INSTRUCTION, GemmaPromptConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-4-e4b-it"


class GemmaPromptPipeline(Pipeline):
    """VLM pipeline using Gemma 4 E4B for real-time frame-to-prompt generation."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return GemmaPromptConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.7,
        max_tokens: int = 256,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_instruction = system_instruction

        self._last_prompt: str | None = None
        self._lock = threading.Lock()

        # Check for HF_TOKEN
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN environment variable not set. "
                "Gemma 4 requires accepting the license on Hugging Face. "
                "Set HF_TOKEN or use the /api/v1/keys endpoint."
            )

        # Load model and processor with 8-bit quantization
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        logger.info(f"Loading Gemma 4 E4B model from {MODEL_ID} with 8-bit quantization...")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=self.dtype,
            quantization_config=quantization_config,
        )
        self.model.eval()

        logger.info("Gemma 4 E4B model loaded successfully")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Passthrough: returns input video unchanged.

        The VLM analysis is triggered separately via analyze_frame(),
        called from the WebRTC data channel handler.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None")

        # Return frames as-is, normalized to [0, 1] float range
        if isinstance(video, list) and len(video) > 0:
            frames = torch.stack([f.squeeze(0) for f in video], dim=0)
            frames = frames.float() / 255.0
            # Output shape: THWC with T=num_frames
            frames = frames.unsqueeze(-1) if frames.dim() == 3 else frames
            return {"video": frames}

        return {"video": video}

    @torch.no_grad()
    def analyze_frame(self, image: Image.Image) -> str:
        """Analyze a single image frame and generate a descriptive prompt.

        Args:
            image: PIL Image from the webcam frame.

        Returns:
            Generated text prompt string.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.system_instruction},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        output = self.model.generate(**inputs, **generation_kwargs)

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_len:]
        prompt_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        prompt_text = prompt_text.strip()

        with self._lock:
            self._last_prompt = prompt_text

        logger.debug(f"Generated prompt: {prompt_text[:100]}...")
        return prompt_text

    def get_last_prompt(self) -> str | None:
        """Return the last generated prompt (thread-safe)."""
        with self._lock:
            return self._last_prompt

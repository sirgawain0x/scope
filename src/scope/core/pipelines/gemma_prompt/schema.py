from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact

from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a master video director. Analyze this image and generate a highly "
    "descriptive, comma-separated text prompt for an AI video pipeline. Focus on "
    "lighting, subject matter, and mood. Output ONLY the comma-separated prompt."
)


class GemmaPromptConfig(BasePipelineConfig):
    pipeline_id: ClassVar[str] = "gemma-prompt"
    pipeline_name: ClassVar[str] = "Gemma Prompt Engine"
    pipeline_description: ClassVar[str] = (
        "VLM pipeline that analyzes webcam frames and generates descriptive "
        "text prompts using Google Gemma 4 E4B. Acts as a real-time Prompt "
        "Engine for other video generation pipelines."
    )
    pipeline_version: ClassVar[str] = "1.0.0"
    estimated_vram_gb: ClassVar[float] = 5.0
    requires_models: ClassVar[bool] = True

    supports_prompts: ClassVar[bool] = False

    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    artifacts: ClassVar[list] = [
        HuggingfaceRepoArtifact(
            repo_id="google/gemma-4-e4b-it",
            files=["config.json", "model.safetensors.index.json"],
        ),
    ]

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for text generation",
        json_schema_extra=ui_field_config(order=1),
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum number of tokens to generate",
        json_schema_extra=ui_field_config(order=2),
    )
    system_instruction: str = Field(
        default=DEFAULT_SYSTEM_INSTRUCTION,
        description="System instruction sent to the VLM before each frame analysis",
        json_schema_extra=ui_field_config(order=3),
    )

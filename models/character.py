"""Character profile data model for consistency vault."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CharacterProfile:
    """Persistent character/style profile for cross-session consistency.

    Stores trigger words, LoRA bindings, reference images, style presets,
    and default generation parameters so that an Agent can maintain visual
    consistency across multiple generation rounds by simply specifying a
    ``character_id``.
    """

    character_id: str  # Unique slug, e.g. "detective_john"
    display_name: str  # Human-readable name
    description: str = ""  # Free-form character description

    # Prompt injection
    trigger_words: str = ""  # Prepended to positive prompt
    negative_trigger: str = ""  # Prepended to negative prompt

    # LoRA binding
    lora_name: Optional[str] = None  # Exact LoRA filename in ComfyUI
    lora_strength: float = 0.75  # LoRA model strength (0.0-1.0)

    # Reference images (paths relative to ComfyUI input/ directory)
    reference_images: List[str] = field(default_factory=list)

    # Style preset keyword
    style_preset: Optional[str] = None  # e.g. "anime", "photorealistic"

    # Default generation parameter overrides
    default_params: Dict[str, Any] = field(default_factory=dict)

    # Categorization
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

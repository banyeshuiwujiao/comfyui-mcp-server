"""Character & Style Consistency Vault — SQLite-persisted profile manager.

Stores character profiles (trigger words, LoRA bindings, reference images,
style presets, default generation params) in SQLite so that AI Agents can
maintain visual consistency across sessions by simply specifying a
``character_id``.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.character import CharacterProfile

logger = logging.getLogger("MCP_Server__character_vault")

# Built-in style preset keyword expansions
STYLE_PRESETS: Dict[str, str] = {
    "anime": "anime style, cel shading, clean lineart, vibrant colors",
    "photorealistic": "photorealistic, 8K, RAW photo, film grain, sharp focus",
    "oil_painting": "oil painting style, visible brushstrokes, canvas texture, rich colors",
    "watercolor": "watercolor painting, soft edges, translucent washes, paper texture",
    "pixel_art": "pixel art style, retro game aesthetic, limited palette, crisp pixels",
    "cyberpunk": "cyberpunk aesthetic, neon lights, dark atmosphere, high-tech low-life",
    "fantasy": "fantasy art style, magical atmosphere, ethereal lighting, detailed illustration",
    "comic": "comic book style, bold outlines, halftone dots, dynamic composition",
    "3d_render": "3D render, Octane render, global illumination, subsurface scattering",
    "sketch": "pencil sketch, graphite drawing, cross-hatching, paper texture",
}


class CharacterVault:
    """SQLite-backed character profile storage and prompt injection engine."""

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()

        if db_path is None:
            db_env = os.getenv("COMFY_MCP_DB_PATH")
            if db_env:
                self.db_path = db_env
            else:
                # Library default stays in-memory for test isolation; the server
                # passes the shared persistent data/assets.db explicitly.
                self.db_path = ":memory:"
        else:
            self.db_path = db_path

        self._shared_mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._shared_mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_mem_conn.row_factory = sqlite3.Row

        self._init_db()
        logger.info(f"Initialized CharacterVault with SQLite DB at {self.db_path}")

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:" and self._shared_mem_conn:
            return self._shared_mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_if_needed(self, conn: sqlite3.Connection):
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self):
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            conn = self._get_connection()
            try:
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS character_profiles (
                        character_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        trigger_words TEXT NOT NULL DEFAULT '',
                        negative_trigger TEXT NOT NULL DEFAULT '',
                        lora_name TEXT,
                        lora_strength REAL NOT NULL DEFAULT 0.75,
                        reference_images_json TEXT NOT NULL DEFAULT '[]',
                        style_preset TEXT,
                        default_params_json TEXT NOT NULL DEFAULT '{}',
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_char_tags ON character_profiles(tags_json);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_char_style ON character_profiles(style_preset);"
                )
                conn.commit()
            finally:
                self._close_if_needed(conn)

    # ------------------------------------------------------------------
    # Row ↔ Model conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> CharacterProfile:
        return CharacterProfile(
            character_id=row["character_id"],
            display_name=row["display_name"],
            description=row["description"] or "",
            trigger_words=row["trigger_words"] or "",
            negative_trigger=row["negative_trigger"] or "",
            lora_name=row["lora_name"],
            lora_strength=row["lora_strength"],
            reference_images=json.loads(row["reference_images_json"] or "[]"),
            style_preset=row["style_preset"],
            default_params=json.loads(row["default_params_json"] or "{}"),
            tags=json.loads(row["tags_json"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def save_profile(
        self,
        character_id: str,
        display_name: str,
        description: str = "",
        trigger_words: str = "",
        negative_trigger: str = "",
        lora_name: Optional[str] = None,
        lora_strength: float = 0.75,
        reference_images: Optional[List[str]] = None,
        style_preset: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> CharacterProfile:
        """Create or update a character profile (upsert)."""
        now = datetime.now().isoformat()
        ref_imgs = reference_images or []
        params = default_params or {}
        tag_list = tags or []

        with self._lock:
            conn = self._get_connection()
            try:
                # Check if exists to preserve created_at on update
                existing = conn.execute(
                    "SELECT created_at FROM character_profiles WHERE character_id = ?",
                    (character_id,),
                ).fetchone()
                created_at = existing["created_at"] if existing else now

                conn.execute(
                    """
                    INSERT INTO character_profiles
                        (character_id, display_name, description, trigger_words,
                         negative_trigger, lora_name, lora_strength,
                         reference_images_json, style_preset, default_params_json,
                         tags_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(character_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        description = excluded.description,
                        trigger_words = excluded.trigger_words,
                        negative_trigger = excluded.negative_trigger,
                        lora_name = excluded.lora_name,
                        lora_strength = excluded.lora_strength,
                        reference_images_json = excluded.reference_images_json,
                        style_preset = excluded.style_preset,
                        default_params_json = excluded.default_params_json,
                        tags_json = excluded.tags_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        character_id, display_name, description, trigger_words,
                        negative_trigger, lora_name, lora_strength,
                        json.dumps(ref_imgs), style_preset, json.dumps(params),
                        json.dumps(tag_list), created_at, now,
                    ),
                )
                conn.commit()
            finally:
                self._close_if_needed(conn)

        return CharacterProfile(
            character_id=character_id,
            display_name=display_name,
            description=description,
            trigger_words=trigger_words,
            negative_trigger=negative_trigger,
            lora_name=lora_name,
            lora_strength=lora_strength,
            reference_images=ref_imgs,
            style_preset=style_preset,
            default_params=params,
            tags=tag_list,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(now),
        )

    def get_profile(self, character_id: str) -> Optional[CharacterProfile]:
        """Retrieve a character profile by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM character_profiles WHERE character_id = ?",
                    (character_id,),
                ).fetchone()
                if row:
                    return self._row_to_profile(row)
                return None
            finally:
                self._close_if_needed(conn)

    def list_profiles(
        self,
        tag: Optional[str] = None,
        style_preset: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[CharacterProfile]:
        """List character profiles with optional filtering."""
        with self._lock:
            conn = self._get_connection()
            try:
                clauses: list[str] = []
                params: list[Any] = []

                if tag:
                    clauses.append("tags_json LIKE ?")
                    params.append(f'%"{tag}"%')
                if style_preset:
                    clauses.append("style_preset = ?")
                    params.append(style_preset)
                if query:
                    clauses.append(
                        "(display_name LIKE ? OR description LIKE ? OR trigger_words LIKE ? OR character_id LIKE ?)"
                    )
                    q = f"%{query}%"
                    params.extend([q, q, q, q])

                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                sql = f"SELECT * FROM character_profiles {where} ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_profile(r) for r in rows]
            finally:
                self._close_if_needed(conn)

    def delete_profile(self, character_id: str) -> bool:
        """Delete a character profile. Returns True if it existed."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM character_profiles WHERE character_id = ?",
                    (character_id,),
                )
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted character profile: {character_id}")
                return deleted
            finally:
                self._close_if_needed(conn)

    # ------------------------------------------------------------------
    # Prompt injection engine
    # ------------------------------------------------------------------

    def apply_character(
        self,
        character_id: str,
        prompt: str,
        negative_prompt: str = "",
    ) -> Dict[str, Any]:
        """Inject character profile features into a prompt.

        Returns a dict with enhanced prompt/negative_prompt plus LoRA and
        reference image info that the caller can feed into generation tools.
        """
        profile = self.get_profile(character_id)
        if profile is None:
            return {"error": f"Character profile '{character_id}' not found"}

        # Positive prompt: prepend trigger words
        enhanced_prompt = prompt
        if profile.trigger_words:
            if prompt and prompt.strip():
                enhanced_prompt = f"{profile.trigger_words}, {prompt}"
            else:
                enhanced_prompt = profile.trigger_words

        # Negative prompt: prepend negative trigger
        enhanced_negative = negative_prompt
        if profile.negative_trigger:
            if negative_prompt and negative_prompt.strip():
                enhanced_negative = f"{profile.negative_trigger}, {negative_prompt}"
            else:
                enhanced_negative = profile.negative_trigger

        # Style preset expansion
        if profile.style_preset:
            style_suffix = STYLE_PRESETS.get(profile.style_preset, profile.style_preset)
            if enhanced_prompt and enhanced_prompt.strip():
                enhanced_prompt = f"{enhanced_prompt}, {style_suffix}"
            else:
                enhanced_prompt = style_suffix

        return {
            "character_id": character_id,
            "display_name": profile.display_name,
            "prompt": enhanced_prompt,
            "negative_prompt": enhanced_negative,
            "lora_name": profile.lora_name,
            "lora_strength": profile.lora_strength,
            "reference_images": profile.reference_images,
            "style_preset": profile.style_preset,
            "default_params": profile.default_params,
        }

    def profile_to_dict(self, profile: CharacterProfile) -> Dict[str, Any]:
        """Serialize a CharacterProfile to a JSON-friendly dict."""
        return {
            "character_id": profile.character_id,
            "display_name": profile.display_name,
            "description": profile.description,
            "trigger_words": profile.trigger_words,
            "negative_trigger": profile.negative_trigger,
            "lora_name": profile.lora_name,
            "lora_strength": profile.lora_strength,
            "reference_images": profile.reference_images,
            "style_preset": profile.style_preset,
            "default_params": profile.default_params,
            "tags": profile.tags,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        }

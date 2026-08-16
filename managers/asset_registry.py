"""Asset registry with SQLite persistence and lineage tracking"""

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.asset import AssetRecord

logger = logging.getLogger("MCP_Server__asset_registry")


def _make_asset_key(filename: str, subfolder: str, folder_type: str) -> str:
    """Create a stable lookup key from asset identity."""
    return f"{folder_type}:{subfolder}:{filename}"


class AssetRegistry:
    """Manages tracking, SQLite persistence, and lineage of generated assets.
    
    Uses (filename, subfolder, type) as stable identity instead of URL,
    making the system robust to URL changes (e.g., different hostnames).
    All records are persisted to SQLite, surviving server restarts.
    """
    
    def __init__(
        self,
        ttl_hours: int = 24,
        comfyui_base_url: str = "http://localhost:8188",
        db_path: Optional[str] = None
    ):
        self.ttl_hours = ttl_hours
        self.comfyui_base_url = comfyui_base_url
        self._lock = threading.RLock()

        # In-memory L1 cache
        self._assets: Dict[str, AssetRecord] = {}  # asset_id -> AssetRecord
        self._asset_key_to_id: Dict[str, str] = {}  # key -> asset_id

        # Determine SQLite database path
        if db_path is None:
            db_env = os.getenv("COMFY_MCP_DB_PATH")
            if db_env:
                self.db_path = db_env
            else:
                # Library default stays in-memory so unit tests remain isolated;
                # the server explicitly passes the persistent data/assets.db
                # path (docs: data/assets.db, lineage survives restarts).
                self.db_path = ":memory:"
        else:
            self.db_path = db_path

        # If in-memory, hold a single shared connection
        self._shared_mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._shared_mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_mem_conn.row_factory = sqlite3.Row

        # Initialize SQLite database
        self._init_db()
        logger.info(f"Initialized AssetRegistry with SQLite DB at {self.db_path} (TTL: {ttl_hours}h)")

    def _get_connection(self) -> sqlite3.Connection:
        """Get connection to SQLite DB."""
        if self.db_path == ":memory:" and self._shared_mem_conn:
            return self._shared_mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create SQLite tables and indices if not present."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            conn = self._get_connection()
            try:
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS assets (
                        asset_id TEXT PRIMARY KEY,
                        parent_asset_id TEXT,
                        root_asset_id TEXT,
                        filename TEXT NOT NULL,
                        subfolder TEXT NOT NULL DEFAULT '',
                        folder_type TEXT NOT NULL DEFAULT 'output',
                        prompt_id TEXT NOT NULL DEFAULT '',
                        workflow_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT,
                        mime_type TEXT NOT NULL,
                        width INTEGER,
                        height INTEGER,
                        bytes_size INTEGER NOT NULL DEFAULT 0,
                        sha256 TEXT,
                        prompt TEXT,
                        negative_prompt TEXT,
                        seed INTEGER,
                        generation_type TEXT,
                        tags_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        comfy_history_json TEXT,
                        submitted_workflow_json TEXT,
                        session_id TEXT
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_parent ON assets(parent_asset_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_root ON assets(root_asset_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_workflow ON assets(workflow_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_identity ON assets(folder_type, subfolder, filename);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_session ON assets(session_id);")
                conn.commit()
            finally:
                if self.db_path != ":memory:":
                    conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> AssetRecord:
        """Convert a SQLite row to an AssetRecord instance."""
        created_at_dt = datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
        expires_at_dt = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None

        tags = []
        if row["tags_json"]:
            try:
                tags = json.loads(row["tags_json"])
            except Exception:
                tags = []

        metadata = {}
        if row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}

        comfy_history = None
        if row["comfy_history_json"]:
            try:
                comfy_history = json.loads(row["comfy_history_json"])
            except Exception:
                comfy_history = None

        submitted_workflow = None
        if row["submitted_workflow_json"]:
            try:
                submitted_workflow = json.loads(row["submitted_workflow_json"])
            except Exception:
                submitted_workflow = None

        record = AssetRecord(
            asset_id=row["asset_id"],
            filename=row["filename"],
            subfolder=row["subfolder"],
            folder_type=row["folder_type"],
            prompt_id=row["prompt_id"],
            workflow_id=row["workflow_id"],
            created_at=created_at_dt,
            expires_at=expires_at_dt,
            mime_type=row["mime_type"],
            width=row["width"],
            height=row["height"],
            bytes_size=row["bytes_size"] or 0,
            sha256=row["sha256"],
            comfy_history=comfy_history,
            submitted_workflow=submitted_workflow,
            metadata=metadata,
            session_id=row["session_id"],
            parent_asset_id=row["parent_asset_id"],
            root_asset_id=row["root_asset_id"],
            generation_type=row["generation_type"],
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            seed=row["seed"],
            tags=tags
        )
        record.set_base_url(self.comfyui_base_url)
        return record

    def register_asset(
        self,
        filename: str,
        subfolder: str,
        folder_type: str,
        workflow_id: str,
        prompt_id: str,
        mime_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bytes_size: Optional[int] = None,
        comfy_history: Optional[Dict[str, Any]] = None,
        submitted_workflow: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        parent_asset_id: Optional[str] = None,
        generation_type: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> AssetRecord:
        """Register a new asset with SQLite persistence and lineage linking."""
        with self._lock:
            asset_key = _make_asset_key(filename, subfolder, folder_type)

            # Check if asset already exists in memory or SQLite
            existing = self.get_asset_by_identity(filename, subfolder, folder_type)
            if existing:
                # Update existing record if needed
                if comfy_history is not None:
                    existing.comfy_history = comfy_history
                if submitted_workflow is not None:
                    existing.submitted_workflow = submitted_workflow
                if parent_asset_id is not None and not existing.parent_asset_id:
                    existing.parent_asset_id = parent_asset_id
                if generation_type is not None:
                    existing.generation_type = generation_type
                self._save_to_db(existing)
                return existing

            # Generate unique asset_id
            asset_id = str(uuid.uuid4())
            now = datetime.now()
            expires_at = now + timedelta(hours=self.ttl_hours) if self.ttl_hours > 0 else None

            # Resolve root_asset_id from parent_asset_id if provided
            root_asset_id = asset_id
            if parent_asset_id:
                parent_rec = self.get_asset(parent_asset_id)
                if parent_rec:
                    root_asset_id = parent_rec.root_asset_id or parent_rec.asset_id
                else:
                    root_asset_id = parent_asset_id

            # Create Record
            record = AssetRecord(
                asset_id=asset_id,
                filename=filename,
                subfolder=subfolder,
                folder_type=folder_type,
                prompt_id=prompt_id or "",
                workflow_id=workflow_id,
                created_at=now,
                expires_at=expires_at,
                mime_type=mime_type or "application/octet-stream",
                width=width,
                height=height,
                bytes_size=bytes_size or 0,
                sha256=None,
                comfy_history=comfy_history,
                submitted_workflow=submitted_workflow,
                metadata=metadata or {},
                session_id=session_id,
                parent_asset_id=parent_asset_id,
                root_asset_id=root_asset_id,
                generation_type=generation_type,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                tags=tags or []
            )
            record.set_base_url(self.comfyui_base_url)

            # Save to L1 memory cache & L2 SQLite
            self._assets[asset_id] = record
            self._asset_key_to_id[asset_key] = asset_id
            self._save_to_db(record)

            logger.debug(f"Registered asset {asset_id} ({asset_key}) with root {root_asset_id}")
            return record

    def _save_to_db(self, record: AssetRecord):
        """Insert or replace record in SQLite database."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO assets (
                    asset_id, parent_asset_id, root_asset_id, filename, subfolder, folder_type,
                    prompt_id, workflow_id, created_at, expires_at, mime_type, width, height,
                    bytes_size, sha256, prompt, negative_prompt, seed, generation_type,
                    tags_json, metadata_json, comfy_history_json, submitted_workflow_json, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.asset_id,
                record.parent_asset_id,
                record.root_asset_id,
                record.filename,
                record.subfolder,
                record.folder_type,
                record.prompt_id,
                record.workflow_id,
                record.created_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
                record.mime_type,
                record.width,
                record.height,
                record.bytes_size,
                record.sha256,
                record.prompt,
                record.negative_prompt,
                record.seed,
                record.generation_type,
                json.dumps(record.tags, ensure_ascii=False),
                json.dumps(record.metadata, ensure_ascii=False),
                json.dumps(record.comfy_history, ensure_ascii=False) if record.comfy_history else None,
                json.dumps(record.submitted_workflow, ensure_ascii=False) if record.submitted_workflow else None,
                record.session_id
            ))
            conn.commit()
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        """Retrieve asset record by ID from memory cache or SQLite."""
        if not asset_id:
            return None

        with self._lock:
            # 1. Check L1 memory cache
            record = self._assets.get(asset_id)
            if record:
                if record.expires_at and datetime.now() > record.expires_at:
                    self._remove_from_memory(asset_id)
                    return None
                return record

            # 2. Check L2 SQLite DB
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
                row = cursor.fetchone()
                if row:
                    rec = self._row_to_record(row)
                    if rec.expires_at and datetime.now() > rec.expires_at:
                        return None
                    # Populate memory cache
                    self._assets[rec.asset_id] = rec
                    asset_key = _make_asset_key(rec.filename, rec.subfolder, rec.folder_type)
                    self._asset_key_to_id[asset_key] = rec.asset_id
                    return rec
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            return None

    def get_asset_by_identity(
        self, filename: str, subfolder: str, folder_type: str
    ) -> Optional[AssetRecord]:
        """Get asset record by stable identity (filename, subfolder, type)."""
        with self._lock:
            asset_key = _make_asset_key(filename, subfolder, folder_type)
            cached_id = self._asset_key_to_id.get(asset_key)
            if cached_id:
                rec = self.get_asset(cached_id)
                if rec:
                    return rec

            # Query SQLite
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE filename = ? AND subfolder = ? AND folder_type = ?",
                    (filename, subfolder, folder_type)
                )
                row = cursor.fetchone()
                if row:
                    rec = self._row_to_record(row)
                    if rec.expires_at and datetime.now() > rec.expires_at:
                        return None
                    self._assets[rec.asset_id] = rec
                    self._asset_key_to_id[asset_key] = rec.asset_id
                    return rec
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            return None

    def list_assets(
        self,
        limit: int = 10,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_asset_id: Optional[str] = None,
        root_asset_id: Optional[str] = None
    ) -> List[AssetRecord]:
        """List recent assets sorted by creation time (newest first)."""
        with self._lock:
            self.cleanup_expired()

            query = "SELECT * FROM assets WHERE 1=1"
            params = []

            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if parent_asset_id:
                query += " AND parent_asset_id = ?"
                params.append(parent_asset_id)

            if root_asset_id:
                query += " AND root_asset_id = ?"
                params.append(root_asset_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            conn = self._get_connection()
            try:
                cursor = conn.execute(query, tuple(params))
                rows = cursor.fetchall()
                records = [self._row_to_record(r) for r in rows]
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            # Update L1 cache
            for rec in records:
                self._assets[rec.asset_id] = rec
                key = _make_asset_key(rec.filename, rec.subfolder, rec.folder_type)
                self._asset_key_to_id[key] = rec.asset_id

            return records

    def search_assets(
        self,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[AssetRecord]:
        """Search assets by keyword query (in prompt/filename), tag, or workflow."""
        with self._lock:
            sql = "SELECT * FROM assets WHERE 1=1"
            params = []

            if query:
                like_pattern = f"%{query}%"
                sql += " AND (prompt LIKE ? OR filename LIKE ? OR metadata_json LIKE ?)"
                params.extend([like_pattern, like_pattern, like_pattern])

            if tag:
                tag_pattern = f'%"{tag}"%'
                sql += " AND tags_json LIKE ?"
                params.append(tag_pattern)

            if workflow_id:
                sql += " AND workflow_id = ?"
                params.append(workflow_id)

            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            conn = self._get_connection()
            try:
                cursor = conn.execute(sql, tuple(params))
                rows = cursor.fetchall()
                records = [self._row_to_record(r) for r in rows]
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            return records

    def get_lineage(self, asset_id: str) -> Dict[str, Any]:
        """Retrieve the complete ancestry chain and derived children for an asset."""
        target = self.get_asset(asset_id)
        if not target:
            return {"error": f"Asset {asset_id} not found"}

        with self._lock:
            # 1. Trace ancestors upwards
            ancestors = []
            curr_parent_id = target.parent_asset_id
            visited = {asset_id}

            while curr_parent_id and curr_parent_id not in visited:
                visited.add(curr_parent_id)
                parent_rec = self.get_asset(curr_parent_id)
                if not parent_rec:
                    break
                ancestors.append({
                    "asset_id": parent_rec.asset_id,
                    "workflow_id": parent_rec.workflow_id,
                    "generation_type": parent_rec.generation_type or "t2i",
                    "filename": parent_rec.filename,
                    "asset_url": parent_rec.asset_url,
                    "created_at": parent_rec.created_at.isoformat(),
                    "prompt": parent_rec.prompt,
                })
                curr_parent_id = parent_rec.parent_asset_id

            # 2. Find direct children & family members
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE parent_asset_id = ? ORDER BY created_at ASC",
                    (asset_id,)
                )
                child_rows = cursor.fetchall()
                children = [
                    {
                        "asset_id": r["asset_id"],
                        "workflow_id": r["workflow_id"],
                        "generation_type": r["generation_type"] or "derived",
                        "filename": r["filename"],
                        "created_at": r["created_at"],
                        "prompt": r["prompt"],
                    }
                    for r in child_rows
                ]

                # 3. Find all family members (under the same root)
                root_id = target.root_asset_id or asset_id
                cursor_fam = conn.execute(
                    "SELECT asset_id, parent_asset_id, workflow_id, generation_type, created_at, filename FROM assets WHERE root_asset_id = ? OR asset_id = ? ORDER BY created_at ASC",
                    (root_id, root_id)
                )
                family_members = [
                    {
                        "asset_id": r["asset_id"],
                        "parent_asset_id": r["parent_asset_id"],
                        "workflow_id": r["workflow_id"],
                        "generation_type": r["generation_type"] or "base",
                        "created_at": r["created_at"],
                        "filename": r["filename"],
                    }
                    for r in cursor_fam.fetchall()
                ]
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            return {
                "asset_id": target.asset_id,
                "root_asset_id": target.root_asset_id,
                "parent_asset_id": target.parent_asset_id,
                "generation_type": target.generation_type or "base",
                "workflow_id": target.workflow_id,
                "created_at": target.created_at.isoformat(),
                "ancestors": ancestors,
                "ancestor_count": len(ancestors),
                "children": children,
                "child_count": len(children),
                "family_tree_count": len(family_members),
                "family_tree": family_members
            }

    def _remove_from_memory(self, asset_id: str):
        """Remove an asset from in-memory cache only."""
        if asset_id in self._assets:
            record = self._assets[asset_id]
            asset_key = _make_asset_key(record.filename, record.subfolder, record.folder_type)
            del self._assets[asset_id]
            if asset_key in self._asset_key_to_id:
                del self._asset_key_to_id[asset_key]

    def cleanup_expired(self) -> int:
        """Remove expired assets from memory and database."""
        with self._lock:
            now = datetime.now()
            now_iso = now.isoformat()

            # Clean memory
            expired_ids = [
                aid for aid, rec in self._assets.items()
                if rec.expires_at and now > rec.expires_at
            ]
            for aid in expired_ids:
                self._remove_from_memory(aid)

            # Clean SQLite
            deleted_count = 0
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM assets WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now_iso,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
            finally:
                if self.db_path != ":memory:":
                    conn.close()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired assets from database")

            return deleted_count

    def close(self):
        """Close shared connection if open."""
        with self._lock:
            if self._shared_mem_conn:
                try:
                    self._shared_mem_conn.close()
                except Exception:
                    pass
                self._shared_mem_conn = None

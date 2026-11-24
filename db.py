"""Persistent SQLite storage for SnapQC inspections (per-user & public demo isolation),
Active Learning YOLO labels, and server-side OAuth sessions."""

import csv
import io
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SnapQCStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with DB_LOCK:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS inspections (
                        id TEXT PRIMARY KEY,
                        report_no TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        owner_sub TEXT NOT NULL DEFAULT '',
                        owner_name TEXT NOT NULL DEFAULT '',
                        vessel_name TEXT NOT NULL DEFAULT '',
                        block_location TEXT NOT NULL DEFAULT '',
                        joint_no TEXT NOT NULL DEFAULT '',
                        wps_no TEXT NOT NULL DEFAULT '',
                        welder_id TEXT NOT NULL DEFAULT '',
                        inspector_name TEXT NOT NULL DEFAULT '',
                        standard_ref TEXT NOT NULL DEFAULT 'ISO 5817 Level B',
                        verdict TEXT NOT NULL DEFAULT 'HOLD',
                        notes TEXT NOT NULL DEFAULT '',
                        image_title TEXT NOT NULL DEFAULT '',
                        model_version TEXT NOT NULL DEFAULT '',
                        tile_mode INTEGER NOT NULL DEFAULT 0,
                        conf_threshold REAL NOT NULL DEFAULT 0.45,
                        detections_json TEXT NOT NULL DEFAULT '[]',
                        decisions_json TEXT NOT NULL DEFAULT '[]',
                        yolo_labels TEXT NOT NULL DEFAULT '',
                        approved_count INTEGER NOT NULL DEFAULT 0,
                        rejected_count INTEGER NOT NULL DEFAULT 0,
                        annotated_data_url TEXT NOT NULL DEFAULT ''
                    );

                    CREATE INDEX IF NOT EXISTS idx_inspections_created_at
                        ON inspections (created_at DESC);

                    CREATE TABLE IF NOT EXISTS oauth_sessions (
                        sid TEXT PRIMARY KEY,
                        sub TEXT NOT NULL,
                        name TEXT NOT NULL DEFAULT '',
                        email TEXT NOT NULL DEFAULT '',
                        roles_json TEXT NOT NULL DEFAULT '[]',
                        refresh_token TEXT,
                        exp REAL NOT NULL,
                        revoked INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL
                    );
                    """
                )
                # Idempotent column migration for existing v0.3.x SQLite databases
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)").fetchall()}
                if "owner_sub" not in cols:
                    conn.execute("ALTER TABLE inspections ADD COLUMN owner_sub TEXT NOT NULL DEFAULT ''")
                if "owner_name" not in cols:
                    conn.execute("ALTER TABLE inspections ADD COLUMN owner_name TEXT NOT NULL DEFAULT ''")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_inspections_owner_created ON inspections (owner_sub, created_at DESC)"
                )

    def create_inspection(self, payload: dict, owner_sub: str = "", owner_name: str = "") -> dict:
        insp_id = uuid.uuid4().hex[:12]
        created_at = utc_now_iso()
        date_compact = created_at[:10].replace("-", "")
        report_no = (payload.get("reportNo") or f"SQC-{date_compact}-{insp_id[:4].upper()}").strip()[:64]
        verdict = (payload.get("verdict") or "HOLD").strip().upper()
        if verdict not in {"ACCEPT", "REPAIR", "HOLD"}:
            verdict = "HOLD"

        detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
        decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
        approved_count = sum(1 for d in decisions if isinstance(d, dict) and d.get("status") == "approved")
        rejected_count = sum(1 for d in decisions if isinstance(d, dict) and d.get("status") == "rejected")

        annotated = str(payload.get("annotatedDataUrl") or "")
        if len(annotated) > 2_000_000 or (annotated and not annotated.startswith("data:image/")):
            annotated = ""

        row = {
            "id": insp_id,
            "report_no": report_no,
            "created_at": created_at,
            "owner_sub": str(owner_sub or "").strip()[:120],
            "owner_name": str(owner_name or "").strip()[:120],
            "vessel_name": str(payload.get("vesselName") or "").strip()[:120],
            "block_location": str(payload.get("blockLocation") or "").strip()[:120],
            "joint_no": str(payload.get("jointNo") or "").strip()[:80],
            "wps_no": str(payload.get("wpsNo") or "").strip()[:80],
            "welder_id": str(payload.get("welderId") or "").strip()[:80],
            "inspector_name": str(payload.get("inspectorName") or owner_name or "").strip()[:120],
            "standard_ref": str(payload.get("standardRef") or "ISO 5817 Level B").strip()[:80],
            "verdict": verdict,
            "notes": str(payload.get("notes") or "").strip()[:2000],
            "image_title": str(payload.get("imageTitle") or "").strip()[:200],
            "model_version": str(payload.get("modelVersion") or "").strip()[:80],
            "tile_mode": 1 if payload.get("tileMode") else 0,
            "conf_threshold": float(payload.get("confThreshold") or 0.45),
            "detections_json": json.dumps(detections),
            "decisions_json": json.dumps(decisions),
            "yolo_labels": str(payload.get("yoloLabels") or "")[:20000],
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "annotated_data_url": annotated,
        }

        with DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO inspections (
                        id, report_no, created_at, owner_sub, owner_name, vessel_name, block_location,
                        joint_no, wps_no, welder_id, inspector_name, standard_ref, verdict, notes,
                        image_title, model_version, tile_mode, conf_threshold,
                        detections_json, decisions_json, yolo_labels,
                        approved_count, rejected_count, annotated_data_url
                    ) VALUES (
                        :id, :report_no, :created_at, :owner_sub, :owner_name, :vessel_name, :block_location,
                        :joint_no, :wps_no, :welder_id, :inspector_name, :standard_ref, :verdict, :notes,
                        :image_title, :model_version, :tile_mode, :conf_threshold,
                        :detections_json, :decisions_json, :yolo_labels,
                        :approved_count, :rejected_count, :annotated_data_url
                    )
                    """,
                    row,
                )
        return self._row_to_dict(row, include_image=False)

    def _row_to_dict(self, r, include_image: bool = False) -> dict:
        out = {
            "id": r["id"],
            "reportNo": r["report_no"],
            "createdAt": r["created_at"],
            "ownerScope": "personal" if r["owner_sub"] else "public-demo",
            "ownerName": r["owner_name"],
            "vesselName": r["vessel_name"],
            "blockLocation": r["block_location"],
            "jointNo": r["joint_no"],
            "wpsNo": r["wps_no"],
            "welderId": r["welder_id"],
            "inspectorName": r["inspector_name"],
            "standardRef": r["standard_ref"],
            "verdict": r["verdict"],
            "notes": r["notes"],
            "imageTitle": r["image_title"],
            "modelVersion": r["model_version"],
            "tileMode": bool(r["tile_mode"]),
            "confThreshold": float(r["conf_threshold"]),
            "detections": json.loads(r["detections_json"] or "[]"),
            "decisions": json.loads(r["decisions_json"] or "[]"),
            "yoloLabels": r["yolo_labels"],
            "approvedCount": int(r["approved_count"]),
            "rejectedCount": int(r["rejected_count"]),
            "hasImage": bool(r["annotated_data_url"]),
        }
        if include_image:
            out["annotatedDataUrl"] = r["annotated_data_url"]
        return out

    def list_inspections(self, owner_sub: str = "", limit: int = 100) -> list[dict]:
        limit = max(1, min(200, int(limit)))
        owner_sub = str(owner_sub or "").strip()
        with DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM inspections WHERE owner_sub = ? ORDER BY created_at DESC LIMIT ?",
                    (owner_sub, limit),
                ).fetchall()
        return [self._row_to_dict(r, include_image=False) for r in rows]

    def get_inspection(self, insp_id: str, owner_sub: str = "") -> dict | None:
        owner_sub = str(owner_sub or "").strip()
        with DB_LOCK:
            with self._connect() as conn:
                r = conn.execute(
                    "SELECT * FROM inspections WHERE id = ? AND owner_sub = ?",
                    (insp_id, owner_sub),
                ).fetchone()
        return self._row_to_dict(r, include_image=True) if r else None

    def delete_inspection(self, insp_id: str, owner_sub: str = "") -> bool:
        owner_sub = str(owner_sub or "").strip()
        with DB_LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM inspections WHERE id = ? AND owner_sub = ?",
                    (insp_id, owner_sub),
                )
                return cur.rowcount > 0

    def export_csv(self, owner_sub: str = "") -> str:
        items = self.list_inspections(owner_sub=owner_sub, limit=200)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "Report No", "Created At (UTC)", "Vessel", "Block/Frame", "Joint No",
            "WPS", "Welder ID", "Standard", "Inspector", "Verdict",
            "Model", "Micro Tiling", "Total Detections", "Approved", "Rejected", "Notes",
        ])
        for it in items:
            writer.writerow([
                it["reportNo"], it["createdAt"], it["vesselName"], it["blockLocation"], it["jointNo"],
                it["wpsNo"], it["welderId"], it["standardRef"], it["inspectorName"], it["verdict"],
                it["modelVersion"], "Yes" if it["tileMode"] else "No",
                len(it["detections"]), it["approvedCount"], it["rejectedCount"], it["notes"],
            ])
        return buf.getvalue()

    def export_active_learning(self, owner_sub: str = "") -> dict:
        items = self.list_inspections(owner_sub=owner_sub, limit=200)
        samples = []
        total_boxes = 0
        for it in items:
            yolo = (it.get("yoloLabels") or "").strip()
            if not yolo or yolo.startswith("#"):
                continue
            lines = [ln for ln in yolo.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if not lines:
                continue
            total_boxes += len(lines)
            samples.append({
                "inspectionId": it["id"],
                "reportNo": it["reportNo"],
                "createdAt": it["createdAt"],
                "imageTitle": it["imageTitle"],
                "vesselName": it["vesselName"],
                "jointNo": it["jointNo"],
                "verdict": it["verdict"],
                "yoloLines": lines,
            })
        return {
            "exportedAt": utc_now_iso(),
            "classes": {"0": "Bad Weld", "1": "Good Weld", "2": "Defect"},
            "sampleCount": len(samples),
            "totalApprovedBoxes": total_boxes,
            "samples": samples,
        }

    def stats(self, owner_sub: str | None = None) -> dict:
        with DB_LOCK:
            with self._connect() as conn:
                if owner_sub is None:
                    r = conn.execute(
                        "SELECT COUNT(*) AS cnt, COALESCE(SUM(approved_count), 0) AS app FROM inspections"
                    ).fetchone()
                else:
                    r = conn.execute(
                        "SELECT COUNT(*) AS cnt, COALESCE(SUM(approved_count), 0) AS app FROM inspections WHERE owner_sub = ?",
                        (str(owner_sub or "").strip(),),
                    ).fetchone()
        return {"inspections": int(r["cnt"]), "approvedLabels": int(r["app"])}

    def put_session(self, sid: str, sess: dict):
        with DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO oauth_sessions
                        (sid, sub, name, email, roles_json, refresh_token, exp, revoked, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        sess["sub"],
                        sess.get("name") or "",
                        sess.get("email") or "",
                        json.dumps(sess.get("roles") or []),
                        sess.get("refresh_token"),
                        float(sess.get("exp") or (time.time() + 3600)),
                        1 if sess.get("revoked") else 0,
                        time.time(),
                    ),
                )

    def get_session(self, sid: str) -> dict | None:
        with DB_LOCK:
            with self._connect() as conn:
                r = conn.execute("SELECT * FROM oauth_sessions WHERE sid = ?", (sid,)).fetchone()
        if not r or r["revoked"] or float(r["exp"]) <= time.time():
            return None
        return {
            "sub": r["sub"],
            "name": r["name"],
            "email": r["email"],
            "roles": json.loads(r["roles_json"] or "[]"),
            "refresh_token": r["refresh_token"],
            "exp": float(r["exp"]),
            "revoked": bool(r["revoked"]),
        }

    def revoke_session(self, sid: str) -> str | None:
        with DB_LOCK:
            with self._connect() as conn:
                r = conn.execute("SELECT refresh_token FROM oauth_sessions WHERE sid = ?", (sid,)).fetchone()
                conn.execute("DELETE FROM oauth_sessions WHERE sid = ?", (sid,))
                return r["refresh_token"] if r else None

    def prune_sessions(self):
        with DB_LOCK:
            with self._connect() as conn:
                conn.execute("DELETE FROM oauth_sessions WHERE exp < ?", (time.time() - 3600,))

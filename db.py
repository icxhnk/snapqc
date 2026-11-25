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
    ALLOWED_CATEGORIES = {"Pengelasan", "Lambung & Pelat", "Perpipaan", "Kamar Mesin", "K3 & Keselamatan"}
    ALLOWED_URGENCIES = {"normal", "survey", "critical"}
    ALLOWED_OPINIONS = {"pass", "ndt", "rework"}
    ALLOWED_RECS = {"pass", "ndt", "rework", "tips"}
    ALLOWED_ROLES = {"Welder", "QC Inspector", "Fitter", "Marine Engineer", "Surveyor", "K3 / Safety"}

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.community_img_dir = self.db_path.parent / "community_images"
        self.community_img_dir.mkdir(parents=True, exist_ok=True)
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

                    CREATE TABLE IF NOT EXISTS app_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS community_posts (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        author_sub TEXT NOT NULL DEFAULT '',
                        author_token TEXT NOT NULL DEFAULT '',
                        author_name TEXT NOT NULL DEFAULT 'Anonim',
                        author_role TEXT NOT NULL DEFAULT 'Welder',
                        is_anonymous INTEGER NOT NULL DEFAULT 1,
                        is_verified INTEGER NOT NULL DEFAULT 0,
                        category TEXT NOT NULL DEFAULT 'Pengelasan',
                        urgency TEXT NOT NULL DEFAULT 'normal',
                        process_pos TEXT NOT NULL DEFAULT '',
                        material_spec TEXT NOT NULL DEFAULT '',
                        title TEXT NOT NULL,
                        question TEXT NOT NULL,
                        image_url TEXT NOT NULL DEFAULT '',
                        ai_summary TEXT NOT NULL DEFAULT '',
                        ai_detections_json TEXT NOT NULL DEFAULT '[]',
                        has_privacy_blur INTEGER NOT NULL DEFAULT 0,
                        accepted_reply_id TEXT NOT NULL DEFAULT '',
                        is_starter INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE INDEX IF NOT EXISTS idx_community_posts_created
                        ON community_posts (created_at DESC);

                    CREATE TABLE IF NOT EXISTS community_votes (
                        post_id TEXT NOT NULL,
                        voter_key TEXT NOT NULL,
                        opinion TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (post_id, voter_key),
                        FOREIGN KEY (post_id) REFERENCES community_posts(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS community_replies (
                        id TEXT PRIMARY KEY,
                        post_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        author_sub TEXT NOT NULL DEFAULT '',
                        author_token TEXT NOT NULL DEFAULT '',
                        author_name TEXT NOT NULL DEFAULT 'Rekan Maritim',
                        author_role TEXT NOT NULL DEFAULT 'QC Inspector',
                        is_verified INTEGER NOT NULL DEFAULT 0,
                        recommendation TEXT NOT NULL DEFAULT 'tips',
                        body TEXT NOT NULL,
                        helpful_count INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (post_id) REFERENCES community_posts(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_community_replies_post
                        ON community_replies (post_id, created_at ASC);

                    CREATE TABLE IF NOT EXISTS community_helpful (
                        reply_id TEXT NOT NULL,
                        voter_key TEXT NOT NULL,
                        PRIMARY KEY (reply_id, voter_key),
                        FOREIGN KEY (reply_id) REFERENCES community_replies(id) ON DELETE CASCADE
                    );
                    """
                )
                # Idempotent column migration for existing v0.3.x SQLite databases
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)").fetchall()}
                if "owner_sub" not in cols:
                    conn.execute("ALTER TABLE inspections ADD COLUMN owner_sub TEXT NOT NULL DEFAULT ''")
                if "owner_name" not in cols:
                    conn.execute("ALTER TABLE inspections ADD COLUMN owner_name TEXT NOT NULL DEFAULT ''")
                if "is_public" not in cols:
                    conn.execute("ALTER TABLE inspections ADD COLUMN is_public INTEGER NOT NULL DEFAULT 1")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_inspections_owner_created ON inspections (owner_sub, created_at DESC)"
                )
                self._seed_community_if_needed(conn)

    def _seed_community_if_needed(self, conn):
        seeded = conn.execute("SELECT value FROM app_meta WHERE key = 'community_seeded_v1'").fetchone()
        if seeded:
            return
        existing = conn.execute("SELECT COUNT(*) AS cnt FROM community_posts").fetchone()["cnt"]
        if existing > 0:
            conn.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES ('community_seeded_v1', '1')")
            return

        starter_posts = [
            {
                "id": "case01weldfc",
                "created_at": "2026-11-24T06:15:00Z",
                "author_name": "Anonim · Welder 6G",
                "author_role": "Welder",
                "is_anonymous": 1,
                "is_verified": 1,
                "category": "Pengelasan",
                "urgency": "survey",
                "process_pos": "FCAW · 3G Vertikal",
                "material_spec": "Baja Kapal Grade A · 12 mm",
                "title": "Profil las kasar & 2 titik porositas di fillet 3G — wajib gouging ulang sebelum survei BKI?",
                "question": "Rekan QC & senior welder, posisi kerja agak sempit di sudut stiffener jadi sudut torch sempat terlalu rebah. Scan AI menandai 1 jalur las kasar dan 2 titik cacat kecil. Apakah masih bisa lolos dengan gerinda dressing halus atau sebaiknya langsung gouging + re-weld sebelum surveyor datang?",
                "image_url": "/samples/test-04-pred.jpg",
                "ai_summary": "Perlu Periksa · 1 jalur las perlu dicek, 2 titik cacat kecil",
                "has_privacy_blur": 1,
                "accepted_reply_id": "rep01qcbki01",
                "votes": [("seed-1", "rework"), ("seed-2", "rework"), ("seed-3", "rework"), ("seed-4", "rework"), ("seed-5", "ndt"), ("seed-6", "ndt")],
                "replies": [
                    {
                        "id": "rep01qcbki01",
                        "created_at": "2026-11-24T06:42:00Z",
                        "author_name": "Hendra S.",
                        "author_role": "QC Inspector",
                        "is_verified": 1,
                        "recommendation": "rework",
                        "body": "Saran saya langsung gouging lokal di area yang berpori sampai ketemu logam bersih, lalu re-weld pakai kontrol ampere sedikit lebih stabil (turunkan ~10A saat weaving di sudut sempit) dan cek aliran gas CO2 (15–18 L/min) supaya tidak kena angin. Kalau cuma digerinda permukaan, lubang porositas di bawah permukaan sering ketahuan saat MPI/UT.",
                        "helpful_count": 7,
                    },
                    {
                        "id": "rep01surv02",
                        "created_at": "2026-11-24T07:10:00Z",
                        "author_name": "Capt. Dimas",
                        "author_role": "Surveyor",
                        "is_verified": 1,
                        "recommendation": "ndt",
                        "body": "Setuju dengan Pak Hendra. Setelah repair selesai dan dingin, lanjutkan MPI (Magnetic Particle Inspection) minimal 150 mm di sepanjang zona repair agar dokumentasi survei kelas langsung mulus tanpa pending.",
                        "helpful_count": 4,
                    },
                ],
            },
            {
                "id": "case02hull04",
                "created_at": "2026-11-24T09:30:00Z",
                "author_name": "Anonim · Fitter Lambung",
                "author_role": "Fitter",
                "is_anonymous": 1,
                "is_verified": 0,
                "category": "Lambung & Pelat",
                "urgency": "critical",
                "process_pos": "SMAW E7018 · 4G Overhead",
                "material_spec": "AH36 · 16 mm",
                "title": "Undercut terasa dalam di sambungan pelat bawah (overhead) — cukup capping tambahan?",
                "question": "Baru selesai pengelasan posisi 4G di area bottom shell. Pas diraba pakai kuku dan dicek visual ada lekukan undercut di tepi jalur las sekitar 1 mm. Apakah aman langsung ditimpa stringer bead tipis (E7018 2.6 mm) atau wajib dibersihkan sampai root?",
                "image_url": "/samples/test-03-pred.jpg",
                "ai_summary": "Perlu Periksa · 1 jalur las perlu dicek",
                "has_privacy_blur": 1,
                "accepted_reply_id": "",
                "votes": [("seed-1", "rework"), ("seed-2", "rework"), ("seed-3", "rework"), ("seed-4", "ndt")],
                "replies": [
                    {
                        "id": "rep02eng01",
                        "created_at": "2026-11-24T10:05:00Z",
                        "author_name": "Rudi Pratama",
                        "author_role": "Marine Engineer",
                        "is_verified": 1,
                        "recommendation": "rework",
                        "body": "Untuk pelat bawah AH36 (area tegangan tinggi & beban gelombang), undercut > 0,5 mm biasanya langsung ditolak standar ISO 5817 Level B / BKI. Ukur dulu pakai welding gauge; kalau memang mendekati 1 mm, jangan potong sampai root kecuali ada retak — cukup bersihkan slag dengan gerinda tipis, preheat ringan jika lembap, lalu isi stringer bead kecil E7018 2.6 mm kering (dari oven) dan blending halus.",
                        "helpful_count": 5,
                    },
                ],
            },
            {
                "id": "case03pipe06",
                "created_at": "2026-11-24T13:20:00Z",
                "author_name": "Anonim · Welder Pipa",
                "author_role": "Welder",
                "is_anonymous": 1,
                "is_verified": 0,
                "category": "Perpipaan",
                "urgency": "normal",
                "process_pos": "SMAW · 6G Pipa",
                "material_spec": "Carbon Steel Sch 40",
                "title": "2 titik kecil bekas slag di permukaan las pipa balas — cukup PT test setelah gerinda halus?",
                "question": "Di jalur capping pipa balas ada 2 bintik kecil mirip pin-hole dangkal. Sudah dicoba sikat kawat baja tapi masih terlihat samar. Mohon masukan rekan senior, apakah aman langsung uji Dye Penetrant (PT) setelah dipoles ringan?",
                "image_url": "/samples/test-06-pred.jpg",
                "ai_summary": "Perlu Periksa · 1 jalur las perlu dicek, 1 titik cacat kecil",
                "has_privacy_blur": 0,
                "accepted_reply_id": "",
                "votes": [("seed-1", "ndt"), ("seed-2", "ndt"), ("seed-3", "ndt"), ("seed-4", "pass")],
                "replies": [
                    {
                        "id": "rep03qc01",
                        "created_at": "2026-11-24T14:00:00Z",
                        "author_name": "Budi Santoso",
                        "author_role": "QC Inspector",
                        "is_verified": 1,
                        "recommendation": "ndt",
                        "body": "Lakukan gerinda blending ringan saja (jangan sampai mengurangi ketebalan dinding pipa di bawah nominal), bersihkan pakai cleaner, lalu lakukan Dye Penetrant Test (PT). Jika developer tidak mengeluarkan warna merah melebar, berarti hanya bekas slag permukaan dan aman lanjut hydrotest.",
                        "helpful_count": 3,
                    },
                ],
            },
        ]

        for p in starter_posts:
            conn.execute(
                """
                INSERT INTO community_posts (
                    id, created_at, author_sub, author_token, author_name, author_role,
                    is_anonymous, is_verified, category, urgency, process_pos, material_spec,
                    title, question, image_url, ai_summary, ai_detections_json,
                    has_privacy_blur, accepted_reply_id, is_starter
                ) VALUES (?, ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, 1)
                """,
                (
                    p["id"], p["created_at"], p["author_name"], p["author_role"],
                    p["is_anonymous"], p["is_verified"], p["category"], p["urgency"],
                    p["process_pos"], p["material_spec"], p["title"], p["question"],
                    p["image_url"], p["ai_summary"], p["has_privacy_blur"], p["accepted_reply_id"],
                ),
            )
            for vkey, op in p["votes"]:
                conn.execute(
                    "INSERT OR IGNORE INTO community_votes (post_id, voter_key, opinion, updated_at) VALUES (?, ?, ?, ?)",
                    (p["id"], vkey, op, p["created_at"]),
                )
            for rep in p["replies"]:
                conn.execute(
                    """
                    INSERT INTO community_replies (
                        id, post_id, created_at, author_sub, author_token, author_name,
                        author_role, is_verified, recommendation, body, helpful_count
                    ) VALUES (?, ?, ?, '', '', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rep["id"], p["id"], rep["created_at"], rep["author_name"],
                        rep["author_role"], rep["is_verified"], rep["recommendation"],
                        rep["body"], rep["helpful_count"],
                    ),
                )

        conn.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES ('community_seeded_v1', '1')")

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

        owner_clean = str(owner_sub or "").strip()[:120]
        # Signed-in users default to private unless they explicitly choose public; guests post to public demo feed
        if "isPublic" in payload:
            is_public = 1 if payload.get("isPublic") else 0
        else:
            is_public = 0 if owner_clean else 1

        row = {
            "id": insp_id,
            "report_no": report_no,
            "created_at": created_at,
            "owner_sub": owner_clean,
            "owner_name": str(owner_name or "").strip()[:120],
            "is_public": is_public,
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
                        id, report_no, created_at, owner_sub, owner_name, is_public, vessel_name, block_location,
                        joint_no, wps_no, welder_id, inspector_name, standard_ref, verdict, notes,
                        image_title, model_version, tile_mode, conf_threshold,
                        detections_json, decisions_json, yolo_labels,
                        approved_count, rejected_count, annotated_data_url
                    ) VALUES (
                        :id, :report_no, :created_at, :owner_sub, :owner_name, :is_public, :vessel_name, :block_location,
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
            "isPublic": bool(r["is_public"]) if "is_public" in r.keys() else (not bool(r["owner_sub"])),
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

    def list_public_feed(self, limit: int = 30) -> list[dict]:
        limit = max(1, min(60, int(limit)))
        with DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM inspections WHERE is_public = 1 ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(r, include_image=True) for r in rows]

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

    # ---------------- Maritime Peer-Help Community ("Second Opinion") ----------------

    def _can_manage(self, post_row, viewer_sub: str, viewer_token: str) -> bool:
        if post_row["is_starter"]:
            return False
        if post_row["author_sub"] and viewer_sub and post_row["author_sub"] == viewer_sub:
            return True
        if not post_row["author_sub"] and post_row["author_token"] and viewer_token and post_row["author_token"] == viewer_token:
            return True
        return False

    def _hydrate_post(self, conn, post_row, voter_key: str, viewer_sub: str, viewer_token: str) -> dict:
        pid = post_row["id"]
        vote_rows = conn.execute(
            "SELECT opinion, COUNT(*) AS cnt FROM community_votes WHERE post_id = ? GROUP BY opinion",
            (pid,),
        ).fetchall()
        votes = {"pass": 0, "ndt": 0, "rework": 0}
        for vr in vote_rows:
            if vr["opinion"] in votes:
                votes[vr["opinion"]] = int(vr["cnt"])
        votes["total"] = votes["pass"] + votes["ndt"] + votes["rework"]

        my_vote_row = None
        if voter_key:
            my_vote_row = conn.execute(
                "SELECT opinion FROM community_votes WHERE post_id = ? AND voter_key = ?",
                (pid, voter_key),
            ).fetchone()
        my_vote = my_vote_row["opinion"] if my_vote_row else None

        rep_rows = conn.execute(
            "SELECT * FROM community_replies WHERE post_id = ? ORDER BY created_at ASC",
            (pid,),
        ).fetchall()
        my_helpful_ids = set()
        if voter_key and rep_rows:
            rep_ids = [r["id"] for r in rep_rows]
            placeholders = ",".join("?" for _ in rep_ids)
            h_rows = conn.execute(
                f"SELECT reply_id FROM community_helpful WHERE voter_key = ? AND reply_id IN ({placeholders})",
                [voter_key, *rep_ids],
            ).fetchall()
            my_helpful_ids = {r["reply_id"] for r in h_rows}

        accepted_id = post_row["accepted_reply_id"] or ""
        replies = []
        for rr in rep_rows:
            replies.append({
                "id": rr["id"],
                "postId": rr["post_id"],
                "createdAt": rr["created_at"],
                "authorName": rr["author_name"],
                "authorRole": rr["author_role"],
                "isVerified": bool(rr["is_verified"]),
                "recommendation": rr["recommendation"],
                "body": rr["body"],
                "helpfulCount": int(rr["helpful_count"]),
                "myHelpful": rr["id"] in my_helpful_ids,
                "isAccepted": bool(accepted_id and rr["id"] == accepted_id),
            })
        # Put accepted solution first, then sort by helpful count desc, then oldest first
        replies.sort(key=lambda x: (0 if x["isAccepted"] else 1, -x["helpfulCount"], x["createdAt"]))

        return {
            "id": pid,
            "createdAt": post_row["created_at"],
            "authorName": post_row["author_name"],
            "authorRole": post_row["author_role"],
            "isAnonymous": bool(post_row["is_anonymous"]),
            "isVerified": bool(post_row["is_verified"]),
            "category": post_row["category"],
            "urgency": post_row["urgency"],
            "processPos": post_row["process_pos"],
            "materialSpec": post_row["material_spec"],
            "title": post_row["title"],
            "question": post_row["question"],
            "imageUrl": post_row["image_url"],
            "aiSummary": post_row["ai_summary"],
            "aiDetections": json.loads(post_row["ai_detections_json"] or "[]"),
            "hasPrivacyBlur": bool(post_row["has_privacy_blur"]),
            "acceptedReplyId": accepted_id,
            "isSolved": bool(accepted_id),
            "isStarter": bool(post_row["is_starter"]),
            "canManage": self._can_manage(post_row, viewer_sub, viewer_token),
            "votes": votes,
            "myVote": my_vote,
            "replyCount": len(replies),
            "replies": replies,
        }

    def list_community_posts(
        self,
        *,
        category: str = "",
        urgency: str = "",
        status: str = "",
        query: str = "",
        limit: int = 40,
        voter_key: str = "",
        viewer_sub: str = "",
        viewer_token: str = "",
    ) -> dict:
        limit = max(1, min(80, int(limit)))
        clauses = []
        params: list = []

        cat = str(category or "").strip()
        if cat and cat in self.ALLOWED_CATEGORIES:
            clauses.append("category = ?")
            params.append(cat)

        urg = str(urgency or "").strip().lower()
        if urg and urg in self.ALLOWED_URGENCIES:
            clauses.append("urgency = ?")
            params.append(urg)

        st = str(status or "").strip().lower()
        if st == "solved":
            clauses.append("accepted_reply_id != ''")
        elif st == "open":
            clauses.append("accepted_reply_id = ''")
        elif st == "critical":
            clauses.append("urgency = 'critical'")

        q = str(query or "").strip().lower()
        if q:
            like = f"%{q}%"
            clauses.append("(LOWER(title) LIKE ? OR LOWER(question) LIKE ? OR LOWER(process_pos) LIKE ? OR LOWER(material_spec) LIKE ? OR LOWER(category) LIKE ?)")
            params.extend([like, like, like, like, like])

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT * FROM community_posts {where_sql} ORDER BY created_at DESC LIMIT ?",
                    [*params, limit],
                ).fetchall()
                items = [self._hydrate_post(conn, r, voter_key, viewer_sub, viewer_token) for r in rows]
                s_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_posts,
                        SUM(CASE WHEN accepted_reply_id != '' THEN 1 ELSE 0 END) AS solved_posts,
                        SUM(CASE WHEN accepted_reply_id = '' THEN 1 ELSE 0 END) AS open_posts,
                        (SELECT COUNT(*) FROM community_replies) AS total_replies
                    FROM community_posts
                    """
                ).fetchone()
        return {
            "items": items,
            "stats": {
                "totalPosts": int(s_row["total_posts"] or 0),
                "solvedPosts": int(s_row["solved_posts"] or 0),
                "openPosts": int(s_row["open_posts"] or 0),
                "totalReplies": int(s_row["total_replies"] or 0),
            },
        }

    def create_community_post(
        self,
        payload: dict,
        *,
        image_bytes: bytes | None = None,
        author_sub: str = "",
        author_token: str = "",
        sso_name: str = "",
        voter_key: str = "",
    ) -> dict:
        title = str(payload.get("title") or "").strip()
        question = str(payload.get("question") or "").strip()
        if len(title) < 6:
            raise ValueError("Judul kasus minimal 6 karakter.")
        if len(question) < 10:
            raise ValueError("Jelaskan pertanyaan atau kondisi lapangan minimal 10 karakter.")

        category = str(payload.get("category") or "Pengelasan").strip()
        if category not in self.ALLOWED_CATEGORIES:
            category = "Pengelasan"

        urgency = str(payload.get("urgency") or "normal").strip().lower()
        if urgency not in self.ALLOWED_URGENCIES:
            urgency = "normal"

        role = str(payload.get("authorRole") or "Welder").strip()
        if role not in self.ALLOWED_ROLES:
            role = "Welder"

        is_anonymous = bool(payload.get("isAnonymous", True))
        custom_name = str(payload.get("authorName") or "").strip()[:80]
        if is_anonymous:
            author_name = f"Anonim · {role}"
        else:
            author_name = custom_name or str(sso_name or "").strip()[:80] or f"Rekan · {role}"

        post_id = uuid.uuid4().hex[:12]
        created_at = utc_now_iso()
        image_url = ""
        if image_bytes:
            self.community_img_dir.mkdir(parents=True, exist_ok=True)
            img_file = self.community_img_dir / f"{post_id}.jpg"
            img_file.write_bytes(image_bytes)
            image_url = f"/api/community/images/{post_id}.jpg"
        else:
            raw_sample = str(payload.get("sampleImageUrl") or "").strip()
            if raw_sample.startswith("/samples/") and ".." not in raw_sample:
                image_url = raw_sample[:120]

        ai_detections = payload.get("aiDetections") if isinstance(payload.get("aiDetections"), list) else []

        with DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO community_posts (
                        id, created_at, author_sub, author_token, author_name, author_role,
                        is_anonymous, is_verified, category, urgency, process_pos, material_spec,
                        title, question, image_url, ai_summary, ai_detections_json,
                        has_privacy_blur, accepted_reply_id, is_starter
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0)
                    """,
                    (
                        post_id,
                        created_at,
                        str(author_sub or "").strip()[:120],
                        str(author_token or "").strip()[:120],
                        author_name,
                        role,
                        1 if is_anonymous else 0,
                        1 if author_sub else 0,
                        category,
                        urgency,
                        str(payload.get("processPos") or "").strip()[:80],
                        str(payload.get("materialSpec") or "").strip()[:80],
                        title[:140],
                        question[:2000],
                        image_url,
                        str(payload.get("aiSummary") or "").strip()[:240],
                        json.dumps(ai_detections[:40]),
                        1 if payload.get("hasPrivacyBlur") else 0,
                    ),
                )
                row = conn.execute("SELECT * FROM community_posts WHERE id = ?", (post_id,)).fetchone()
                return self._hydrate_post(conn, row, voter_key, author_sub, author_token)

    def vote_community_post(self, post_id: str, voter_key: str, opinion: str) -> dict | None:
        op = str(opinion or "").strip().lower()
        if op not in self.ALLOWED_OPINIONS:
            raise ValueError("Pilihan pendapat harus pass, ndt, atau rework.")
        if not voter_key:
            raise ValueError("Identitas voter tidak valid.")

        with DB_LOCK:
            with self._connect() as conn:
                exists = conn.execute("SELECT id FROM community_posts WHERE id = ?", (post_id,)).fetchone()
                if not exists:
                    return None
                prev = conn.execute(
                    "SELECT opinion FROM community_votes WHERE post_id = ? AND voter_key = ?",
                    (post_id, voter_key),
                ).fetchone()
                if prev and prev["opinion"] == op:
                    conn.execute("DELETE FROM community_votes WHERE post_id = ? AND voter_key = ?", (post_id, voter_key))
                    my_vote = None
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO community_votes (post_id, voter_key, opinion, updated_at) VALUES (?, ?, ?, ?)",
                        (post_id, voter_key, op, utc_now_iso()),
                    )
                    my_vote = op

                vote_rows = conn.execute(
                    "SELECT opinion, COUNT(*) AS cnt FROM community_votes WHERE post_id = ? GROUP BY opinion",
                    (post_id,),
                ).fetchall()
                votes = {"pass": 0, "ndt": 0, "rework": 0}
                for vr in vote_rows:
                    if vr["opinion"] in votes:
                        votes[vr["opinion"]] = int(vr["cnt"])
                votes["total"] = votes["pass"] + votes["ndt"] + votes["rework"]
                return {"postId": post_id, "votes": votes, "myVote": my_vote}

    def add_community_reply(
        self,
        post_id: str,
        payload: dict,
        *,
        author_sub: str = "",
        author_token: str = "",
        sso_name: str = "",
    ) -> dict | None:
        body = str(payload.get("body") or "").strip()
        if len(body) < 4:
            raise ValueError("Saran atau komentar minimal 4 karakter.")

        role = str(payload.get("authorRole") or "QC Inspector").strip()
        if role not in self.ALLOWED_ROLES:
            role = "QC Inspector"

        rec = str(payload.get("recommendation") or "tips").strip().lower()
        if rec not in self.ALLOWED_RECS:
            rec = "tips"

        custom_name = str(payload.get("authorName") or "").strip()[:80]
        author_name = custom_name or str(sso_name or "").strip()[:80] or f"Rekan · {role}"

        reply_id = uuid.uuid4().hex[:12]
        created_at = utc_now_iso()

        with DB_LOCK:
            with self._connect() as conn:
                exists = conn.execute("SELECT id FROM community_posts WHERE id = ?", (post_id,)).fetchone()
                if not exists:
                    return None
                conn.execute(
                    """
                    INSERT INTO community_replies (
                        id, post_id, created_at, author_sub, author_token, author_name,
                        author_role, is_verified, recommendation, body, helpful_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        reply_id,
                        post_id,
                        created_at,
                        str(author_sub or "").strip()[:120],
                        str(author_token or "").strip()[:120],
                        author_name,
                        role,
                        1 if author_sub else 0,
                        rec,
                        body[:2000],
                    ),
                )
        return {
            "id": reply_id,
            "postId": post_id,
            "createdAt": created_at,
            "authorName": author_name,
            "authorRole": role,
            "isVerified": bool(author_sub),
            "recommendation": rec,
            "body": body[:2000],
            "helpfulCount": 0,
            "myHelpful": False,
            "isAccepted": False,
        }

    def accept_community_reply(self, post_id: str, reply_id: str, viewer_sub: str, viewer_token: str) -> dict | None:
        with DB_LOCK:
            with self._connect() as conn:
                post_row = conn.execute("SELECT * FROM community_posts WHERE id = ?", (post_id,)).fetchone()
                if not post_row:
                    return None
                if not self._can_manage(post_row, viewer_sub, viewer_token):
                    raise PermissionError("Hanya pembuat kasus yang dapat menandai solusi terbaik.")
                if reply_id:
                    rep = conn.execute(
                        "SELECT id FROM community_replies WHERE id = ? AND post_id = ?",
                        (reply_id, post_id),
                    ).fetchone()
                    if not rep:
                        raise ValueError("Komentar tidak ditemukan pada kasus ini.")
                new_val = "" if post_row["accepted_reply_id"] == reply_id else reply_id
                conn.execute("UPDATE community_posts SET accepted_reply_id = ? WHERE id = ?", (new_val, post_id))
                return {"postId": post_id, "acceptedReplyId": new_val, "isSolved": bool(new_val)}

    def toggle_reply_helpful(self, reply_id: str, voter_key: str) -> dict | None:
        if not voter_key:
            raise ValueError("Identitas voter tidak valid.")
        with DB_LOCK:
            with self._connect() as conn:
                rep = conn.execute("SELECT id, helpful_count FROM community_replies WHERE id = ?", (reply_id,)).fetchone()
                if not rep:
                    return None
                prev = conn.execute(
                    "SELECT 1 FROM community_helpful WHERE reply_id = ? AND voter_key = ?",
                    (reply_id, voter_key),
                ).fetchone()
                if prev:
                    conn.execute("DELETE FROM community_helpful WHERE reply_id = ? AND voter_key = ?", (reply_id, voter_key))
                    conn.execute("UPDATE community_replies SET helpful_count = MAX(0, helpful_count - 1) WHERE id = ?", (reply_id,))
                    my_helpful = False
                else:
                    conn.execute("INSERT INTO community_helpful (reply_id, voter_key) VALUES (?, ?)", (reply_id, voter_key))
                    conn.execute("UPDATE community_replies SET helpful_count = helpful_count + 1 WHERE id = ?", (reply_id,))
                    my_helpful = True
                updated = conn.execute("SELECT helpful_count FROM community_replies WHERE id = ?", (reply_id,)).fetchone()
                return {"replyId": reply_id, "helpfulCount": int(updated["helpful_count"]), "myHelpful": my_helpful}

    def delete_community_post(self, post_id: str, viewer_sub: str, viewer_token: str) -> bool:
        with DB_LOCK:
            with self._connect() as conn:
                post_row = conn.execute("SELECT * FROM community_posts WHERE id = ?", (post_id,)).fetchone()
                if not post_row:
                    return False
                if not self._can_manage(post_row, viewer_sub, viewer_token):
                    raise PermissionError("Hanya pembuat kasus yang dapat menghapus postingan ini.")
                conn.execute("DELETE FROM community_posts WHERE id = ?", (post_id,))
        img_path = self.community_img_dir / f"{post_id}.jpg"
        try:
            if img_path.exists():
                img_path.unlink()
        except Exception:
            pass
        return True

    def get_community_image_path(self, post_id: str) -> Path | None:
        clean = str(post_id or "").strip()
        if not clean or not clean.isalnum() or len(clean) > 24:
            return None
        path = (self.community_img_dir / f"{clean}.jpg").resolve()
        base = self.community_img_dir.resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return None
        return path

    def community_stats(self) -> dict:
        with DB_LOCK:
            with self._connect() as conn:
                r = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS posts,
                        (SELECT COUNT(*) FROM community_replies) AS replies
                    FROM community_posts
                    """
                ).fetchone()
        return {"posts": int(r["posts"] or 0), "replies": int(r["replies"] or 0)}

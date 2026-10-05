"""
AgroIntelli — Database & Storage Module
Handles MongoDB Atlas connection, user authentication, and temporal leaf record persistence
with an automatic local JSON fallback for zero-downtime offline execution.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Auto-load .env if available
for candidate_env in [Path(__file__).parent.parent / ".env", Path(__file__).parent / ".env"]:
    if candidate_env.exists():
        try:
            with open(candidate_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("ℹ️ pymongo library not installed yet. Operating in local storage mode.")

ATLAS_DEFAULT = "mongodb+srv://rahul90930kr_db_user:odDmDKkoa1Vqa17p@cluster0.c5skeva.mongodb.net/?appName=Cluster0"
MONGO_URI = os.environ.get("MONGO_URI", ATLAS_DEFAULT)
DB_NAME   = os.environ.get("MONGO_DB_NAME", "agrointelli")


class StorageManager:
    """
    Manages MongoDB persistence for user profiles and leaf health journals.
    Features an automatic fallback to local JSON storage if MongoDB server is offline,
    ensuring zero-downtime execution while setup is in progress.
    """
    def __init__(self):
        self.client = None
        self.db = None
        self.is_mongo = False
        self.fallback_file = Path(__file__).parent / "data_store.json"
        self._init_connection()

    def _init_connection(self):
        if PYMONGO_AVAILABLE:
            try:
                # 2 second server selection timeout so startup never hangs
                self.client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
                self.client.admin.command('ping')
                self.db = self.client[DB_NAME]
                self.is_mongo = True
                self.db.users.create_index("username", unique=True)
                self.db.leaf_records.create_index("record_id", unique=True)
                self.db.leaf_records.create_index("user_id")
                print(f"🍃 Connected to MongoDB ({DB_NAME}) at {MONGO_URI}")
                return
            except Exception as e:
                print(f"ℹ️ MongoDB offline ({e}). Using local fallback store at data_store.json")
        self.is_mongo = False
        self._init_fallback_file()

    def _init_fallback_file(self):
        if not self.fallback_file.exists():
            with open(self.fallback_file, "w", encoding="utf-8") as f:
                json.dump({"users": {}, "records": {}}, f, indent=2)

    def _load_fallback(self):
        self._init_fallback_file()
        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}, "records": {}}

    def _save_fallback(self, data):
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── User Operations ──
    def get_user(self, username_or_email):
        target = username_or_email.strip().lower()
        if self.is_mongo:
            doc = self.db.users.find_one({
                "$or": [{"username_lower": target}, {"email_lower": target}]
            })
            if doc and "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        else:
            data = self._load_fallback()
            for u in data.get("users", {}).values():
                if u.get("username_lower") == target or u.get("email_lower") == target:
                    return u
            return None

    def create_user(self, username, password, email=""):
        uname_lower = username.strip().lower()
        email_lower = email.strip().lower() if email else ""

        if self.get_user(uname_lower):
            return None, "Username already taken."
        if email_lower and self.get_user(email_lower):
            return None, "Email already registered."

        user_id = f"user_{username}_{int(datetime.now().timestamp())}"
        user_doc = {
            "user_id": user_id,
            "username": username,
            "username_lower": uname_lower,
            "email": email,
            "email_lower": email_lower,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now().isoformat()
        }

        if self.is_mongo:
            self.db.users.insert_one(user_doc.copy())
        else:
            data = self._load_fallback()
            data["users"][user_id] = user_doc
            self._save_fallback(data)

        # Return clean user profile without password hash
        clean = {k: v for k, v in user_doc.items() if k not in ("password_hash", "_id")}
        return clean, None

    # ── Leaf Record Operations ──
    def save_leaf_record(self, record):
        record["updated_at"] = datetime.now().isoformat()
        if self.is_mongo:
            try:
                # Exclude _id to prevent immutable field modification errors on update
                doc_to_save = {k: v for k, v in record.items() if k != "_id"}
                self.db.leaf_records.update_one(
                    {"record_id": record["record_id"]},
                    {"$set": doc_to_save},
                    upsert=True
                )
                print(f"🍃 [MongoDB] Successfully saved leaf record: {record['record_id']} ({record.get('plant_name')})")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ [MongoDB] Save failed: {e}. Writing to fallback storage.")
                sys.stdout.flush()
                data = self._load_fallback()
                data["records"][record["record_id"]] = record
                self._save_fallback(data)
        else:
            data = self._load_fallback()
            data["records"][record["record_id"]] = record
            self._save_fallback(data)
            print(f"📁 [Local JSON] Saved leaf record: {record['record_id']}")
            sys.stdout.flush()
        return record

    def get_leaf_record(self, record_id):
        if self.is_mongo:
            doc = self.db.leaf_records.find_one({"record_id": record_id})
            if doc and "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        else:
            data = self._load_fallback()
            return data.get("records", {}).get(record_id)

    def get_user_records(self, user_id):
        if not user_id:
            user_id = "guest"
        if self.is_mongo:
            # Strictly isolate: only return records matching the requested user_id
            cursor = self.db.leaf_records.find({"user_id": user_id}).sort("updated_at", -1)
            docs = []
            for d in cursor:
                d["_id"] = str(d["_id"])
                docs.append(d)
            return docs
        else:
            data = self._load_fallback()
            items = [r for r in data.get("records", {}).values() if r.get("user_id") == user_id]
            items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return items

    def delete_record(self, record_id, user_id=None):
        if not record_id:
            return False
        query = {"record_id": record_id}
        if user_id:
            query["user_id"] = user_id
        if self.is_mongo:
            # 1. Match directly by unique record_id (scoped to user_id if provided)
            res = self.db.leaf_records.delete_one(query)
            if res.deleted_count > 0:
                return True
            # 2. Match by MongoDB ObjectId if passed
            try:
                from bson import ObjectId
                if ObjectId.is_valid(record_id):
                    del_q = {"_id": ObjectId(record_id)}
                    if user_id:
                        del_q["user_id"] = user_id
                    res = self.db.leaf_records.delete_one(del_q)
                    if res.deleted_count > 0:
                        return True
            except Exception:
                pass
            return False
        else:
            data = self._load_fallback()
            records = data.get("records", {})
            if record_id in records:
                if not user_id or records[record_id].get("user_id") == user_id:
                    del records[record_id]
                    self._save_fallback(data)
                    return True
            return False


db_store = StorageManager()

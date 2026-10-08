"""
AgroIntelli — Database & Storage Module
Handles MongoDB Atlas connection, user authentication, and temporal leaf record persistence
with an automatic local JSON fallback for zero-downtime offline execution.
"""

import os
import sys
import json
import uuid
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
                self.db.community_posts.create_index("post_id", unique=True)
                self.db.community_posts.create_index("plant_id")
                self.db.community_comments.create_index("comment_id", unique=True)
                self.db.community_comments.create_index("post_id")
                self.db.community_votes.create_index([("post_id", 1), ("user_id", 1)], unique=True)
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
            if doc:
                doc.setdefault("last_selected_plant", "tomato")
            return doc
        else:
            data = self._load_fallback()
            for u in data.get("users", {}).values():
                if u.get("username_lower") == target or u.get("email_lower") == target:
                    u.setdefault("last_selected_plant", "tomato")
                    return u
            return None

    def create_user(self, username, password, email="", last_selected_plant="tomato"):
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
            "last_selected_plant": (last_selected_plant or "tomato").lower(),
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

    # ── User Preferences (Last Selected Plant) ──
    def update_user_preference(self, user_id, last_selected_plant):
        if not user_id or not last_selected_plant:
            return None
        plant_id = str(last_selected_plant).strip().lower()
        if self.is_mongo:
            try:
                self.db.users.update_one(
                    {"$or": [{"user_id": user_id}, {"username_lower": str(user_id).lower()}]},
                    {"$set": {"last_selected_plant": plant_id}}
                )
                doc = self.db.users.find_one({"$or": [{"user_id": user_id}, {"username_lower": str(user_id).lower()}]})
                if doc and "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                return {k: v for k, v in doc.items() if k != "password_hash"} if doc else None
            except Exception as e:
                print(f"⚠️ [MongoDB] update_user_preference error: {e}")
        data = self._load_fallback()
        for u in data.get("users", {}).values():
            if u.get("user_id") == user_id or u.get("username_lower") == str(user_id).lower():
                u["last_selected_plant"] = plant_id
                self._save_fallback(data)
                return {k: v for k, v in u.items() if k != "password_hash"}
        return None

    # ── Plant Health Community Operations ──
    def get_community_posts(self, plant=None, disease=None):
        """Retrieve community posts filtered by plant and optionally disease."""
        posts = []
        p_filter = str(plant).strip().lower() if plant and str(plant).strip().lower() not in ("all", "all plants", "none", "") else None
        d_filter = str(disease).strip().lower() if disease and str(disease).strip().lower() not in ("all", "none", "") else None

        if self.is_mongo:
            try:
                query = {}
                if p_filter:
                    query["plant_id"] = p_filter
                if d_filter:
                    query["disease_id"] = d_filter
                cursor = self.db.community_posts.find(query).sort("created_at", -1)
                for doc in cursor:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    posts.append(doc)
                if posts:
                    return posts
            except Exception as e:
                print(f"⚠️ [MongoDB] get_community_posts error: {e}")

        data = self._load_fallback()
        all_posts = list(data.get("posts", {}).values())
        if not all_posts:
            # Seed posts if empty
            self._seed_community_posts()
            data = self._load_fallback()
            all_posts = list(data.get("posts", {}).values())

        filtered = []
        for p in all_posts:
            if p_filter and p.get("plant_id", "").lower() != p_filter:
                continue
            if d_filter and p.get("disease_id", "").lower() != d_filter:
                continue
            filtered.append(p)

        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return filtered

    def save_community_post(self, post):
        """Saves a new user or official community post."""
        post_id = post.get("post_id") or f"post_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"
        post["post_id"] = post_id
        post.setdefault("created_at", datetime.now().isoformat())
        post.setdefault("upvotes", 0)
        post.setdefault("downvotes", 0)
        post.setdefault("comment_count", 0)
        post.setdefault("comments", [])
        post.setdefault("voted_users", {})

        if self.is_mongo:
            try:
                doc = {k: v for k, v in post.items() if k != "_id"}
                self.db.community_posts.update_one({"post_id": post_id}, {"$set": doc}, upsert=True)
                return post
            except Exception as e:
                print(f"⚠️ [MongoDB] save_community_post error: {e}")

        data = self._load_fallback()
        data.setdefault("posts", {})
        data["posts"][post_id] = post
        self._save_fallback(data)
        return post

    def vote_community_post(self, post_id, vote_type="up", user_id="guest"):
        """Upvote or downvote a post with single-vote idempotency per user."""
        post = None
        if self.is_mongo:
            try:
                post = self.db.community_posts.find_one({"post_id": post_id})
                if post and "_id" in post:
                    post["_id"] = str(post["_id"])
            except Exception:
                pass
        if not post:
            data = self._load_fallback()
            post = data.get("posts", {}).get(post_id)

        if not post:
            return None

        voted_users = post.get("voted_users", {})
        prev_vote = voted_users.get(user_id)
        upvotes = int(post.get("upvotes", 0))
        downvotes = int(post.get("downvotes", 0))

        if vote_type == "up":
            if prev_vote == "up":
                upvotes = max(0, upvotes - 1)
                del voted_users[user_id]
            else:
                if prev_vote == "down":
                    downvotes = max(0, downvotes - 1)
                upvotes += 1
                voted_users[user_id] = "up"
        elif vote_type == "down":
            if prev_vote == "down":
                downvotes = max(0, downvotes - 1)
                del voted_users[user_id]
            else:
                if prev_vote == "up":
                    upvotes = max(0, upvotes - 1)
                downvotes += 1
                voted_users[user_id] = "down"

        post["upvotes"] = upvotes
        post["downvotes"] = downvotes
        post["voted_users"] = voted_users

        self.save_community_post(post)

        # Decoupled entity persistence for SQL / analytics migration
        if self.is_mongo:
            try:
                if user_id in voted_users:
                    self.db.community_votes.update_one(
                        {"post_id": post_id, "user_id": str(user_id)},
                        {"$set": {
                            "post_id": post_id,
                            "user_id": str(user_id),
                            "vote_type": voted_users[user_id],
                            "updated_at": datetime.now().isoformat()
                        }},
                        upsert=True
                    )
                else:
                    self.db.community_votes.delete_one({"post_id": post_id, "user_id": str(user_id)})
            except Exception as e:
                print(f"⚠️ [MongoDB] community_votes notice: {e}")

        return {
            "post_id": post_id,
            "upvotes": upvotes,
            "downvotes": downvotes,
            "user_vote": voted_users.get(user_id)
        }

    def get_post_comments(self, post_id):
        """Returns comments for a post, querying normalized community_comments when available."""
        if not post_id:
            return []
        p_str = str(post_id).strip()
        if self.is_mongo:
            try:
                # Query decoupled community_comments collection first
                cursor = self.db.community_comments.find({"post_id": p_str}).sort("created_at", 1)
                comments_list = []
                for doc in cursor:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    comments_list.append(doc)
                if comments_list:
                    return comments_list

                # Fallback to embedded array on post document
                from bson import ObjectId
                q = {"$or": [{"post_id": p_str}, {"_id": p_str}]}
                if ObjectId.is_valid(p_str):
                    q["$or"].append({"_id": ObjectId(p_str)})
                p_doc = self.db.community_posts.find_one(q)
                if p_doc and "comments" in p_doc:
                    return p_doc["comments"]
            except Exception as e:
                print(f"⚠️ [MongoDB] get_post_comments notice: {e}")

        posts = self.get_community_posts()
        for p in posts:
            if p.get("post_id") == p_str or str(p.get("_id")) == p_str or str(p.get("id")) == p_str:
                return p.get("comments", [])
        return []

    def add_post_comment(self, post_id, comment):
        """Adds a comment to a community post, writing to both decoupled collection and post record."""
        if not post_id:
            return None
        p_str = str(post_id).strip()
        comment_id = comment.get("comment_id") or f"c_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:4]}"
        comment["comment_id"] = comment_id
        comment["post_id"] = p_str
        comment.setdefault("user_id", "guest")
        comment.setdefault("author", "Grower")
        comment.setdefault("author_badge", "Grower")
        comment.setdefault("created_at", datetime.now().isoformat())

        post = None
        if self.is_mongo:
            try:
                from bson import ObjectId
                q = {"$or": [{"post_id": p_str}, {"_id": p_str}]}
                if ObjectId.is_valid(p_str):
                    q["$or"].append({"_id": ObjectId(p_str)})
                post = self.db.community_posts.find_one(q)
            except Exception:
                pass
        if not post:
            data = self._load_fallback()
            post = data.get("posts", {}).get(p_str)
            if not post:
                for p in data.get("posts", {}).values():
                    if p.get("post_id") == p_str or str(p.get("_id")) == p_str or str(p.get("id")) == p_str:
                        post = p
                        break

        if not post:
            return None

        # 1. Decoupled collection persistence (SQL-ready relational record)
        if self.is_mongo:
            try:
                c_doc = {k: v for k, v in comment.items() if k != "_id"}
                self.db.community_comments.update_one({"comment_id": comment_id}, {"$set": c_doc}, upsert=True)
            except Exception as e:
                print(f"⚠️ [MongoDB] decoupled comment insert notice: {e}")

        # 2. Aggregated updates on post
        comments = post.get("comments", [])
        comments.append(comment)
        post["comments"] = comments
        post["comment_count"] = len(comments)

        self.save_community_post(post)
        return comment

    def _seed_community_posts(self):
        """Seed realistic agronomic community posts for major crops and diseases."""
        seed_posts = [
            {
                "post_id": "seed_post_tomato_early_blight",
                "author": "Rahul Kumar",
                "author_badge": "Verified Grower",
                "plant_id": "tomato",
                "plant_name": "Tomato",
                "disease_id": "tomato_early_blight",
                "disease_name": "Early Blight",
                "content": "Noticed classic concentric target rings forming on my lower tomato leaves after 3 consecutive days of morning rain. Pruned the bottom 12 inches of foliage and applied a copper octanoate spray. The new canopy growth is completely clean!",
                "upvotes": 42,
                "downvotes": 2,
                "comment_count": 3,
                "created_at": "2026-10-06T14:30:00",
                "comments": [
                    {
                        "comment_id": "c_1",
                        "author": "Dr. Sarah Jenkins (Agronomist)",
                        "content": "Excellent intervention, Rahul! Removing lower foliage prevents soil-splash dispersal of Alternaria solani spores.",
                        "created_at": "2026-10-06T16:15:00"
                    },
                    {
                        "comment_id": "c_2",
                        "author": "Marcus Lee",
                        "content": "Adding straw mulch underneath the plants helped me prevent this from recurring last season.",
                        "created_at": "2026-10-06T18:40:00"
                    },
                    {
                        "comment_id": "c_3",
                        "author": "Anita Ray",
                        "content": "How often did you re-spray after heavy rainfall?",
                        "created_at": "2026-10-07T08:10:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_tomato_late_blight",
                "author": "Dr. Aris Thorne",
                "author_badge": "AgroIntelli Agronomist",
                "plant_id": "tomato",
                "plant_name": "Tomato",
                "disease_id": "tomato_late_blight",
                "disease_name": "Late Blight",
                "content": "⚠️ Regional Alert for Solanaceous Crops: Cool night temperatures (12–18°C) paired with prolonged humidity (>80%) create prime infection windows for Phytophthora infestans. Check leaf undersides for faint white velvety mold before lesions turn water-soaked brown.",
                "upvotes": 89,
                "downvotes": 1,
                "comment_count": 2,
                "created_at": "2026-10-07T09:00:00",
                "comments": [
                    {
                        "comment_id": "c_4",
                        "author": "David Miller",
                        "content": "Lost half my crop to this two years ago. Early morning inspection is non-negotiable.",
                        "created_at": "2026-10-07T10:20:00"
                    },
                    {
                        "comment_id": "c_5",
                        "author": "FarmTech Collective",
                        "content": "Systemic fungicides (e.g. cymoxanil or mefenoxam) must be applied within 24 hours of first symptom detection.",
                        "created_at": "2026-10-07T11:45:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_potato_late_blight",
                "author": "Elena Vance",
                "author_badge": "Organic Farmer",
                "plant_id": "potato",
                "plant_name": "Potato",
                "disease_id": "potato_late_blight",
                "disease_name": "Late Blight",
                "content": "We noticed dark water-soaked patches spreading on potato margins right after foggy mornings. Bordeaux mixture (copper sulphate + lime) combined with hilling loose soil around tubers stopped rot from reaching underground potatoes.",
                "upvotes": 36,
                "downvotes": 3,
                "comment_count": 1,
                "created_at": "2026-10-05T11:00:00",
                "comments": [
                    {
                        "comment_id": "c_6",
                        "author": "Vikram Patel",
                        "content": "Make sure to cut and dispose of top foliage 2 weeks before harvesting tubers if blight was present.",
                        "created_at": "2026-10-05T13:30:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_corn_common_rust",
                "author": "Marcus Green",
                "author_badge": "Corn Producer",
                "plant_id": "corn",
                "plant_name": "Corn",
                "disease_id": "corn_common_rust",
                "disease_name": "Common Rust",
                "content": "Reddish-brown powdery pustules showed up on the upper leaf canopy of our sweet corn. Increasing inter-row spacing to 30 inches and switching to morning drip irrigation stopped spore germination entirely.",
                "upvotes": 29,
                "downvotes": 0,
                "comment_count": 1,
                "created_at": "2026-10-06T17:45:00",
                "comments": [
                    {
                        "comment_id": "c_7",
                        "author": "Agronomy Today",
                        "content": "Puccinia sorghi thrives in moderate temperatures (16–25°C). Planting resistant hybrids is the most sustainable prevention.",
                        "created_at": "2026-10-06T19:00:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_apple_apple_scab",
                "author": "Sarah Chen",
                "author_badge": "Orchardist",
                "plant_id": "apple",
                "plant_name": "Apple",
                "disease_id": "apple_apple_scab",
                "disease_name": "Apple Scab",
                "content": "Olive-green velvety spots on young apple leaves in early spring. Complete orchard floor leaf sanitation in late autumn plus wettable sulfur at green tip stage reduced ascospore pressure by over 85% this season.",
                "upvotes": 54,
                "downvotes": 1,
                "comment_count": 2,
                "created_at": "2026-10-04T12:00:00",
                "comments": [
                    {
                        "comment_id": "c_8",
                        "author": "Ben Robertson",
                        "content": "Flail mowing fallen leaves with a nitrogen spray accelerates decomposition and neutralizes over-wintering spores.",
                        "created_at": "2026-10-04T15:00:00"
                    },
                    {
                        "comment_id": "c_9",
                        "author": "Sarah Chen",
                        "content": "Exactly! The flail mower was our best investment this year.",
                        "created_at": "2026-10-04T16:20:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_pepper_bacterial_spot",
                "author": "Carlos Morales",
                "author_badge": "Horticulturist",
                "plant_id": "pepper",
                "plant_name": "Pepper",
                "disease_id": "pepper_bacterial_spot",
                "disease_name": "Bacterial Leaf Spot",
                "content": "Small water-soaked circular spots with yellow halos appeared on bell pepper leaves after severe wind-driven rain. Used fixed copper bactericide + mancozeb tank mix, and removed infected lower leaves. Canopy stabilized in 5 days.",
                "upvotes": 38,
                "downvotes": 1,
                "comment_count": 1,
                "created_at": "2026-10-05T08:30:00",
                "comments": [
                    {
                        "comment_id": "c_10",
                        "author": "Lucia Gomez",
                        "content": "Never cultivate or walk through pepper rows when foliage is wet — bacteria spreads rapidly via contact.",
                        "created_at": "2026-10-05T09:45:00"
                    }
                ]
            },
            {
                "post_id": "seed_post_tomato_healthy",
                "author": "AgroIntelli Community",
                "author_badge": "Official Guide",
                "plant_id": "tomato",
                "plant_name": "Tomato",
                "disease_id": "tomato_healthy",
                "disease_name": "Healthy Foliage",
                "content": "🌱 Guide to Ideal Foliar Health: Vibrant green cuticle with balanced transpiration and zero chlorotic margins. Companion planting with basil and marigolds acts as a natural deterrent against whiteflies and hornworms.",
                "upvotes": 72,
                "downvotes": 0,
                "comment_count": 1,
                "created_at": "2026-10-03T10:00:00",
                "comments": [
                    {
                        "comment_id": "c_11",
                        "author": "Young Farmers Network",
                        "content": "Routine scouting every 3-5 days is the best medicine for keeping plants healthy!",
                        "created_at": "2026-10-03T12:15:00"
                    }
                ]
            }
        ]
        for p in seed_posts:
            self.save_community_post(p)


db_store = StorageManager()

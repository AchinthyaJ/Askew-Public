from flask import Flask, request, jsonify, render_template, send_from_directory, session, make_response
import json
import random
import os
import re
import requests
import time
import secrets
import threading
from datetime import timedelta
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_askew_local')
app.permanent_session_lifetime = timedelta(minutes=30)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_BOTS_DIR = os.path.join(BASE_DIR, 'bots')       # Built-in bots
USER_BOTS_DIR = os.path.join(BASE_DIR, 'user_bots')  # User-created bots
FLAGS_DIR = os.path.join(BASE_DIR, 'flagged_reports')

# External Services (Optional - set env vars to enable)
LSTM_API_URL = os.environ.get('LSTM_API_URL', 'http://127.0.0.1:5000/train')
LSTM_CHAT_URL = os.environ.get('LSTM_CHAT_URL', 'http://127.0.0.1:5000/chat')

if not os.path.exists(USER_BOTS_DIR):
    os.makedirs(USER_BOTS_DIR)
if not os.path.exists(FLAGS_DIR):
    os.makedirs(FLAGS_DIR)

# --- Global Cache for Loaded Bots ---
LOADED_BOTS = {}

# --- Training Queue ---
PENDING_TRAINING = {} # { bot_id: intents_json }
TRAINING_LOCK = threading.Lock()

# --- Unknown Queries Cache ---
UNKNOWN_QUERIES = [] 

# --- Unknown Query Handler ---
class UnknownQueryHandler:
    @staticmethod
    def log(bot_id, query):
        """Logs a low-confidence query to memory."""
        UNKNOWN_QUERIES.append({
            "bot_id": bot_id,
            "query": query,
            "timestamp": time.time()
        })
        if len(UNKNOWN_QUERIES) > 2000:
            UNKNOWN_QUERIES.pop(0)

    @staticmethod
    def get_report(bot_id, active_bot_data):
        """Groups unknown queries by similarity and suggests intents."""
        queries = [q['query'] for q in UNKNOWN_QUERIES if q['bot_id'] == bot_id]
        if not queries: 
            return []
        
        vectorizer = TfidfVectorizer()
        try:
            X = vectorizer.fit_transform(queries)
        except ValueError:
            return [] 
            
        sim_matrix = cosine_similarity(X)
        clusters = [] 
        visited = [False] * len(queries)
        
        for i in range(len(queries)):
            if visited[i]: continue
            
            cluster_queries = [queries[i]]
            visited[i] = True
            
            for j in range(i+1, len(queries)):
                if not visited[j]:
                    if sim_matrix[i][j] > 0.65:
                        cluster_queries.append(queries[j])
                        visited[j] = True
            
            suggested_intent = "Unknown"
            confidence = 0.0
            
            if active_bot_data and active_bot_data.get('vectorizer') and active_bot_data.get('clf'):
                rep_query = cluster_queries[0]
                vec = active_bot_data['vectorizer'].transform([rep_query])
                probs = active_bot_data['clf'].predict_proba(vec)[0]
                max_prob = max(probs)
                best_idx = np.argmax(probs)
                
                if max_prob > 0.3:
                    suggested_intent = active_bot_data['clf'].classes_[best_idx]
                    confidence = float(max_prob)

            clusters.append({
                "queries": list(set(cluster_queries)),
                "count": len(cluster_queries),
                "suggested_intent": suggested_intent,
                "confidence": round(confidence, 2)
            })
            
        clusters.sort(key=lambda x: x['count'], reverse=True)
        return clusters

# --- Presets ---
PRESETS = {
    "chatty": {
        "intents": [
            {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey"], "responses": ["Hello there!", "Hi! How can I help?"]},
            {"tag": "goodbye", "patterns": ["Bye", "See ya"], "responses": ["Goodbye!", "See you later!"]},
            {"tag": "name", "patterns": ["What is your name?", "Who are you?"], "responses": ["I am a chatty bot.", "Call me Bot."]}
        ]
    }
}

# --- File Storage Helpers ---

def get_bot_path(bot_id, filename):
    """Returns the full path for a bot file, checking user dir first then repo."""
    # 1. Check User Bots (Writable)
    user_path = os.path.join(USER_BOTS_DIR, bot_id, filename)
    if os.path.exists(user_path):
        return user_path
        
    # 2. Check Repo Bots (ReadOnly)
    repo_path = os.path.join(REPO_BOTS_DIR, bot_id, filename)
    if os.path.exists(repo_path):
        return repo_path
        
    return None

def get_writable_bot_path(bot_id, filename):
    """Returns path in USER_BOTS_DIR, creating dir if needed."""
    bot_dir = os.path.join(USER_BOTS_DIR, bot_id)
    if not os.path.exists(bot_dir):
        os.makedirs(bot_dir)
    return os.path.join(bot_dir, filename)

def get_all_bot_ids():
    """Aggregates bot IDs from repo and user directories."""
    bot_ids = set()
    
    # 1. Repo
    if os.path.exists(REPO_BOTS_DIR):
        for name in os.listdir(REPO_BOTS_DIR):
            if os.path.isdir(os.path.join(REPO_BOTS_DIR, name)):
                bot_ids.add(name)
                
    # 2. User Bots
    if os.path.exists(USER_BOTS_DIR):
        for name in os.listdir(USER_BOTS_DIR):
            if os.path.isdir(os.path.join(USER_BOTS_DIR, name)):
                bot_ids.add(name)
        
    return list(bot_ids)

def load_bot_meta(bot_id):
    """Loads bot metadata."""
    path = get_bot_path(bot_id, 'meta.json')
    if path:
        try:
            with open(path, 'r') as f: return json.load(f)
        except: pass
    return None

def load_bot(bot_id):
    """Loads a bot's model and intents into memory."""
    # 1. Load Intents
    path = get_bot_path(bot_id, 'intents.json')
    intents = None
    if path:
        try:
            with open(path, 'r') as f: intents = json.load(f)
        except: pass

    if intents is None:
        return None

    # 2. Build Model
    patterns = []
    tags = []
    responses_map = {}
    links_map = {}
    
    for intent in intents.get('intents', []):
        tag = intent['tag']
        responses_map[tag] = intent['responses']
        if 'link' in intent and isinstance(intent['link'], dict):
            links_map[tag] = intent['link']
            
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(tag)

    if not patterns:
        return {
            "clf": None,
            "vectorizer": None,
            "responses": {},
            "links": {},
            "intents_data": intents
        }

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, tags)
    
    bot_data = {
        "clf": clf,
        "vectorizer": vectorizer,
        "responses": responses_map,
        "links": links_map,
        "intents_data": intents
    }
    LOADED_BOTS[bot_id] = bot_data
    return bot_data

# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        bot_id = kwargs.get('bot_id')
        if not bot_id:
             return jsonify({"error": "Bot ID required"}), 400
        
        session_key = f"editor_{bot_id}"
        if session_key not in session:
             return jsonify({"error": "Unauthorized. Please login to the editor."} ), 401
        return f(*args, **kwargs)
    return decorated_function

def generate_user_id(name):
    clean = re.sub(r'[^a-z0-9]', '', name.lower())
    if not clean: clean = "bot"
    return clean[:15] + str(random.randint(100,999))

def generate_bot_json(data):
    mode = data.get('mode', 'ai')
    if mode == 'preset':
        return PRESETS.get(data.get('preset_name', 'chatty'), PRESETS['chatty'])
    elif mode == 'scratch':
        return { "intents": [{"tag": "greeting", "patterns": ["Hi"], "responses": ["Hello!"]}] }
    elif mode == 'import':
        try:
            return json.loads(data.get('imported_json', '{}'))
        except: return {"error": "Invalid JSON"}

    # AI Mode
    api_key = data.get('api_key')
    if not api_key: return { "error": "API Key is required." }

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        prompt = f"Generate intents.json for bot: {data.get('purpose', 'General Chat')}."
        payload = { "contents": [{ "parts": [{"text": prompt}] }] }
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        content = r.json()['candidates'][0]['content']['parts'][0]['text']
        # Simple cleanup
        if '```json' in content: content = content.split('```json')[1].split('```')[0]
        elif '```' in content: content = content.split('```')[1].split('```')[0]
        return json.loads(content.strip())
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

# --- Training (Local .pth) ---
def train_bot_task(bot_id, intents):
    # This function attempts to call a local training service if available.
    # If not, the bot runs on Sklearn (in-memory) via load_bot.
    try:
        resp = requests.post(LSTM_API_URL, json=intents, timeout=5)
        if resp.status_code == 200:
            path = get_writable_bot_path(bot_id, "model.pth")
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
    except:
        return False # Fallback to Sklearn

# --- Routes ---

@app.route("/")
def home():
    bot_id = request.args.get('bot')
    bot_name = None
    bot_avatar = 'default'
    
    if bot_id:
        meta = load_bot_meta(bot_id)
        if meta:
            bot_name = meta.get('name', bot_id)
            bot_avatar = meta.get('avatar_id', 'default')
        else:
            bot_name = bot_id 

    all_bot_ids = get_all_bot_ids()
    public_bots = []
    for bid in all_bot_ids:
        meta = load_bot_meta(bid)
        if meta and not meta.get('private', False):
            public_bots.append({
                "id": bid,
                "name": meta.get('name', bid),
                "avatar": meta.get('avatar_id', 'default'),
                "created_at": meta.get('created_at', 0)
            })

    return render_template("index.html", bot_id=bot_id, bot_name=bot_name, bot_avatar=bot_avatar, bots=public_bots)

@app.route("/get_bot_display_name")
def get_bot_display_name():
    bot_id = request.args.get('bot_id')
    meta = load_bot_meta(bot_id) if bot_id else None
    if meta:
        return jsonify({"name": meta.get('name', bot_id), "avatar": meta.get('avatar_id', 'default')})
    return jsonify({"name": bot_id or "Unknown", "avatar": "default"})

@app.route("/resolve_bot")
def resolve_bot():
    query = request.args.get('query', '').strip()
    if not query: return jsonify({"error": "Query required"}), 400
    
    # Check ID
    meta = load_bot_meta(query)
    if meta:
        return jsonify({"bot_id": query, "name": meta.get('name', query), "avatar": meta.get('avatar_id', 'default')})
        
    # Check Name
    query_lower = query.lower()
    for bid in get_all_bot_ids():
        meta = load_bot_meta(bid)
        if meta and meta.get('name', '').strip().lower() == query_lower:
             return jsonify({"bot_id": bid, "name": meta.get('name'), "avatar": meta.get('avatar_id', 'default')})
                
    return jsonify({"error": "Bot not found"}), 404

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    bot_json = generate_bot_json(data)
    if "error" in bot_json: return jsonify(bot_json), 500

    creator_name = data.get('creator_name', 'user')
    user_id = generate_user_id(creator_name)
    
    # Save Intents
    with open(get_writable_bot_path(user_id, "intents.json"), "w") as f:
        json.dump(bot_json, f, indent=2)

    # Save Meta
    password = secrets.token_hex(4)
    meta_data = {
        "password": password, 
        "created_at": time.time(),
        "name": data.get('bot_name', f"Bot {user_id}"),
        "private": False,
        "avatar_id": "default"
    }
    with open(get_writable_bot_path(user_id, "meta.json"), "w") as f:
        json.dump(meta_data, f, indent=2)
            
    train_bot_task(user_id, bot_json)
    return jsonify({
        "message": "Bot created!",
        "bot_id": user_id,
        "password": password,
        "download_url": f"/bots/{user_id}/intents.json"
    })

@app.route("/editor-login")
def editor_login():
    return render_template("editor_login.html", bot_id=request.args.get('bot'))

@app.route("/editor")
def editor_root():
    return render_template("editor_login.html")

@app.route("/auth/editor", methods=["POST"])
def auth_editor():
    data = request.get_json()
    bot_id = data.get("bot_id")
    password = data.get("password")
    
    meta = load_bot_meta(bot_id)
    if not meta or meta.get("password") != password:
        return jsonify({"error": "Invalid Bot ID or Password"}), 401
        
    session[f"editor_{bot_id}"] = True
    session[f"editor_expiry_{bot_id}"] = time.time() + 1800
    return jsonify({"success": True, "redirect": f"/editor/{bot_id}"})

@app.route("/editor/<bot_id>")
def editor_page(bot_id):
    if f"editor_{bot_id}" not in session:
        return render_template("editor_login.html", bot_id=bot_id)
    
    path = get_bot_path(bot_id, 'intents.json')
    if not path: return "Bot not found", 404
    
    with open(path, 'r') as f: intents = json.load(f)
    meta = load_bot_meta(bot_id) or {}
    
    meta_for_tpl = {
        "name": meta.get("name", bot_id),
        "private": meta.get("private", False),
        "password": meta.get("password", ""),
        "avatar_id": meta.get("avatar_id", "default")
    }
    return render_template("editor.html", bot_id=bot_id, intents=intents, meta=meta_for_tpl)

@app.route("/editor/<bot_id>/save", methods=["POST"])
@login_required
def save_editor(bot_id):
    data = request.get_json()
    with open(get_writable_bot_path(bot_id, "intents.json"), "w") as f:
        json.dump(data, f, indent=2)
    
    if bot_id in LOADED_BOTS: del LOADED_BOTS[bot_id]
    train_bot_task(bot_id, data)
    return jsonify({"message": "Saved!"})

@app.route("/editor/<bot_id>/save-meta", methods=["POST"])
@login_required
def save_editor_meta(bot_id):
    data = request.get_json()
    current_meta = load_bot_meta(bot_id) or {}
    
    current_meta["name"] = data.get("name", current_meta.get("name", bot_id))
    if data.get("password"): current_meta["password"] = data["password"]
    current_meta["private"] = data.get("private", False)
    current_meta["agentic"] = data.get("agentic", False)
    current_meta["avatar_id"] = data.get("avatar_id", "default")
    
    with open(get_writable_bot_path(bot_id, "meta.json"), "w") as f:
        json.dump(current_meta, f, indent=2)
    return jsonify({"message": "Settings saved!"})

@app.route("/editor/<bot_id>/delete", methods=["POST"])
@login_required
def delete_bot(bot_id):
    import shutil
    # Delete from User Bots
    user_dir = os.path.join(USER_BOTS_DIR, bot_id)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        
    session.pop(f"editor_{bot_id}", None)
    return jsonify({"message": "Deleted", "redirect": "/"})

@app.route("/chat/<bot_id>", methods=["POST"])
def chat_bot(bot_id):
    data = request.get_json()
    user_input = data.get("message", "").strip()
    
    # 1. Load Bot
    bot = load_bot(bot_id)
    if not bot or not bot.get("clf"):
        return jsonify({"reply": "I'm offline or have no brain.", "debug_source": "Error"}), 404
        
    # 2. Predict
    vec = bot["vectorizer"].transform([user_input])
    probs = bot["clf"].predict_proba(vec)[0]
    max_prob = max(probs)
    
    if max_prob < 0.1:
        UnknownQueryHandler.log(bot_id, user_input)
        return jsonify({"reply": "I don't understand.", "debug_source": "Low Confidence"})
        
    tag = bot["clf"].classes_[np.argmax(probs)]
    reply = random.choice(bot["responses"].get(tag, ["..."]))
    link = bot["links"].get(tag)
    
    return jsonify({"reply": reply, "link": link, "debug_source": "Sklearn"})

# --- Static/Template Routes ---
@app.route("/builder")
def builder(): return render_template("builder.html")

@app.route("/how-it-works")
def how_it_works(): return render_template("how_it_works.html")

@app.route("/ui-preview")
def ui_preview(): return render_template("ui_preview.html")

@app.route("/bots/<bot_id>/intents.json")
def download_bot(bot_id):
    path = get_bot_path(bot_id, 'intents.json')
    if path:
        return send_from_directory(os.path.dirname(path), 'intents.json', as_attachment=True)
    return "Not found", 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
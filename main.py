import os

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from sklearn.svm import SVC

app = Flask(__name__, 
    static_folder='../frontend'  # Change this to point to the frontend root
)
CORS(app)  # Enable CORS for all routes


# Sample game data
games = [
    {"id": 1, "name": "Call of Duty", "tags": ["fighting", "multiplayer", "competitive"]},
    {"id": 2, "name": "Minecraft", "tags": ["world-building", "multiplayer", "casual"]},
    {"id": 3, "name": "The Last of Us", "tags": ["story", "scary", "single-player"]},
    {"id": 4, "name": "Fortnite", "tags": ["fighting", "multiplayer", "competitive", "world-building"]},
    {"id": 5, "name": "Candy Crush", "tags": ["casual", "mobile"]},
    {"id": 6, "name": "Resident Evil", "tags": ["scary", "story", "fighting"]},
    {"id": 7, "name": "Roblox", "tags": ["obby", "multiplayer", "world-building", "casual"]},
    {"id": 8, "name": "FIFA", "tags": ["competitive", "multiplayer", "purchase"]},
    {"id": 9, "name": "Clash of Clans", "tags": ["mobile", "multiplayer", "casual"]},
    {"id": 10, "name": "God of War", "tags": ["story", "fighting", "single-player"]}
]

# Map questions to tags
question_to_tag = {
    0: "fighting",
    1: "multiplayer",
    2: "world-building",
    3: "scary",
    4: "casual",
    5: "story",
    6: "obby",
    7: "purchase",
    8: "competitive",
    9: "mobile"
}

# Simple SVM model for game recommendations
def train_svm_model():
    # Create feature vectors for each game (1 if tag present, 0 if not)
    all_tags = list(set(tag for game in games for tag in game["tags"]))
    X = []
    y = []
    
    for i, game in enumerate(games):
        feature_vector = [1 if tag in game["tags"] else 0 for tag in all_tags]
        X.append(feature_vector)
        y.append(i)  # Game index as label
    
    # Train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    return model, all_tags

# Train the model
model, all_tags = train_svm_model()

@app.route('/recommend', methods=['POST'])
def recommend_games():
    try:
        # Get user answers from request
        data = request.json
        user_answers = data.get('answers', {})
        
        # Convert answers to feature vector
        user_features = []
        for tag in all_tags:
            # Check if any question corresponding to this tag was answered 'yes'
            has_tag = False
            for q_idx, answer in user_answers.items():
                if question_to_tag[int(q_idx)] == tag and answer:
                    has_tag = True
                    break
            user_features.append(1 if has_tag else 0)
        
        # Get probabilities for each game
        probs = model.predict_proba([user_features])[0]
        
        # Get top 3 game recommendations
        top_indices = np.argsort(probs)[-3:][::-1]
        recommendations = [games[idx] for idx in top_indices]
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get-news', methods=['GET'])
def get_news():
    try:
        response = requests.get('https://www.freetogame.com/api/games?sort-by=release-date')
        if response.status_code == 200:
            games = response.json()[:12]  # Get top 12 games (3 for main + 9 for grid)
            return jsonify({
                "success": True,
                "news": games
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to fetch news"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Update routes to use absolute paths
@app.route('/')
def index():
    return send_from_directory('../frontend/pages', 'index.html')

@app.route('/login.html')  # Match the exact file names
def login():
    return send_from_directory('../frontend/pages', 'login.html')

@app.route('/signup.html')
def signup():
    return send_from_directory('../frontend/pages', 'signup.html')

@app.route('/form.html')
def form():
    return send_from_directory('../frontend/pages', 'form.html')

@app.route('/info.html')
def info():
    return send_from_directory('../frontend/pages', 'info.html')

@app.route('/reccomender.html')  # Note: matches your filename spelling
def recommender():
    return send_from_directory('../frontend/pages', 'reccomender.html')

# Add route for serving images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../frontend/images', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
import os
import warnings

warnings.filterwarnings("ignore")

# --- 1. TRAIN THE NEURAL NETWORK ---
def train_difficulty_nn():
    """Trains a small Neural Network to predict the optimal next difficulty level."""
    X = [] # Features: [current_difficulty, score_percentage]
    y = [] # Labels: target_difficulty (1 to 8, where 8 signals a Topic Progression)
    
    # Generate synthetic student performance data
    for diff in range(1, 8):
        for score in np.linspace(0, 1, 21): # Scores from 0.0 to 1.0
            X.append([diff, score])
            if score < 0.5:
                y.append(max(1, diff - 1)) # Drop a level
            elif score < 0.8:
                y.append(diff)             # Maintain level
            else:
                y.append(min(8, diff + 1)) # Advance a level (8 means next topic)
                
    # Build a small Multi-Layer Perceptron (NN)
    nn = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)
    nn.fit(X, y)
    return nn

# Initialize the trained Neural Network
difficulty_model = train_difficulty_nn()

# Define the exact progression pathway requested
TOPIC_PROGRESSION = [
    "ARRAY", "Binary Search", "Stack & Queue", 
    "LINKED LIST", "Bit Manipulation", "Tree", "Graph"
]

def recommend_next_video(current_video_id, current_title, current_subject, current_topic, score, total_questions):
    """Recommends next video using NN prediction and Topic Progression."""
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'video_database.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to older name if they didn't rename it
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'videos.csv')
        df = pd.read_csv(csv_path)

    # Clean the video_id column in case it contains full URLs
    from modules.extractor import extract_video_id
    df['clean_id'] = df['video_id'].apply(lambda x: extract_video_id(str(x)) if 'http' in str(x) else str(x))

    # Calculate Score Ratio
    score_ratio = score / total_questions
    
    # Get Current Difficulty
    current_vid_data = df[df['clean_id'] == current_video_id]
    current_diff = current_vid_data.iloc[0]['difficulty'] if not current_vid_data.empty else 1
    
    # --- 2. NEURAL NETWORK PREDICTION ---
    # Ask the AI what difficulty the student needs next
    pred_target = difficulty_model.predict([[current_diff, score_ratio]])[0]
    
    target_topic = current_topic
    target_diff = pred_target
    transition_msg = None
    
    # --- 3. TOPIC PROGRESSION LOGIC ---
    # If the NN predicts level 8, it means they beat level 7 with a high score.
    if pred_target == 8 or (current_diff == 7 and score_ratio >= 0.8):
        # Normalize casing to find the topic in our progression list
        progression_lower = [t.lower() for t in TOPIC_PROGRESSION]
        curr_topic_lower = current_topic.lower()
        
        if curr_topic_lower in progression_lower:
            curr_idx = progression_lower.index(curr_topic_lower)
            if curr_idx < len(TOPIC_PROGRESSION) - 1:
                # Advance to next topic, reset difficulty to 1
                target_topic = TOPIC_PROGRESSION[curr_idx + 1]
                target_diff = 1 
                transition_msg = f"**Topic Mastered!** Promoting you from {current_topic} to **{target_topic}**."
            else:
                target_diff = 7
                transition_msg = "**Curriculum Mastered!** You have completed the highest level of Graph algorithms."
        else:
            target_diff = 7 # Failsafe if topic isn't in syllabus
            
    # --- 4. FILTER CANDIDATES ---
    # Filter case-insensitively for the target topic
    candidates = df[df['topic'].str.lower() == target_topic.lower()].copy()
    candidates = candidates[candidates['difficulty'] == target_diff]
    candidates = candidates[candidates['clean_id'] != current_video_id]
    
    if candidates.empty:
        # Fallback if no specific difficulty video exists in that topic
        candidates = df[df['topic'].str.lower() == target_topic.lower()].copy()
        candidates = candidates[candidates['clean_id'] != current_video_id]
        
    if candidates.empty:
        return {"error": f"No available videos found for {target_topic}."}

    # --- 5. NLP SIMILARITY RANKING ---
    candidates = candidates.reset_index(drop=True)
    text_corpus = [current_title] + candidates['title'].tolist()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    best_match_idx = cosine_sim.argmax()
    best_video = candidates.iloc[best_match_idx].to_dict()
    best_video['transition_msg'] = transition_msg # Pass message to UI
    
    return best_video
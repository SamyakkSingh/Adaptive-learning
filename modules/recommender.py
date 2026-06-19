import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def recommend_next_video(current_video_id, current_title, current_subject, score, total_questions):
    """
    Recommends a video strictly from the same subject, adjusting difficulty based on score,
    and using Cosine Similarity to find the most contextually relevant title.
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'videos.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": "Dataset not found. Please verify data/videos.csv exists."}

    # 1. Hard Filter: Stay strictly within the same academic subject
    subject_pool = df[df['subject'] == current_subject].copy()
    
    if subject_pool.empty:
        return {"error": f"No videos found in the subject pool: {current_subject}"}

    # 2. Determine Difficulty Target based on Score
    percentage = (score / total_questions) * 100
    current_vid_data = subject_pool[subject_pool['video_id'] == current_video_id]
    
    if not current_vid_data.empty:
        current_diff = current_vid_data.iloc[0]['difficulty']
    else:
        current_diff = 1 # Default to beginner

    # Core logic requested by mentor
    if percentage < 50:
        target_diff = max(1, current_diff - 1)  # Drop down a tier
    elif percentage < 80:
        target_diff = current_diff              # Maintain current tier
    else:
        target_diff = min(3, current_diff + 1)  # Scale up a tier

    # 3. Filter Candidates by Target Difficulty
    candidates = subject_pool[subject_pool['difficulty'] == target_diff].copy()
    
    # Remove the video the user just watched so we don't recommend it again
    candidates = candidates[candidates['video_id'] != current_video_id]
    
    # Fallback: If no alternative videos exist in that specific difficulty tier, 
    # relax the constraint to any other video in the same subject
    if candidates.empty:
        candidates = subject_pool[subject_pool['video_id'] != current_video_id].copy()
        
    if candidates.empty:
        return {"error": "No alternative videos available in this subject category."}

    # 4. Content-Based NLP Filtering (TF-IDF + Cosine Similarity)
    candidates = candidates.reset_index(drop=True)
    
    # We compare the currently watched title against the list of candidate titles
    text_corpus = [current_title] + candidates['title'].tolist()
    
    # Create the TF-IDF vector matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    # Compute similarity matrix between the seed video (index 0) and candidates (index 1 onwards)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Find the candidate with the highest similarity score
    best_match_idx = cosine_sim.argmax()
    
    # Return the dictionary data of the winning video
    return candidates.iloc[best_match_idx].to_dict()
    
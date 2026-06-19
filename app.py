import streamlit as st
import pandas as pd
import os
from modules.extractor import get_youtube_transcript
from modules.quiz_gen import generate_quiz
from modules.recommender import recommend_next_video

st.set_page_config(page_title="Adaptive Learning Platform", layout="centered")

# Load the video database into session state for global access
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'videos.csv')
if os.path.exists(csv_path):
    video_db = pd.read_csv(csv_path)
else:
    st.error("Please ensure data/videos.csv is created before running the application.")
    st.stop()

# --- SIDEBAR API CONFIG ---
with st.sidebar:
    st.title("🔑 Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    st.markdown("[Get a free key here](https://aistudio.google.com/app/apikey)")

st.title("🎓 Smart Learning Curated Pathway")
st.write("Select a subject field to begin your adaptive learning experience.")

# --- STEP 1: INITIAL VIDEO SELECTION FROM DATABASE ---
subjects = video_db['subject'].unique()
selected_subject = st.selectbox("Choose a Subject Field:", subjects)

# Filter videos by the chosen subject
filtered_videos = video_db[video_db['subject'] == selected_subject]
video_titles = filtered_videos['title'].tolist()
selected_title = st.selectbox("Select a Video to Watch:", video_titles)

# Retrieve the matching video row data
video_row = filtered_videos[filtered_videos['title'] == selected_title].iloc[0]
video_id = video_row['video_id']
video_url = f"https://www.youtube.com/watch?v={video_id}"

# Clear previous quiz if the user changes the base video
if 'last_video_id' in st.session_state and st.session_state['last_video_id'] != video_id:
    if 'quiz_data' in st.session_state:
        del st.session_state['quiz_data']
st.session_state['last_video_id'] = video_id

# --- LAYOUT DISPLAY ---
st.markdown("---")
st.subheader(f"📹 Active Lecture: {selected_title}")
st.video(video_url)

st.subheader("📝 Assessment Verification")
if not api_key:
    st.warning("⚠️ Enter your Gemini API Key in the left sidebar to unlock the evaluation quiz.")
else:
    if st.button("🧠 Generate Quiz from Lecture Material"):
        with st.spinner("Extracting localized transcript and processing questions..."):
            transcript = get_youtube_transcript(video_id)
            
            if "Error:" in transcript:
                st.error(transcript)
            else:
                quiz_data = generate_quiz(transcript, api_key)
                if isinstance(quiz_data, str) and "Error:" in quiz_data:
                    st.error(quiz_data)
                else:
                    st.session_state['quiz_data'] = quiz_data
                    st.rerun()

# --- INTERACTIVE EVALUATION & RECOMMENDATION FORM ---
if 'quiz_data' in st.session_state:
    st.markdown("---")
    st.subheader("📝 Test Your Knowledge")
    quiz_data = st.session_state['quiz_data']
    
    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(quiz_data):
            st.write(f"**Q{i+1}: {q['question']}**")
            user_answers[i] = st.radio("Select options:", q['options'], key=f"q_{i}", index=None)
            st.write("---")
            
        submitted = st.form_submit_button("Submit Performance Analytics")
        
        if submitted:
            score = 0
            for i, q in enumerate(quiz_data):
                if user_answers[i] == q['correct_answer']:
                    score += 1
            
            total_q = len(quiz_data)
            st.success(f"🎉 Performance Evaluation: You scored {score} out of {total_q}!")
            
            # Trigger the contextual subject recommender
            with st.spinner("Executing Content-Based Filtering Vector Pipeline..."):
                next_video = recommend_next_video(video_id, selected_title, selected_subject, score, total_q)
                
                if "error" in next_video:
                    st.error(next_video["error"])
                else:
                    st.markdown("### 🎯 Dynamic Next Step Recommendation")
                    
                    # Output rationale
                    perf_ratio = score / total_q
                    if perf_ratio < 0.5:
                        st.info(f"📉 Scaling down complexity within **{selected_subject}** to reinforce baseline principles.")
                    elif perf_ratio < 0.8:
                        st.info(f"⚖️ Maintaining complexity profile within **{selected_subject}** for conceptual variance.")
                    else:
                        st.info(f"🚀 Advancing complexity profile within **{selected_subject}** based on complete proficiency.")
                        
                    st.write(f"**Recommended Title:** {next_video['title']}")
                    st.write(f"**Specific Topic Integration:** {next_video['topic']} | **Assigned Difficulty:** {next_video['difficulty']}/3")
                    
                    rec_url = f"https://www.youtube.com/watch?v={next_video['video_id']}"
                    # Add a clickable fallback link
                    st.markdown(f"**[🔗 Click here to open the video directly on YouTube]({rec_url})**")
                    st.video(rec_url)
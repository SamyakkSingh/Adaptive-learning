import streamlit as st
import pandas as pd
import os
from modules.extractor import extract_video_id, get_youtube_transcript
from modules.quiz_gen import generate_quiz
from modules.recommender import recommend_next_video

st.set_page_config(page_title="AI Adaptive DSA Platform", layout="centered")

# --- LOAD DATABASE ---
csv_path_new = os.path.join(os.path.dirname(__file__), 'data', 'video_database.csv')
csv_path_old = os.path.join(os.path.dirname(__file__), 'data', 'videos.csv')

if os.path.exists(csv_path_new):
    video_db = pd.read_csv(csv_path_new)
elif os.path.exists(csv_path_old):
    video_db = pd.read_csv(csv_path_old)
else:
    st.error("Please ensure your CSV file is inside the 'data' folder.")
    st.stop()

# Clean raw URLs into IDs just in case
video_db['clean_id'] = video_db['video_id'].apply(lambda x: extract_video_id(str(x)) if 'http' in str(x) else str(x))

with st.sidebar:
    st.title("🔑 Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    st.markdown("[Get a free key here](https://aistudio.google.com/app/apikey)")

st.title("🎓 Smart Learning: DSA Pathway")
st.write("Navigate the syllabus: Array ➔ Binary Search ➔ Stack & Queue ➔ Linked List ➔ Bit Manipulation ➔ Tree ➔ Graph")
st.markdown("---")

# --- 3-TIER DROPDOWN NAVIGATION (WITH LOOPING STATE) ---
col1, col2 = st.columns(2)

subjects = video_db['subject'].unique().tolist()
if 'sb_subject' not in st.session_state or st.session_state.sb_subject not in subjects:
    st.session_state.sb_subject = subjects[0]

with col1:
    selected_subject = st.selectbox("1. Select Subject:", subjects, key="sb_subject")

topics_in_subject = video_db[video_db['subject'] == selected_subject]['topic'].unique().tolist()
if 'sb_topic' not in st.session_state or st.session_state.sb_topic not in topics_in_subject:
    st.session_state.sb_topic = topics_in_subject[0]

with col2:
    selected_topic = st.selectbox("2. Select Topic:", topics_in_subject, key="sb_topic")

filtered_videos = video_db[(video_db['subject'] == selected_subject) & (video_db['topic'] == selected_topic)]
video_titles = filtered_videos['title'].tolist()
if 'sb_title' not in st.session_state or st.session_state.sb_title not in video_titles:
    st.session_state.sb_title = video_titles[0]

selected_title = st.selectbox("3. Select Video Lecture:", video_titles, key="sb_title")

# Retrieve the matching video row data
video_row = filtered_videos[filtered_videos['title'] == selected_title].iloc[0]
video_id = video_row['clean_id']
video_url = f"https://www.youtube.com/watch?v={video_id}"

# Clear previous quiz and recommendations if the base video changes
if 'last_video_id' in st.session_state and st.session_state['last_video_id'] != video_id:
    for key in ['quiz_data', 'quiz_submitted', 'quiz_score', 'next_video']:
        if key in st.session_state:
            del st.session_state[key]
st.session_state['last_video_id'] = video_id

# --- LAYOUT DISPLAY ---
st.markdown("---")
st.subheader(f"Active Lecture: {selected_title}")
st.markdown(f"**Difficulty Level:** {video_row['difficulty']}/7")
st.video(video_url)

st.subheader("Assessment Verification")
if not api_key:
    st.warning("Enter your Gemini API Key in the left sidebar to unlock the evaluation quiz.")
else:
    if st.button("Generate Quiz from Lecture Material"):
        with st.spinner("Extracting transcript and generating quiz..."):
            transcript = get_youtube_transcript(video_id)
            
            # --- GRACEFUL DEGRADATION FALLBACK ---
            if "Error:" in transcript:
                st.warning("YouTube rate-limit detected (IP Block). Falling back to AI Conceptual Generation...")
                
                # Create a synthetic transcript using just the known context
                fallback_context = f"We could not fetch the exact transcript. However, the student is learning the DSA topic '{selected_topic}'. The specific video lecture is titled '{selected_title}'. Please generate a quiz testing standard computer science knowledge regarding this specific concept."
                
                quiz_data = generate_quiz(fallback_context, api_key)
            else:
                # Normal transcript processing
                quiz_data = generate_quiz(transcript, api_key)
                
            # Error handling for the quiz generation itself
            if isinstance(quiz_data, str) and "Error:" in quiz_data:
                st.error(quiz_data)
            else:
                st.session_state['quiz_data'] = quiz_data
                st.rerun()

# --- INTERACTIVE EVALUATION & RECOMMENDATION FORM ---
if 'quiz_data' in st.session_state:
    st.markdown("---")
    st.subheader("Test Your Knowledge")
    quiz_data = st.session_state['quiz_data']
    
    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(quiz_data):
            st.write(f"**Q{i+1}: {q['question']}**")
            user_answers[i] = st.radio("Select options:", q['options'], key=f"q_{i}", index=None)
            st.write("---")
            
        submitted = st.form_submit_button("Submit Analytics & Get Recommendation")
        
        # Save score into state so the UI doesn't disappear when we click other buttons later
        if submitted:
            score = 0
            for i, q in enumerate(quiz_data):
                if user_answers[i] == q['correct_answer']:
                    score += 1
            st.session_state['quiz_submitted'] = True
            st.session_state['quiz_score'] = score
            # Clear old recommendation so a new one is generated
            if 'next_video' in st.session_state:
                del st.session_state['next_video']

# --- DISPLAY AI RECOMMENDATION & CONTINUOUS LOOP ---
if st.session_state.get('quiz_submitted', False):
    score = st.session_state['quiz_score']
    total_q = len(st.session_state['quiz_data'])
    st.success(f"Performance Evaluation: You scored {score} out of {total_q}!")
    
    # Only run the Neural Network once per quiz submission
    if 'next_video' not in st.session_state:
        with st.spinner("Neural Network is predicting your optimal next video..."):
            next_video = recommend_next_video(video_id, selected_title, selected_subject, selected_topic, score, total_q)
            st.session_state['next_video'] = next_video
    
    next_video = st.session_state['next_video']
    
    if "error" in next_video:
        st.error(next_video["error"])
    else:
        st.markdown("### AI Curriculum Recommendation")
        
        # Display special topic progression message if triggered by NN
        if next_video.get('transition_msg'):
            st.success(next_video['transition_msg'])
        else:
            perf_ratio = score / total_q
            if perf_ratio < 0.5:
                st.info(f"Neural Net identified knowledge gaps. Recommending a slightly easier video in **{selected_topic}**.")
            elif perf_ratio < 0.8:
                st.info(f"Good job! Recommending lateral material to solidify **{selected_topic}** concepts.")
            else:
                st.info(f"Excellent! Neural Net is increasing complexity in **{selected_topic}**.")
                
        st.write(f"**Next Title:** {next_video['title']}")
        st.write(f"**Topic:** {next_video['topic']} | **Difficulty:** {next_video['difficulty']}/7")
        
        st.markdown("---")
        
        # THE MAGIC LOOP BUTTON
        def load_next_video():
            # 1. Update the Dropdown Menus to match the AI recommendation
            nv = st.session_state['next_video']
            st.session_state.sb_subject = nv['subject']
            st.session_state.sb_topic = nv['topic']
            st.session_state.sb_title = nv['title']
            
            # 2. Erase the old quiz history so it's a fresh start
            del st.session_state['quiz_data']
            del st.session_state['quiz_submitted']
            del st.session_state['quiz_score']
            del st.session_state['next_video']

        st.button("🎬 Watch Next Recommended Video ➔", type="primary", use_container_width=True, on_click=load_next_video)
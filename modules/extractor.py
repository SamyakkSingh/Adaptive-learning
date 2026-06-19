import re
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url):
    """Extracts the 11-character YouTube video ID from various URL formats."""
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_id):
    """Fetches text transcript for a given YouTube video ID."""
    try:
        # 1. Initialize the new API instance
        api = YouTubeTranscriptApi()
        
        # 2. Fetch the transcript using the new method
        fetched_transcript = api.fetch(video_id)
        
        # 3. Join the text. Notice we use .text now instead of ['text']!
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
        return transcript_text
        
    except Exception as e:
        return f"Error: Could not retrieve transcript. (Details: {e})"
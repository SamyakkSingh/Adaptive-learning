import json
import time
from google import genai
from google.genai import types

def generate_quiz(transcript_text, api_key, max_retries=3):
    """Uses Gemini API to generate a structured JSON quiz, with auto-retry for busy servers."""
    try:
        # Initialize the modern Gemini client
        client = genai.Client(api_key=api_key)
        
        # The prompt with a strict schema definition
        prompt = f"""
        You are an AI educational tutor. Generate a 3-question multiple-choice quiz based ONLY on the following transcript.
        
        The output MUST be a JSON array of objects. 
        Each object must follow exactly this schema:
        [{{
            "question": "The question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "The exact string from the options list that is correct"
        }}]
        
        Transcript:
        {transcript_text}
        """
        
        # Fault Tolerance: Attempt to call the API up to 'max_retries' times
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                
                # Convert the JSON string response directly into a Python dictionary/list
                quiz_data = json.loads(response.text)
                return quiz_data
                
            except Exception as e:
                # If it's a 503 Server Busy error, and we haven't run out of retries...
                if "503" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Wait 1s, then 2s, then 4s
                    time.sleep(wait_time)
                    continue # Try the loop again
                else:
                    # If it's a different error, or we tried 3 times and failed, break and report it.
                    raise e
                    
    except Exception as e:
        return f"Error: Could not generate quiz after multiple attempts. (Details: {e})"
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
        You are an AI educational tutor. Generate a 7-question multiple-choice quiz based ONLY on the following transcript.
        
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
                
                # Safely check if the response has text before parsing
                if not response or not response.text:
                    raise ValueError("Empty response returned from Gemini API.")
                    
                # Convert the JSON string response directly into a Python dictionary/list
                quiz_data = json.loads(response.text)
                return quiz_data
                
            except Exception as e:
                error_msg = str(e)
                # Retry if: Server Busy (503), JSON parsing fails, or Empty Response (ValueError)
                if ("503" in error_msg or isinstance(e, json.JSONDecodeError) or isinstance(e, ValueError)) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Wait 1s, then 2s, then 4s
                    time.sleep(wait_time)
                    continue # Try the loop again
                else:
                    # If it's a different error, or we tried 3 times and failed, break and report it.
                    raise e
                    
    except Exception as e:
        return f"Error: Could not generate quiz after multiple attempts. (Details: {e})"
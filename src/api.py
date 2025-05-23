import tempfile
import os
import time
import logging
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
import boto3
from fastapi.middleware.cors import CORSMiddleware
import base64
from .agent import chat_engine
from .utils import CustomCallBackHandler, thread_local, setup_improved_logging, log_thread_local_state
from src.AESEncryptor import AESEncryptor
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import re

# Import our essential tools and functionality
from .agent_tools import analyze_symptoms, dynamic_doctor_search, store_patient_details
from .query_builder_agent import unified_doctor_search, unified_doctor_search_tool, extract_search_criteria_tool
from .specialty_matcher import SpecialtyDataCache, get_recommended_specialty

# Create a custom JSON encoder class to handle Arabic text properly
class ArabicJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(obj)

# Store the original json.dumps function before replacing it
_original_json_dumps = json.dumps

# Override the default json.dumps function to ensure Arabic characters are properly handled
def arabic_json_dumps(obj, **kwargs):
    """
    Custom JSON dumps function that ensures Arabic characters are properly encoded
    without being converted to Unicode escape sequences.
    """
    # Always set ensure_ascii=False to preserve Arabic characters
    kwargs['ensure_ascii'] = False
    return _original_json_dumps(obj, **kwargs)

# Replace the standard json.dumps with our custom function
json.dumps = arabic_json_dumps

# Define a custom JSONResponse class that uses our encoder
class ArabicJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,  # This is crucial for Arabic text
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# Set up improved logging at startup
setup_improved_logging()

# Preload specialty data cache at application startup
logger = logging.getLogger(__name__)
logger.info("Preloading specialty data cache at application startup")
try:
    specialty_data = SpecialtyDataCache.get_instance()

    # Print loaded subspecialties to console
    unique_subspecialties = set()
    for entry in specialty_data:
        if entry.get("subspecialty"):
            unique_subspecialties.add(entry.get("subspecialty"))

    logger.info(f"Specialty data cache preloaded with {len(specialty_data)} specialty records")
    logger.info(f"Loaded {len(unique_subspecialties)} unique subspecialties:")
    for subspecialty in sorted(unique_subspecialties):
        logger.info(f"  - {subspecialty}")

    # No longer need to log variant mappings as we're using GPT for this now
    logger.info(f"Using GPT for subspecialty matching instead of hardcoded mappings")

except Exception as e:
    logger.error(f"Error preloading specialty data: {str(e)}")
    logger.warning("Application will continue but specialty matching may be affected")

# Workflow notes:
# 1. User describes symptoms → analyze_symptoms → detect symptoms and matching specialties
# 2. Results stored in session history including specialties and subspecialties
# 3. When searching for doctors, structured specialty/subspecialty information is passed 
#    directly to the query builder rather than just using natural language
# 4. This ensures consistent specialty mapping between symptom analysis and doctor search

# Create FastAPI app with our custom JSON response class as default
app = FastAPI(default_response_class=ArabicJSONResponse)
engine = chat_engine()
callBackHandler = CustomCallBackHandler()
encryptor = AESEncryptor(os.getenv("ENCRYPTION_SALT", "default_salt"))

# Set environment variables - load from .env directly
load_dotenv()  # This will load variables from .env file

# Get API key from environment with better error handling
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    error_msg = """
    ⚠️ ERROR: No OpenAI API key found in environment variables.
    
    To fix this:
    1. Create a .env file in the project root if it doesn't exist
    2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here
    3. Restart the application
    
    Alternatively, you can set the API_KEY or OPENAI_API_KEY environment variable directly.
    """
    logger.error(error_msg)
else:
    logger.info("✅ OpenAI API key found in environment variables")

# Set environment variables
os.environ["OPENAI_API_KEY"] = api_key if api_key else ""
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")

# Create a boto3 session
session = boto3.Session()

# Define allowed origins from environment variable or use defaults
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:8509").split(",")

# Create the AWS Polly client
try:
    polly_client = boto3.client('polly', region_name=os.environ.get("AWS_REGION"))
    logger.info("Polly client initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing Polly client: {str(e)}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from openai import OpenAI
# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=api_key)

def get_token_usage(response):
    # Extract token usage from the API response
    token_usage = response['usage']
    tokens_used = token_usage['total_tokens']
    return tokens_used

@app.post("/chat")
async def chat(
    message: str = Form(None),
    session_id: str = Form(...),
    audio: UploadFile = File(None),
    lat : Optional[float] = Form(None),
    long : Optional[float] = Form(None),
):
    start_time = time.time()
    
    try:

        if session_id == None or session_id == "":
            return {
                "response": {
                    "error" : "EMPTY SESSION ID NOT ALLOWED"
                }
            }

        
        pattern = r'^www\|.+'

        if re.match(pattern, session_id):
            print("Valid Session ID match: ", session_id)
        else:
            return {
                "response": {
                    "error" : "INVALID SESSION ID FORMAT"
                }
            }
        # Handle audio input
        if audio:
            if not audio.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")
            
                # Process audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
                contents = await audio.read()
                temp_file.write(contents)
            
            try:
                    # Transcribe audio
                with open(temp_file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        response_format="text"
                    )
                    print("-------------------------")
                    print("LAT: ", lat)
                    print("LONG: ", long)
                    print("-------------------------")
                    # Process transcribed message
                    response = process_message(transcription, session_id, lat, long)
                    
                    # Generate audio response
                audio_base64 = await text_to_speech(response['response']['message'])
                
                # Return a single response object with audio data added
                if "response" in response:
                    # Add audio data to existing response
                    response["response"]["Audio_base_64"] = audio_base64
                    response["response"]["transcription"] = transcription
                    return response
                else:
                    # Create a properly formatted response
                    return {
                        "response": {
                            "message": response.get('message', ''),
                            "transcription": transcription,
                            "Audio_base_64": audio_base64,
                            "patient": response.get('patient', {"session_id": session_id}),
                            "data": response.get('data', [])
                        }
                    }
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
            # Handle text input
        if message:
                print("-------------------------")
                print("LAT: ", lat)
                print("LONG: ", long)
                print("-------------------------")
                response = process_message(message, session_id, lat, long)
                return response
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {
            "response": {
                "message": f"I encountered an error processing your request: {str(e)}",
                "patient": {"session_id": session_id},
                "data": []
            }
        }

def process_message(message: str, session_id: str, lat: float = None, long: float = None) -> dict:
    """Process incoming message by sending it to the agent"""
    try:
        logger.info(f"Processing message for session {session_id}")
        if lat is not None and long is not None:
            logger.info(f"Using coordinates: lat={lat}, long={long}")
        
        # Forward the message to the agent and get response
        response = engine.invoke(
            {
                "input": message, 
                "session_id": session_id,
                "lat": lat,
                "long": long
            },
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [callBackHandler]
            }
        )
        
        # Return the response without wrapping it in another response object
        return response

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        try:
            # Get LLM's response for error handling
            error_response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a medical assistant. Provide a natural error response that matches the conversation style."},
                    {"role": "user", "content": f"There was an error processing the request. Generate an apologetic response in the same language style as the conversation."}
                ]
            )
            error_message = error_response.choices[0].message.content
        except Exception as llm_error:
            logger.error(f"Error getting LLM error response: {str(llm_error)}")
            error_message = ""  # Empty message will be handled by main LLM
        
        return {
            "response": {
                "message": error_message,
                "patient": {"session_id": session_id},
                "data": []
            }
        }

class ImageRequest(BaseModel):
    base64_image: str  # Expecting a base64-encoded image string

@app.post("/analyze-image")
async def analyze_image(request: ImageRequest):
    try:
        # Extract the base64 image from request
        base64_image = request.base64_image

        # Ensure it is a valid base64 string
        try:
            base64.b64decode(base64_image.split(",")[1])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image format")

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is an image of a medical condition. Your job is to provide the corresponding medical specialty along with relevant medical tags, comma-separated."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image,  
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        description = response.choices[0].message.content
        print(description)
        
        return {"status": "success", "description": description}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    
@app.get("/")
def read_root():
    return {"response": "Hello World"}

@app.post("/reset")
async def reset(session_id: str = Form(...)):
    """Reset a specific session, clearing all history and thread_local data"""
    try:
        logger.info(f"Resetting session {session_id}")
        
        # Reset engine history for this session if it exists
        from .agent import store
        if session_id in store:
            logger.info(f"Clearing chat history for session {session_id}")
            store[session_id].clear()
            del store[session_id]
        
        # Reset thread_local data if it matches this session
        if hasattr(thread_local, 'session_id') and thread_local.session_id == session_id:
            logger.info(f"Clearing thread_local data for session {session_id}")
            # Log current state before clearing
            log_thread_local_state(logger, session_id)
            # Clear all session-specific attributes
            for attr in dir(thread_local):
                if not attr.startswith('_'):
                    logger.info(f"Deleting thread_local.{attr}")
                    delattr(thread_local, attr)
        
        return {"status": "success", "message": f"Session {session_id} has been reset"}
    except Exception as e:
        logger.error(f"Error resetting session {session_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
        
async def text_to_speech(text: str) -> str:
    try:
        # Use Polly to synthesize speech
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",  
            VoiceId="Joanna"    
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(response['AudioStream'].read())
            temp_audio_file_path = temp_audio_file.name

        with open(temp_audio_file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        os.remove(temp_audio_file_path)
        return audio_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text-to-speech conversion: {str(e)}")



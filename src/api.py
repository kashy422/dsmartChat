import tempfile
import os
import time
import logging
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

# Import our essential tools and functionality
from .agent_tools import analyze_symptoms, dynamic_doctor_search, store_patient_details
from .query_builder_agent import unified_doctor_search, unified_doctor_search_tool, extract_search_criteria_tool
from .specialty_matcher import SpecialtyDataCache, get_recommended_specialty

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

app = FastAPI()
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
    audio: UploadFile = File(None)
):
    start_time = time.time()
    
    try:
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
            
                    # Process transcribed message
                    response = process_message(transcription, session_id)
                    
                    # Generate audio response
                audio_base64 = await text_to_speech(response['response']['message'])
                
                return {
                    "response": {
                        "message": response['response']['message'],
                        "transcription": transcription,
                        "Audio_base_64": audio_base64
                    }
                }
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
            # Handle text input
        if message:
                response = process_message(message, session_id)
                return response
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_message(message: str, session_id: str) -> dict:
    """Process incoming message by sending it to the agent"""
    try:
        # Clear any previous session data that might be lingering in thread_local
        if hasattr(thread_local, 'session_id') and thread_local.session_id != session_id:
            logger.info(f"Switching from session {thread_local.session_id} to {session_id}")
            # Log current state before clearing
            log_thread_local_state(logger, thread_local.session_id)
            # Clear previous session data
            for attr in ['symptom_analysis', 'location', 'last_search_results', 'extracted_criteria']:
                if hasattr(thread_local, attr):
                    logger.info(f"Clearing previous {attr} data from thread_local")
                    delattr(thread_local, attr)
        
        # Set current session_id
        thread_local.session_id = session_id
        logger.info(f"Processing message for session {session_id}")
        
        # Log current thread_local state
        log_thread_local_state(logger, session_id)

        # Check if we have a stored symptom analysis for this session
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        location = getattr(thread_local, 'location', None)
            
        # If we have symptom analysis but no location, this might be a location response
        if symptom_analysis and not location and not any(keyword in message.lower() for keyword in ['pain', 'hurt', 'ache', 'symptoms']):
            logger.info(f"Detected location response for previous symptom analysis")
            thread_local.location = message
            # Now we can search for doctors
            try:
                logger.info("DEBUG API: About to access symptom_analysis for doctor search")
                logger.info(f"DEBUG API: symptom_analysis keys: {list(symptom_analysis.keys()) if isinstance(symptom_analysis, dict) else 'Not a dict'}")
            
                # Detailed debugging of the structure
                if isinstance(symptom_analysis, dict) and "symptom_analysis" in symptom_analysis:
                    sa = symptom_analysis.get("symptom_analysis", {})
                    logger.info(f"DEBUG API: symptom_analysis.symptom_analysis keys: {list(sa.keys()) if isinstance(sa, dict) else 'Not a dict'}")
                    
                    if "matched_specialties" in sa:
                        matched = sa["matched_specialties"]
                        logger.info(f"DEBUG API: matched_specialties type: {type(matched)}")
                        logger.info(f"DEBUG API: matched_specialties length: {len(matched) if matched else 0}")
                        if matched and len(matched) > 0:
                            logger.info(f"DEBUG API: First matched specialty: {matched[0]}")
                
                # Safely get specialty information
                matched_specialties = []
                if isinstance(symptom_analysis, dict) and "symptom_analysis" in symptom_analysis:
                    sa = symptom_analysis.get("symptom_analysis", {})
                    if "matched_specialties" in sa and sa["matched_specialties"]:
                        matched_specialties = sa["matched_specialties"]
                    elif "recommended_specialties" in sa and sa["recommended_specialties"]:
                        matched_specialties = sa["recommended_specialties"]
                    elif "specialties" in sa and sa["specialties"]:
                        matched_specialties = sa["specialties"]
                
                # Only try to access the first item if the list is not empty
                specialty_info = None
                if matched_specialties and len(matched_specialties) > 0:
                    specialty_info = matched_specialties[0]
                    logger.info(f"DEBUG API: Using specialty info: {specialty_info}")
                else:
                    logger.warning("DEBUG API: No matched specialties found")
                    # Default to general practitioner if no specialties found
                    specialty_info = {"specialty": "General Practice", "subspecialty": ""}
                
                # Create a string search query that includes location and specialty
                import json
                search_data = {
                    "location": message,
                    "specialty": specialty_info
                }
                # Convert to JSON string for dynamic_doctor_search
                search_query = json.dumps(search_data)
                logger.info(f"DEBUG API: Converted search parameters to JSON string: {search_query}")
                
                # Call with string parameter as expected by the function definition
                doctor_search_result = dynamic_doctor_search(search_query)
                return doctor_search_result
            except Exception as e:
                logger.error(f"DEBUG API: Error accessing symptom_analysis: {str(e)}", exc_info=True)
                # Fallback to a simple location search
                import json
                # Create a simple search with just location
                simple_search = json.dumps({"location": message})
                logger.info(f"DEBUG API: Falling back to simple location search: {simple_search}")
                doctor_search_result = dynamic_doctor_search(simple_search)
                return doctor_search_result

        # Forward the message to the agent and get response
        response = engine.invoke(
            {"input": message, "session_id": session_id},
                config={
                    "configurable": {"session_id": session_id},
                    "callbacks": [callBackHandler]
                }
            )

        # Log thread_local state after processing
        logger.info("Thread local state after processing:")
        log_thread_local_state(logger, session_id)
            
        # If the response contains symptom analysis, store it for the session
        if isinstance(response, dict) and response.get("symptom_analysis"):
            thread_local.symptom_analysis = response
            # Don't return the raw response, let the agent handle it
            return {"response": response}
        
        # If we have a response with doctor data, ensure the message doesn't include doctor details
        if isinstance(response, dict) and isinstance(response.get("response"), dict) and "data" in response["response"]:
            doctor_data = response["response"]["data"]
            # If there are doctors in the data array
            if isinstance(doctor_data, list) and len(doctor_data) > 0:
                # Extract message details
                message = response["response"].get("message", "")
                
                # Check if message contains detailed doctor information
                if "Dr." in message or "doctor" in message.lower() or "clinic" in message.lower() or "branch" in message.lower() or "fee" in message.lower():
                    # Replace with a simple message
                    doctor_count = len(doctor_data)
                    simple_message = f"I found {doctor_count} doctors based on your search."
                    
                    # Try to extract specialty information
                    specialty = None
                    for doc in doctor_data:
                        if doc.get("Speciality"):
                            specialty = doc.get("Speciality")
                            break
                    
                    if specialty:
                        simple_message = f"I found {doctor_count} {specialty} specialists based on your search."
                    
                    # Update the message
                    response["response"]["message"] = simple_message
                    logger.info(f"Simplified doctor search result message: {simple_message}")
        
        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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



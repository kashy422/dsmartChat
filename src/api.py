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
from .utils import CustomCallBackHandler, thread_local
from src.AESEncryptor import AESEncryptor
from typing import Dict, List, Any, Optional

# Import our essential tools and functionality
from .agent_tools import analyze_symptoms, dynamic_doctor_search, store_patient_details
from .query_builder_agent import search_doctors, detect_symptoms_and_specialties
from .specialty_matcher import SpecialtyDataCache

# Import improved logging
from src.utils.logging_utils import setup_improved_logging

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

    # Also print the variant mappings for reference
    from .specialty_matcher import DYNAMIC_SUBSPECIALTY_VARIANT_MAP
    logger.info(f"Created {len(DYNAMIC_SUBSPECIALTY_VARIANT_MAP)} subspecialty variant mappings:")
    for variant, canonical in sorted(DYNAMIC_SUBSPECIALTY_VARIANT_MAP.items())[:10]:  # Show first 10 to avoid log spam
        logger.info(f"  - '{variant}' → '{canonical}'")
    if len(DYNAMIC_SUBSPECIALTY_VARIANT_MAP) > 10:
        logger.info(f"  - ... and {len(DYNAMIC_SUBSPECIALTY_VARIANT_MAP) - 10} more mappings")
except Exception as e:
    logger.error(f"Error preloading specialty data: {str(e)}")
    logger.warning("Application will continue but specialty matching may be affected")

app = FastAPI()
engine = chat_engine()
callBackHandler = CustomCallBackHandler()
encryptor = AESEncryptor("BadoBadiBadoBadi") #dont change the salt
valid_sources = ["www", "wap", "call"]
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
# Set AWS credentials and region dynamically
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_REGION"] = os.getenv("AWS_REGION")

# Create a boto3 session to verify the region
session = boto3.Session()
# print("Default region from boto3 session:", session.region_name)
# Define allowed origins
origins = [
    "http://localhost",
    "http://localhost:8509",
    "https://draide.itsai.tech",
    "https://api.dsmart.ai"     # Example: Add frontend URL
    # Add other origins as needed
]


# Create the AWS Polly client with the region explicitly passed
try:
    polly_client = boto3.client('polly', region_name=os.environ.get("AWS_REGION"))
    print("Polly client initialized successfully!")
except Exception as e:
    print("Error initializing Polly client:", str(e))


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from openai import OpenAI
client = OpenAI()

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
    
    # Split and validate session_id
    try:
        split_text = session_id.split('|')
        if split_text[0] not in valid_sources:
            raise HTTPException(status_code=400, detail="Invalid request type detected")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing session_id: {str(e)}")

    # Check if an audio file is provided
    if audio:
        audio_processing_start = time.time()
        # Log the MIME type for diagnostic purposes
        print(f"Received audio MIME type: {audio.content_type}")

        # Check if the file has the expected .wav extension
        if not audio.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")
        
        # Create a temporary file to store the uploaded WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            contents = await audio.read()  # Read the content of the uploaded file
            temp_file.write(contents)  # Save the file to the temporary location
        
        # Use OpenAI's Whisper model to transcribe the audio file
        try:
            transcription_start = time.time()
            with open(temp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            transcription_time = time.time() - transcription_start
            print(f"Transcription took {transcription_time:.2f}s")
        
            # Process the transcription result
            process_start = time.time()
            response = process_message(transcription, session_id=split_text[1])
            processing_time = time.time() - process_start
            
            # Collect performance metrics
            audio_processing_time = time.time() - audio_processing_start
            total_time = time.time() - start_time
            
            # If callBackHandler has docs_data, return the data and clear it
            if callBackHandler.docs_data:
                response_data = {
                    "response": callBackHandler.docs_data,
                    "performance": {
                        "total_time": round(total_time, 2),
                        "audio_processing_time": round(audio_processing_time, 2),
                        "transcription_time": round(transcription_time, 2),
                        "processing_time": round(processing_time, 2)
                    }
                }
                callBackHandler.docs_data = []  # Clear the data after it's returned
                return response_data
            
            # Generate TTS if needed
            tts_start = time.time()
            audio_base64 = await text_to_speech(response['response']['message'])
            tts_time = time.time() - tts_start
            
            # If no docs_data, return the normal engine response along with audio base64
            return {
                "response": {
                    "message": response['response']['message'],
                    "transcription": transcription,
                    "Audio_base_64": audio_base64
                },
                "performance": {
                    "total_time": round(total_time, 2),
                    "audio_processing_time": round(audio_processing_time, 2),
                    "transcription_time": round(transcription_time, 2),
                    "processing_time": round(processing_time, 2),
                    "tts_time": round(tts_time, 2)
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
        finally:
            # Clean up the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    # Process the request as a text message if no audio file is provided
    if message:
        # Process the message and generate a response
        process_start = time.time()
        response = process_message(message, session_id=split_text[1])
        processing_time = time.time() - process_start
        total_time = time.time() - start_time
        
        # If callBackHandler has docs_data, return the data and clear it
        if callBackHandler.docs_data:
            response_data = {
                "response": callBackHandler.docs_data,
                "performance": {
                    "total_time": round(total_time, 2),
                    "processing_time": round(processing_time, 2)
                }
            }
            callBackHandler.docs_data = []  # Clear the data after it's returned
            return response_data
        
        # Include processing time in the response
        response_with_metrics = response
        response_with_metrics["performance"] = {
            "total_time": round(total_time, 2),
            "processing_time": round(processing_time, 2)
        }
        
        # Return the response as is since it's already in the correct format
        return response_with_metrics
    
    # If neither audio nor message is provided, raise an error
    raise HTTPException(status_code=400, detail="Either a message or an audio file must be provided.")




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
def reset():
    engine.history_messages_key = []


def process_message(message: str, session_id: str) -> dict["str", "str"]:
    thread_local.session_id = session_id
    
    start_time = time.time()  # Start timing
    
    try:
        print(f"Processing message for session {session_id}")
        
        # Time the AI invocation
        invoke_start = time.time()
        response = engine.invoke(
            {"input": message, "session_id": session_id},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [callBackHandler]
            }
        )
        invoke_time = time.time() - invoke_start
        
        # Add debug logging
        print(f"DEBUG: Response from engine.invoke: {response}")
        if isinstance(response, dict):
            print(f"DEBUG: Response keys: {response.keys()}")
            if 'data' in response:
                print(f"DEBUG: Data section keys: {response.get('data', {}).keys()}")
                doctor_count = len(response.get('data', {}).get('doctors', []))
                print(f"DEBUG: Data contains {doctor_count} doctors")
                
                # Make sure doctors data is included in the final response
                if doctor_count > 0:
                    print(f"DEBUG: Found {doctor_count} doctors in the response - ensuring they are included in final response")
            
            # Check if the callback handler has doctor data
            if callBackHandler.docs_data:
                print(f"DEBUG: CallBackHandler has docs_data")
                callback_data = callBackHandler.docs_data
                if isinstance(callback_data, dict) and 'data' in callback_data:
                    callback_doctor_count = len(callback_data.get('data', {}).get('doctors', []))
                    print(f"DEBUG: CallBackHandler docs_data has {callback_doctor_count} doctors")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Log performance metrics
        print(f"Performance metrics - Total: {total_time:.2f}s, AI invoke: {invoke_time:.2f}s")
        
        if '</EXIT>' in str(response):
            engine.history_messages_key = []

        # Create a comprehensive response that ensures doctor data is included
        final_response = {"response": response, "processing_time": total_time}
        
        # Special handling to ensure doctor data is passed through
        if isinstance(response, dict) and 'data' in response and response['data'].get('doctors', []):
            print(f"DEBUG: Ensuring doctor data from response is included in final response")
            # The data is already in response, so it should be passed through
        
        # Check if the callBackHandler has additional data
        elif callBackHandler.docs_data and isinstance(callBackHandler.docs_data, dict):
            print(f"DEBUG: Using doctor data from callBackHandler")
            # Override the response with the callback data which has doctor info
            final_response["response"] = callBackHandler.docs_data
            callBackHandler.docs_data = {}  # Clear after use

        return final_response
    except Exception as e:
        error_time = time.time() - start_time
        print(f"Error in process_message after {error_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


#
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

# New request models for doctor search
class DoctorSearchRequest(BaseModel):
    query: str
    
class SymptomAnalysisRequest(BaseModel):
    description: str

@app.post("/search-doctors")
async def doctor_search_endpoint(request: DoctorSearchRequest):
    """
    Search for doctors based on natural language query.
    
    Example queries:
    - "Show me dentists in Riyadh"
    - "Find cardiologists with 5+ years experience in Jeddah"
    - "Dermatologists with good ratings in Riyadh" 
    """
    try:
        start_time = time.time()
        result = search_doctors(request.query)
        processing_time = time.time() - start_time
        
        # Add performance metrics
        result["performance"] = {
            "processing_time": round(processing_time, 2)
        }
        
        # Make sure we return a clear message when more information is needed
        if result.get('status') == 'needs_more_info':
            # Make the message stand out in the response
            print(f"Need more information for doctor search: {result.get('message')}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for doctors: {str(e)}")

@app.post("/analyze-symptoms")
async def analyze_symptoms_endpoint(request: SymptomAnalysisRequest):
    """
    Analyze symptoms described by the user and recommend appropriate medical specialties.
    
    This endpoint uses the signs and symptoms stored in the database to match the user's
    description to the most relevant medical specialties.
    
    Example input:
    - "I have a toothache and my gums are swollen"
    - "I'm experiencing chest pain and shortness of breath"
    - "My skin has a rash and is itchy"
    """
    start_time = time.time()
    request_id = f"req_{int(time.time())}_{hash(request.description) % 10000}"
    
    logger.info(f"API [/analyze-symptoms] [{request_id}]: New request received")
    logger.info(f"API [/analyze-symptoms] [{request_id}]: Symptom description: '{request.description[:100]}...'")
    
    try:
        # Input validation
        if not request.description or len(request.description.strip()) < 3:
            logger.warning(f"API [/analyze-symptoms] [{request_id}]: Invalid input - Empty or too short description")
            return {
                "status": "error",
                "message": "Please provide a valid symptom description (at least 3 characters)",
                "performance": {
                    "processing_time": round(time.time() - start_time, 2)
                }
            }
        
        # Match symptoms to specialties
        analysis_start = time.time()
        logger.info(f"API [/analyze-symptoms] [{request_id}]: Calling match_symptoms_to_specialties")
        result = match_symptoms_to_specialties(request.description)
        analysis_time = time.time() - analysis_start
        logger.info(f"API [/analyze-symptoms] [{request_id}]: Analysis completed in {analysis_time:.2f}s")
        
        # Log key analysis results
        if result.get("status") == "success":
            detected_symptoms = result.get("detected_symptoms", [])
            recommendations = result.get("recommended_specialties", [])
            
            logger.info(f"API [/analyze-symptoms] [{request_id}]: Analysis successful - "
                        f"Detected {len(detected_symptoms)} symptoms, recommended {len(recommendations)} specialties")
            
            if recommendations:
                top_specialty = recommendations[0]
                logger.info(f"API [/analyze-symptoms] [{request_id}]: Top recommendation - "
                           f"Specialty: {top_specialty.get('specialty')}, "
                           f"Subspecialty: {top_specialty.get('subspecialty', 'None')}, "
                           f"Confidence: {top_specialty.get('confidence', 0):.2f}")
        else:
            logger.error(f"API [/analyze-symptoms] [{request_id}]: Analysis failed - "
                        f"Status: {result.get('status')}, Error: {result.get('error_details', 'Unknown error')}")
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Add performance metrics if not already present
        if "performance" not in result:
            result["performance"] = {}
        
        result["performance"].update({
            "endpoint_processing_time": round(processing_time, 2),
            "analysis_time": round(analysis_time, 2)
        })
        
        # Check if we have any good matches for doctor recommendation
        doctor_recommendation_start = time.time()
        top_specialty = get_recommended_specialty(result, confidence_threshold=0.6)
        doctor_recommendation_time = time.time() - doctor_recommendation_start
        
        if top_specialty:
            logger.info(f"API [/analyze-symptoms] [{request_id}]: Found recommended specialty for doctor search - "
                       f"{top_specialty['specialty']}" + 
                       (f" ({top_specialty['subspecialty']})" if top_specialty.get('subspecialty') else ""))
                       
            result["recommended_doctor_search"] = {
                "specialty": top_specialty["specialty"],
                "subspecialty": top_specialty.get("subspecialty")
            }
        else:
            logger.info(f"API [/analyze-symptoms] [{request_id}]: No specialty met confidence threshold for doctor recommendation")
        
        # Add doctor recommendation timing
        result["performance"]["doctor_recommendation_time"] = round(doctor_recommendation_time, 3)
        
        # Log final result size
        result_size = len(str(result))
        logger.info(f"API [/analyze-symptoms] [{request_id}]: Returning result with size {result_size} bytes, "
                   f"total processing time: {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"API [/analyze-symptoms] [{request_id}]: Unexpected error after {error_time:.2f}s - {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error analyzing symptoms: {str(e)}",
            "error_details": str(e),
            "performance": {
                "processing_time": round(error_time, 2)
            }
        }

class SymptomDebugRequest(BaseModel):
    symptoms: List[str]
    max_results: Optional[int] = 5

@app.post("/debug/symptom-matching")
async def debug_symptom_matching_endpoint(request: SymptomDebugRequest):
    """
    Debug endpoint for testing symptom matching with detailed diagnostics
    
    This endpoint directly accesses the symptom matching functionality without GPT analysis,
    allowing you to see exactly how symptoms match to specialties in the database.
    
    Example input:
    {
        "symptoms": ["headache", "fever", "cough"],
        "max_results": 5
    }
    """
    try:
        from .specialty_matcher import debug_symptom_matching
        
        start_time = time.time()
        request_id = f"dbg_{int(time.time())}_{hash(','.join(request.symptoms)) % 10000}"
        
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Debugging {len(request.symptoms)} symptoms")
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Symptoms: {request.symptoms}")
        
        result = debug_symptom_matching(request.symptoms, request.max_results)
        
        processing_time = time.time() - start_time
        if "performance" not in result:
            result["performance"] = {}
        result["performance"]["endpoint_processing_time"] = round(processing_time, 3)
        
        result_size = len(str(result))
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Completed in {processing_time:.3f}s, returning {result_size} bytes")
        
        return result
        
    except Exception as e:
        logger.error(f"API [/debug/symptom-matching]: Error during debug - {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error during symptom matching debug: {str(e)}",
            "error_details": str(e)
        }

@app.get("/debug/specialty/{specialty_id}")
async def get_specialty_details(specialty_id: int):
    """
    Get detailed information about a specific specialty by ID,
    including raw signs and symptoms data.
    
    This is useful for debugging the symptom matching process
    and verifying that specialties are loaded correctly.
    """
    try:
        from .db import DB
        
        db = DB()
        query = "SELECT * FROM Speciality WHERE ID = :id"
        cursor = db.engine.connect()
        result = cursor.execute(text(query), {"id": specialty_id})
        rows = [dict(row) for row in result.mappings()]
        cursor.close()
        
        if not rows:
            return {
                "status": "error",
                "message": f"No specialty found with ID {specialty_id}"
            }
            
        # Format the response with both raw and parsed data
        specialty = rows[0]
        
        # Parse the signs and symptoms
        raw_signs = specialty.get("Signs", "")
        raw_symptoms = specialty.get("Symptoms", "")
        
        signs = []
        if raw_signs:
            signs = [s.strip().lower() for s in raw_signs.split(',') if s.strip()]
            
        symptoms = []
        if raw_symptoms:
            symptoms = [s.strip().lower() for s in raw_symptoms.split(',') if s.strip()]
        
        # Prepare response
        response = {
            "status": "success",
            "specialty": {
                "id": specialty.get("ID"),
                "name": specialty.get("SpecialityName"),
                "subspecialty": specialty.get("SubSpeciality"),
                "raw_data": {
                    "signs": raw_signs,
                    "symptoms": raw_symptoms
                },
                "parsed_data": {
                    "signs": signs,
                    "symptoms": symptoms,
                    "sign_count": len(signs),
                    "symptom_count": len(symptoms)
                },
                "parsing_validation": {
                    "signs_comma_count": raw_signs.count(',') if raw_signs else 0,
                    "symptoms_comma_count": raw_symptoms.count(',') if raw_symptoms else 0,
                    "expected_sign_count": raw_signs.count(',') + 1 if raw_signs else 0,
                    "expected_symptom_count": raw_symptoms.count(',') + 1 if raw_symptoms else 0
                }
            }
        }
        
        # Add a consistency check
        if raw_signs:
            expected_sign_count = raw_signs.count(',') + 1 if raw_signs else 0
            response["specialty"]["parsing_validation"]["signs_consistent"] = (
                expected_sign_count == len(signs)
            )
            
        if raw_symptoms:
            expected_symptom_count = raw_symptoms.count(',') + 1 if raw_symptoms else 0
            response["specialty"]["parsing_validation"]["symptoms_consistent"] = (
                expected_symptom_count == len(symptoms)
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving specialty details: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error retrieving specialty details: {str(e)}"
        }

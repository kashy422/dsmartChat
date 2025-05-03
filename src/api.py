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
from .utils import CustomCallBackHandler, thread_local, setup_improved_logging
from src.AESEncryptor import AESEncryptor
from typing import Dict, List, Any, Optional

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
        
        # Check if this is a doctor search result with data field
        if isinstance(response, dict) and 'data' in response and 'doctors' in response.get('data', {}):
            print(f"DEBUG: Detected direct doctor search result with {len(response['data']['doctors'])} doctors")
            
            # Special handling for doctor data to ensure proper format separation
            doctors_data = response['data']['doctors']
            doctor_count = len(doctors_data)
            
            # Extract specialty information
            specialty = "doctors"
            if 'data' in response and 'criteria' in response['data'] and 'speciality' in response['data'].get('criteria', {}):
                specialty = response['data']['criteria']['speciality']
                
            # Extract location information
            location = ""
            if 'data' in response and 'criteria' in response['data'] and 'location' in response['data'].get('criteria', {}):
                location = response['data']['criteria']['location']
                location_text = f" in {location}"
            else:
                location_text = ""
            
            # Replace any detailed doctor description in the message with a simple summary
            # This ensures proper separation between UI components
            clean_message = f"I found {doctor_count} {specialty} specialists{location_text} that match your criteria."
            
            # Check if the message has detailed doctor information and replace it
            current_message = response.get('message', '')
            if current_message and (
                "rating" in current_message.lower() or 
                "fee" in current_message.lower() or
                "clinic" in current_message.lower() or
                "**" in current_message or
                "-" in current_message or
                any(f"{i}." in current_message for i in range(1, 10))
            ):
                print(f"DEBUG: Replacing detailed doctor information in message with clean summary")
                response['message'] = clean_message
                print(f"DEBUG: Clean message: {clean_message}")
            elif not current_message:
                response['message'] = clean_message
                
            # Update the conversation history in the engine to maintain context
            engine.invoke(
                {
                    "input": "__update_history__", 
                    "session_id": session_id, 
                    "message": response['message']
                },
                config={
                    "configurable": {"session_id": session_id},
                    "callbacks": [callBackHandler]
                }
            )
            print(f"DEBUG: Updated conversation history with message: '{response['message']}'")
            
            # Ensure standard format with status field
            if 'status' not in response:
                response['status'] = 'success'
            
            # Add processing time
            response['processing_time'] = time.time() - start_time
            
            # Log the final doctor search result structure 
            print(f"DEBUG: Returning doctor search result with structure: {list(response.keys())}")
            print(f"DEBUG: Data field contains: {list(response['data'].keys()) if 'data' in response else 'None'}")
            
            return response
        
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
    """
    Process a user message and return an AI response.
    
    Args:
        message: The user message to process
        session_id: The session ID for the conversation
        
    Returns:
        A dictionary containing the AI response
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Log the incoming message and session
        thread_local.session_id = session_id
        logger.info(f"Processing message: '{message}' for session {session_id}")
        
        # Check for direct doctor search queries
        if any(term in message.lower() for term in ["find doctor", "find a doctor", "doctor near", "looking for doctor"]):
            logger.info(f"Direct doctor search detected: '{message}'")
            search_result = dynamic_doctor_search({"user_message": message})
            error_time = time.time() - start_time
            logger.info(f"Doctor search completed in {error_time:.2f}s")
            return search_result
        
        # Check for direct symptom analysis
        if "what could" in message.lower() and "symptom" in message.lower():
            logger.info(f"Direct symptom analysis detected: '{message}'")
            symptom_result = analyze_symptoms({"symptom_description": message})
            error_time = time.time() - start_time
            logger.info(f"Symptom analysis completed in {error_time:.2f}s")
            return {"response": {"message": symptom_result["message"]}}
        
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
        
        error_time = time.time() - start_time
        logger.info(f"Message processed in {error_time:.2f}s")
        return {"response": response}
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        print(f"Error in process_message after {error_time:.2f}s: {str(e)}")
        
        # Provide a fallback response
        return {"response": {"message": "I'm sorry, I encountered an issue processing your request. Please try again."}}


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
        result = unified_doctor_search(request.query)
        processing_time = time.time() - start_time
        
        # Add performance metrics to the data object
        if "data" not in result:
            result["data"] = {}
        
        # Add performance data to the data object
        result["data"]["performance"] = {
            "processing_time": round(processing_time, 2)
        }
        
        # Make sure we return a clear message when more information is needed
        if result.get('status') == 'needs_more_info':
            # Make the message stand out in the response
            print(f"Need more information for doctor search: {result.get('message')}")
        
        return result
    except Exception as e:
        # Return error in the new format
        return {
            "status": "error",
            "message": f"Error searching for doctors: {str(e)}",
            "data": {
                "count": 0,
                "doctors": []
            }
        }

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
        logger.info(f"API [/analyze-symptoms] [{request_id}]: Calling unified_doctor_search")
        result = unified_doctor_search(request.description)
        analysis_time = time.time() - analysis_start
        logger.info(f"API [/analyze-symptoms] [{request_id}]: Analysis completed in {analysis_time:.2f}s")
        
        # Log key analysis results
        if result.get("status") == "success":
            # If using the new format, extract symptom analysis from the nested structure
            symptom_analysis = result.get("symptom_analysis", {})
            detected_symptoms = symptom_analysis.get("detected_symptoms", [])
            recommendations = symptom_analysis.get("recommended_specialties", [])
            
            logger.info(f"API [/analyze-symptoms] [{request_id}]: Analysis successful - "
                        f"Detected {len(detected_symptoms)} symptoms, recommended {len(recommendations)} specialties")
            
            if recommendations:
                top_specialty = recommendations[0]
                logger.info(f"API [/analyze-symptoms] [{request_id}]: Top recommendation - "
                           f"Specialty: {top_specialty.get('name')}, "
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
    Debug endpoint for testing symptom matching with detailed diagnostics.
    Now provides a simplified version that uses the main unified_doctor_search function.
    """
    try:
        start_time = time.time()
        request_id = f"dbg_{int(time.time())}_{hash(','.join(request.symptoms)) % 10000}"
        
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Analyzing {len(request.symptoms)} symptoms")
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Symptoms: {request.symptoms}")
        
        # Combine symptoms into a single description
        combined_symptoms = ", ".join(request.symptoms)
        
        # Call the unified function to analyze symptoms
        analysis = unified_doctor_search(combined_symptoms)
        
        # Format the response
        if analysis.get("is_describing_symptoms", False):
            result = {
                "status": "success",
                "summary": {
                    "total_symptoms": len(request.symptoms),
                    "is_symptom_description": True,
                },
                "recommended_specialties": analysis.get("specialties", [])[:request.max_results],
                "symptom_analysis": analysis.get("symptom_analysis", {})
            }
        else:
            result = {
                "status": "no_symptoms",
                "summary": {
                    "total_symptoms": len(request.symptoms),
                    "is_symptom_description": False,
                },
                "message": "The input doesn't appear to be describing symptoms"
            }
        
        processing_time = time.time() - start_time
        result["performance"] = {"endpoint_processing_time": round(processing_time, 3)}
        
        logger.info(f"API [/debug/symptom-matching] [{request_id}]: Completed in {processing_time:.3f}s")
        
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

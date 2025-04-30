import tempfile
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
import boto3
from fastapi.middleware.cors import CORSMiddleware
import base64
from .agent import chat_engine
from .utils import CustomCallBackHandler
from src.AESEncryptor import AESEncryptor
from .utils import CustomCallBackHandler, thread_local  # Import thread_local from utils.py

# Import our query builder functionality
from .query_builder_agent import search_doctors, detect_symptoms_and_specialties

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
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Log performance metrics
        print(f"Performance metrics - Total: {total_time:.2f}s, AI invoke: {invoke_time:.2f}s")
        
        if '</EXIT>' in str(response):
            engine.history_messages_key = []

        return {"response": response, "processing_time": total_time}
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
    symptoms: str

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
async def symptom_analysis_endpoint(request: SymptomAnalysisRequest):
    """
    Analyze symptoms and recommend appropriate medical specialties.
    
    Example input:
    - "I have a persistent toothache and my gums are swollen"
    - "I'm experiencing chest pain and shortness of breath"
    """
    try:
        start_time = time.time()
        result = detect_symptoms_and_specialties(request.symptoms)
        processing_time = time.time() - start_time
        
        # Add performance metrics
        result["performance"] = {
            "processing_time": round(processing_time, 2)
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")

import os
import time
import json
import logging
import traceback
from typing import Dict, List, Any
from sqlalchemy import text
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    logger.warning("No OpenAI API key found in environment variables. Set either OPENAI_API_KEY or API_KEY.")

client = OpenAI(api_key=api_key)

class SpecialtyDataCache:
    """
    Singleton class to cache specialty data from the database.
    This prevents repeated database queries for the same data.
    """
    _instance = None
    _data = None
    _last_update = None
    
    @classmethod
    def get_instance(cls) -> List[Dict[str, Any]]:
        """
        Get specialty data from cache or load it from database if not available
        
        Returns:
            List of specialty data dictionaries
        """
        # Check if we need to create or refresh the instance
        if cls._instance is None or cls._data is None or cls._should_refresh():
            cls._instance = cls()
            cls._load_data()
            
        return cls._data
    
    @classmethod
    def _should_refresh(cls) -> bool:
        """
        Check if data should be refreshed (older than 1 hour)
        
        Returns:
            Boolean indicating if data should be refreshed
        """
        if cls._last_update is None:
            return True
            
        # Refresh data if it's older than 1 hour
        return time.time() - cls._last_update > 3600  # 1 hour
    
    @classmethod
    def _load_data(cls) -> None:
        """
        Load specialty data from database
        """
        try:
            from .db import DB
            db = DB()
            
            # Query to load specialty data with signs and symptoms
            query = """
            SELECT 
                s.ID,
                s.SpecialityName as specialty,
                s.SubSpeciality as subspecialty,
                s.Signs as signs,
                s.Symptoms as symptoms
            FROM 
                dbo.Speciality s
            """
            
            # Execute query
            cursor = db.engine.connect()
            logger.info(f"DB Engine connected: {db.engine.url}")
            result = cursor.execute(text(query))
            rows = [dict(row) for row in result.mappings()]
            cursor.close()
            
            # Process the results
            cls._data = []
            for row in rows:
                # Convert signs and symptoms from comma-separated strings to lists
                signs = []
                symptoms = []
                
                if row.get("signs"):
                    signs = [s.strip() for s in row["signs"].split(",") if s.strip()]
                    
                if row.get("symptoms"):
                    symptoms = [s.strip() for s in row["symptoms"].split(",") if s.strip()]
                
                formatted_row = {
                    "specialty": row.get("specialty", ""),
                    "subspecialty": row.get("subspecialty", ""),
                    "signs": signs,
                    "symptoms": symptoms
                }
                
                cls._data.append(formatted_row)
                
            # Update the last update timestamp
            cls._last_update = time.time()
            logger.info(f"Loaded {len(cls._data)} specialty records from database")
            
        except Exception as e:
            logger.error(f"Error loading specialty data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Initialize with empty data if loading fails
            cls._data = []
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the cache to force a reload on next access
        """
        cls._instance = None
        cls._data = None
        cls._last_update = None

def detect_symptoms_and_specialties(user_message: str) -> Dict[str, Any]:
    """
    Unified function that performs both symptom detection and specialty/subspecialty matching
    in a single GPT call.
    
    Args:
        user_message: User's message which may or may not describe symptoms
        
    Returns:
        Dictionary with analysis including:
        - is_describing_symptoms: Whether the message is describing symptoms
        - specialties: List of matched specialties if symptoms are detected
        - status: Success, no_symptoms, or error
    """
    try:
        # Skip processing for very short messages
        if not user_message or len(user_message.strip()) < 3:
            logger.info("SYMPTOM ANALYZER: Message too short, skipping analysis")
            return {
                "status": "no_symptoms",
                "is_describing_symptoms": False,
                "message": "Message too short to analyze"
            }
        
        # Quick check for common greeting words - avoid unnecessary API calls
        common_greetings = ["hammad", "hello", "hi", "hey", "salam", "marhaba", "ahlan"]
        message_lower = user_message.lower().strip()
        
        # If the message is just a greeting or contains only greeting words, return immediately
        if message_lower in common_greetings or (
            any(greeting in message_lower for greeting in common_greetings) and 
            len(message_lower.split()) <= 3
        ):
            logger.info(f"SYMPTOM ANALYZER: Detected greeting message, skipping analysis: '{message_lower}'")
            return {
                "status": "no_symptoms",
                "is_describing_symptoms": False,
                "message": "Message appears to be a greeting, not symptom description"
            }
            
        logger.info(f"SYMPTOM ANALYZER: Analyzing message: '{user_message[:50]}...'")
        start_time = time.time()
        
        # Get specialty data from database cache
        try:
            specialty_data = SpecialtyDataCache.get_instance()
            logger.info(f"SYMPTOM ANALYZER: Using {len(specialty_data)} specialty records from database cache")
        except Exception as db_err:
            logger.error(f"SYMPTOM ANALYZER: Database error: {str(db_err)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "is_describing_symptoms": False,
                "message": "Error accessing medical specialty database. Please try again.",
                "error_details": str(db_err)
            }
        
        # Create system prompt for GPT with specialty data
        system_prompt = f"""
        You are a medical assistant specializing in symptom analysis and specialist referrals.
        
        TASK:
        1. FIRST determine if the user's message is describing medical symptoms or health concerns.
        2. If it IS describing symptoms, match these symptoms to appropriate specialties from the database.
        3. If it is NOT describing symptoms, respond accordingly.
        
        IMPORTANT NOTES:
        - Common greeting phrases like "hello", "hi", "hey", "salam", "marhaba", etc. are NOT symptom descriptions
        - Very short messages with just greetings should be classified as NOT describing symptoms
        
        SPECIALTY DATABASE:
        Use ONLY the following specialty data loaded from our medical database:
        {specialty_data}
        
        ANALYSIS STEPS (only if message describes symptoms):
        1. Identify all symptoms and health concerns in the user's message
        2. Match these symptoms to the most appropriate medical specialties in the database
        3. For each match, determine the level of confidence (0.0-1.0)
        4. Prioritize specialties that directly address the primary symptoms
        
        REQUIREMENTS:
        - ONLY recommend specialties and subspecialties that are listed in the provided database
        - Use EXACT names of specialties and subspecialties as they appear in the database
        - Do NOT invent or suggest specialties not in the database
        - Assign realistic confidence levels (higher for clearer symptom matches)
        
        RESPONSE FORMAT:
        Return a JSON object with these fields:
        - is_describing_symptoms: boolean (true if message describes symptoms, false otherwise)
        
        If is_describing_symptoms is true, also include:
        - detected_symptoms: List of all identified symptoms and health concerns
        - recommended_specialties: Array of objects, each containing:
          * name: The specialty name exactly as it appears in the database
          * subspecialty: The subspecialty name exactly as it appears in the database
          * confidence: Number between 0.0 and 1.0 indicating match confidence
        - specialty_explanation: Brief explanation of the specialty recommendations
        """
        
        # Call GPT to analyze the message
        logger.info(f"SYMPTOM ANALYZER: Calling GPT for unified symptom detection and analysis")
        gpt_start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
            timeout=15
        )
        
        gpt_time = time.time() - gpt_start_time
        logger.info(f"SYMPTOM ANALYZER: GPT response received in {gpt_time:.2f}s")
        
        # Parse the GPT response
        result_json = response.choices[0].message.content
        logger.info(f"SYMPTOM ANALYZER: Processing GPT response")
        
        result = json.loads(result_json)
        
        # Check if the message is describing symptoms
        is_describing_symptoms = result.get("is_describing_symptoms", False)
        logger.info(f"SYMPTOM ANALYZER: Message {'' if is_describing_symptoms else 'is not '}describing symptoms")
        
        if not is_describing_symptoms:
            # If not describing symptoms, return early
            return {
                "status": "no_symptoms",
                "is_describing_symptoms": False,
                "message": "Message does not appear to be describing symptoms"
            }
            
        # If describing symptoms, extract and log detected symptoms
        detected_symptoms = result.get("detected_symptoms", [])
        if detected_symptoms:
            logger.info(f"SYMPTOM ANALYZER: Detected symptoms: {', '.join(detected_symptoms)}")
            
        # Extract specialty information
        specialties = []
        if "recommended_specialties" in result:
            for specialty in result["recommended_specialties"]:
                specialty_name = specialty.get("name", "")
                subspecialty_name = specialty.get("subspecialty", "")
                confidence = specialty.get("confidence", 0.8)
                
                specialty_item = {
                    "specialty": specialty_name,
                    "subspecialty": subspecialty_name,
                    "confidence": confidence
                }
                specialties.append(specialty_item)
                
                # Log each specialty match with confidence score
                logger.info(f"SYMPTOM ANALYZER: Matched specialty: {specialty_name}/{subspecialty_name} (confidence: {confidence:.2f})")
        
        # Log if no specialties were found
        if not specialties:
            logger.warning("SYMPTOM ANALYZER: No matching specialties found for the detected symptoms")
        
        # Create final result
        analysis_result = {
            "status": "success",
            "is_describing_symptoms": True,
            "symptom_analysis": result,
            "specialties": specialties,
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
        
        logger.info(f"SYMPTOM ANALYZER: Analysis completed in {time.time() - start_time:.2f}s")
        return analysis_result
        
    except Exception as e:
        logger.error(f"SYMPTOM ANALYZER: Error in symptom analysis: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "is_describing_symptoms": False,
            "message": "Could not analyze message. Please try again.",
            "error_details": str(e)
        }

# For backward compatibility with API endpoints that expect this function
def get_recommended_specialty(symptom_description_or_analysis, confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Compatibility function that returns recommended specialty based on symptom description or analysis.
    This is kept to support existing API endpoints.
    
    Args:
        symptom_description_or_analysis: Either a string description or an existing analysis dict
        confidence_threshold: Minimum confidence level for recommendations
        
    Returns:
        Recommended specialty information
    """
    # Check if input is already an analysis result
    if isinstance(symptom_description_or_analysis, dict):
        analysis = symptom_description_or_analysis
    else:
        # Call the unified function to analyze symptoms
        analysis = detect_symptoms_and_specialties(symptom_description_or_analysis)
    
    # Get specialties
    specialties = analysis.get("specialties", [])
    valid_specialties = [s for s in specialties if s.get("confidence", 0) >= confidence_threshold]
    
    if not valid_specialties:
        return None
    
    # Get the top recommendation
    return valid_specialties[0] 